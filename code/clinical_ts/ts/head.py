__all__ = ['RNNHead', 'RNNHeadConfig', 'PoolingHead', 'PoolingHeadConfig', 'MLPHead', 'MLPHeadConfig', 'MLPRegressionHead', 'MLPRegressionHeadConfig', 'S4Head', 'S4HeadConfig','AttentionPoolingHead', 'AttentionPoolingHeadConfig', 'LearnableQueryAttentionPoolingHead', 'LearnableQueryAttentionPoolingHeadConfig', 'TransformerHeadGlobal', 'TransformerHeadGlobalConfig', 'TransformerHeadMulti', 'TransformerHeadMultiConfig', 'FlattenHead', 'FlattenHeadConfig', 'SequentialHead', 'SequentialHeadConfig', 'FlattenMLPHeadConfig', 'PoolingFlattenHeadConfig', 'PoolingFlattenMLPHeadConfig']

import torch
import torch.nn as nn
import numpy as np

from collections.abc import Iterable
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin
from clinical_ts.template_modules import HeadBase, HeadBaseConfig, _string_to_class
from clinical_ts.ts.s4_modules.s4_model import S4Model

import dataclasses
from dataclasses import dataclass, field
from typing import List

class RNNHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        self.batch_first = hparams_head.batch_first
        self.local_pool = hparams_head.multi_prediction
        
        if(self.local_pool):#local pool
            self.local_pool_padding = (hparams_head.local_pool_kernel_size-1)//2
            self.local_pool_kernel_size = hparams_head.local_pool_kernel_size
            self.local_pool_stride = self.local_pool_kernel_size if hparams_head.local_pool_stride==0 else hparams_head.local_pool_stride
            
            if(hparams_head.local_pool_max):
                self.pool = torch.nn.MaxPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)
            else:
                self.pool = torch.nn.AvgPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)        
        else:#global pool
            if(hparams_head.concat_pool):
                self.pool = AdaptiveConcatPoolRNN(bidirectional=not(hparams_head.causal),cls_first= False)
            else:
                self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten()) #None

        #classifier
        output_dim = hparams_input_shape.channels
                
        nf = 3*output_dim if (hparams_head.multi_prediction is False and hparams_head.concat_pool) else output_dim

        #concatenate static input
        self.static_input = hparams_head.static_input
        if(self.static_input):
            nf = nf + hparams_input_shape.static_dim
                
        lin_ftrs = [nf, target_dim] if hparams_head.lin_ftrs is None else [nf] + hparams_head.lin_ftrs + [target_dim]
        ps_head = [hparams_head.dropout] if not isinstance(hparams_head.dropout, Iterable) else hparams_head.dropout
        if len(ps_head)==1:
            ps_head = [ps_head[0]/2] * (len(lin_ftrs)-2) + ps_head
        actns = [nn.ReLU(inplace=False)] * (len(lin_ftrs)-2) + [None]

        layers_head =[]
        for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps_head,actns):
            layers_head+=bn_drop_lin(ni,no,hparams_head.batch_norm,p,actn,layer_norm=False,permute=self.local_pool)
        self.head=nn.Sequential(*layers_head)

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = int(np.floor((hparams_input_shape.length + 2*self.local_pool_padding- self.local_pool_kernel_size)/self.local_pool_stride+1)) if self.local_pool else 0

    def forward(self,**kwargs):
        seq = kwargs["seq"]
        if(self.batch_first):#B,S,E
            seq = seq.transpose(1,2) 
        else:#S,B,E
            seq = seq.transpose(0,1).transpose(1,2)
        seq = self.pool(seq) if self.pool is not None else seq[:,:,-1] #local_pool: B, E, S global_pool: B, E
        if(self.local_pool):
            seq = seq.transpose(1,2)#B,S,E for local_pool
        if("static" in kwargs.keys() and kwargs["static"] is not None and self.static_input):
            if(self.local_pool):
                seq = torch.cat([seq,kwargs["static"].unsqueeze(1).repeat(1,seq.shape[1],1)],dim=1)
            else:
                seq = torch.cat([seq,kwargs["static"]],dim=1)
        return {"seq": self.head(seq)} #B,S,Nc for local_pool B,Nc for global_pool
    
    def get_output_shape(self):
        return self.output_shape


@dataclass
class RNNHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.RNNHead"
    batch_first:bool = True

    causal:bool = True #has to match causal flag of the predictor model
    multi_prediction:bool = False
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    #local_pool_padding= (kernel_size-1)//2

    concat_pool:bool = True
    dropout:float=0.5
    lin_ftrs:List[int]=field(default_factory=lambda: []) #help="Classification head hidden units (space-separated)")
    batch_norm:bool = True
    static_input:bool = True

#copied from RNN1d
class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False, cls_first=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.cls_first = cls_first

    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if(self.bidirectional is False):
            if(self.cls_first):
                t3 = x[:,:,0]
            else:
                t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out
    
class PoolingHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        #assert(target_dim is None or hparams_head.output_layer is True)
        if(target_dim is not None and hparams_head.output_layer is False):
            print("Warning: target_dim",target_dim,"is passed to PoolingHead but output_layer is False. target_dim will be ignored.")
        self.local_pool = hparams_head.multi_prediction
        self.output_dim = hparams_input_shape.channels if not hparams_head.output_layer else target_dim
        
        if(self.local_pool):#local pool
            self.local_pool_padding = (hparams_head.local_pool_kernel_size-1)//2
            self.local_pool_kernel_size = hparams_head.local_pool_kernel_size
            self.local_pool_stride = hparams_head.local_pool_kernel_size if hparams_head.local_pool_stride==0 else hparams_head.local_pool_stride
            if(hparams_head.local_pool_max):
                self.pool = torch.nn.MaxPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)
            else:
                self.pool = torch.nn.AvgPool1d(kernel_size=hparams_head.local_pool_kernel_size,stride=hparams_head.local_pool_stride if hparams_head.local_pool_stride!=0 else hparams_head.local_pool_kernel_size,padding=(hparams_head.local_pool_kernel_size-1)//2)        
        else:#global pool
            if(hparams_head.local_pool_max):
                self.pool = torch.nn.AdaptiveMaxPool1d(1)
            else:
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hparams_input_shape.channels, target_dim) if hparams_head.output_layer else nn.Identity()

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.output_dim
        #assert(hparams.predictor._target_!="clinical_ts.ts.transformer.TransformerPredictor" or (hparams.predictor.cls_token is True or (hparams.predictor.cls_token is False and (hparams_head.head_pooling_type!="cls" and hparams_head.head_pooling_type!="meanmax-cls"))))

        self.output_shape.length = int(np.floor((hparams_input_shape.length + 2*self.local_pool_padding- self.local_pool_kernel_size)/self.local_pool_stride+1)) if self.local_pool else 0

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        #input has shape B,S,E
        seq = seq.transpose(1,2) 
        seq = self.pool(seq)
        return {"seq": self.linear(seq.transpose(1,2))}#return B,S,E
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class PoolingHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.PoolingHead"
    
    multi_prediction:bool = False #local pool vs. global pool
    local_pool_max:bool = False #max pool vs. avg pool
    local_pool_kernel_size: int = 0 #kernel size for local pooling
    local_pool_stride: int = 0 #kernel_size if 0
    #local_pool_padding=(kernel_size-1)//2
    output_layer: bool = False

class Transpose(nn.Module):
    '''helper module: transpose operation that can be used like a nn.module'''
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class MLPHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        self.multi_prediction = hparams_head.multi_prediction
        self.heads = hparams_head.heads

        proj_list = []
        
        target_dims = target_dim if isinstance(target_dim, list) else [target_dim]*max(self.heads,1)
        for i in range(max(self.heads,1)):
        
            proj = []
            if(hparams_input_shape.length>0 and self.multi_prediction is False):#leave out pool and flatten if sequence is already pooled
                proj += [Transpose(1,2),torch.nn.AdaptiveAvgPool1d(1),nn.Flatten()] #transpose to bring input into expected format B,F,S for pooling

            if(hparams_head.mlp):# additional hidden layer as in simclr
                proj += [nn.Linear(hparams_input_shape.channels, hparams_input_shape.channels),nn.ReLU(inplace=True),nn.Linear(hparams_input_shape.channels, target_dims[i],bias=hparams_head.bias)]
            else:
                proj += [nn.Linear(hparams_input_shape.channels, target_dims[i],bias=hparams_head.bias)]
            proj_list.append(nn.Sequential(*proj))
        self.proj = proj_list[0] if self.heads==0 else nn.ModuleList(proj_list)

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dims[0]#these are the output channels of the first head
        self.output_shape.length = self.output_shape.length if self.multi_prediction else 0
        
    def forward(self, **kwargs):

        return {"seq": self.proj(kwargs["seq"]) if self.heads==0 else [self.proj[i](kwargs["seq"]) for i in range(self.heads)]}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class MLPHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.MLPHead"
    multi_prediction: bool = True # sequence level prediction
    
    mlp: bool = False #mlp in prediction head
    bias: bool = True  #bias for final projection in prediction head
    heads: int = 0 # also support multiple heads (returned as list for HuBERT etc) 0 means single head but won't return a list

class MLPRegressionHead(MLPHead):
    '''MLP head with restricted output range for regression problems'''
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        assert(len(hparams_head.max_val)==0 or (target_dim % len(hparams_head.max_val) == 0))
        assert(len(hparams_head.min_val)==0 or (target_dim % len(hparams_head.min_val) == 0))
        
        max_val_np = np.ones(target_dim, dtype=np.float32) if len(hparams_head.max_val)==0 else np.repeat(np.expand_dims(np.array(hparams_head.max_val,dtype=np.float32),axis=1),target_dim // len(hparams_head.max_val),axis=1).reshape(-1)
        min_val_np = np.zeros(target_dim, dtype=np.float32) if len(hparams_head.min_val)==0 else np.repeat(np.expand_dims(np.array(hparams_head.min_val,dtype=np.float32),axis=1),target_dim // len(hparams_head.min_val),axis=1).reshape(-1)
        
        self.register_buffer("max_val",torch.from_numpy(max_val_np),persistent=True)
        self.register_buffer("min_val",torch.from_numpy(min_val_np),persistent=True)
    
    def forward(self, **kwargs):
        #the first line should stay consistent with MLPHead
        #output shape is B,S,E or B,E
        seq = self.proj(kwargs["seq"])
        return {"seq": self.min_val+torch.sigmoid(seq)*(self.max_val-self.min_val)}
    
@dataclass
class MLPRegressionHeadConfig(MLPHeadConfig):
    _target_:str = "clinical_ts.ts.head.MLPRegressionHead"
    max_val:List[float]=field(default_factory=lambda: [])#max values per output dim (1 by default)
    min_val:List[float]=field(default_factory=lambda: [])#max values per output dim (0 by default)  

class S4Head(HeadBase):

    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''S4 analogue of the pretraining head used in modified CPC https://arxiv.org/abs/2002.02848 (in addition to layer norm) can also be used as global prediction head'''
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        self.s4 = S4Model(
            d_input = None,#matches output dim of the encoder
            d_output = target_dim,
            d_state = hparams_head.state_dim,
            d_model = hparams_input_shape.channels,
            n_layers = 1,
            #dropout = hparams_predictor.dropout, #use default
            #prenorm = hparams_predictor.prenorm, #
            l_max = hparams_input_shape.length,
            transposed_input = False,
            bidirectional=not(hparams_head.causal),
            layer_norm=not(hparams_head.batchnorm),
            pooling = not(hparams_head.multi_prediction),
            backbone = hparams_head.backbone)
        
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = hparams_input_shape.length if hparams_head.multi_prediction else 0

    def forward(self, **kwargs):
        return {"seq": self.s4(kwargs["seq"])}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class S4HeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.S4Head"
    multi_prediction: bool = True # sequence level prediction or not

    state_dim:int = 64
    dropout:float=0.2
    prenorm:bool=False
    batchnorm:bool=False
    backbone:str="s42" #help="s4original/s4new/s4d")  

    causal:bool= True #causal layer (e.g. for CPC)

class FlattenHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        assert(hparams_head.multi_prediction is False)
        assert(target_dim is None or hparams_head.output_layer is True)

        channels_after_flatten = hparams_input_shape.channels*hparams_input_shape.length
        self.linear = nn.Linear(channels_after_flatten, target_dim) if hparams_head.output_layer else nn.Identity()

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim if hparams_head.output_layer else channels_after_flatten
        self.output_shape.length = 0

    def forward(self, **kwargs):   
        #input shape is B,S,E
        return {"seq": kwargs["seq"].view(kwargs["seq"].size(0),-1)}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class FlattenHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.FlattenHead"
    multi_prediction:bool=False
    output_layer:bool=False

class SequentialHead(HeadBase):
    '''A sequence of heads (somewhat similar to nn.Sequential). For config parsing reasons only up to three heads head{0,1,2} are allowed
    '''
    def __init__(self, hparams_head, hparams_input_shape, target_dim=None):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        heads = []
        lst_hparams_head = [l for l in [hparams_head.head0, hparams_head.head1, hparams_head.head2] if l._target_!=""]
        assert(len(lst_hparams_head)>0)

        for i,hparams in enumerate(lst_hparams_head):
            heads.append(_string_to_class(hparams._target_)(hparams, hparams_input_shape, None if i!=len(lst_hparams_head)-1 else target_dim))
            hparams_input_shape = heads[-1].get_output_shape()
        self.models = nn.ModuleList(heads)

        self.output_shape = hparams_input_shape

    def forward(self, **kwargs):
        current = kwargs.copy()
        for m in self.models:
            res = m(**current)
            current.update(res)
        return res
    
    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        txt = self.__class__.__name__ + "\t" +str(self.get_output_shape())
        txt+="\n["
        for i,m in enumerate(self.models):
            txt+="\n-"+str(i)+":"+str(m)
        return txt+"\n]"      

@dataclass
class SequentialHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.SequentialHead"
    
    head0: HeadBaseConfig = field(default_factory=HeadBaseConfig)
    head1: HeadBaseConfig = field(default_factory=HeadBaseConfig)
    head2: HeadBaseConfig = field(default_factory=HeadBaseConfig) #not specifying heads (i.e. keeping default values) implies ignoring them


#specific instantiations of SequentialHeadConfig
@dataclass
class FlattenMLPHeadConfig(SequentialHeadConfig):
    head0: FlattenHeadConfig = field(default_factory=FlattenHeadConfig)
    head1: MLPHeadConfig = field(default_factory=MLPHeadConfig)

@dataclass
class PoolingFlattenHeadConfig(SequentialHeadConfig):
    head0: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)#typically a local pooling
    head1: FlattenHeadConfig = field(default_factory=FlattenHeadConfig)

@dataclass
class PoolingFlattenMLPHeadConfig(SequentialHeadConfig):
    head0: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)#typically a local pooling
    head1: FlattenHeadConfig = field(default_factory=FlattenHeadConfig)
    head2: MLPHeadConfig = field(default_factory=MLPHeadConfig)

#Note: PoolingConcatFusionHeadConfig in ..head.multimodal.py