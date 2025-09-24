import torch
from torch import nn
import numpy as np

import dataclasses
from dataclasses import dataclass, field
from typing import List

from clinical_ts.template_modules import EncoderBase, EncoderBaseConfig
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import _conv1d

class RNNEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape, static_stats_train):
        '''RNN Encoder is actually just a conv encoder'''
        super().__init__(hparams_encoder, hparams_input_shape, static_stats_train)
        assert(len(hparams_encoder.strides)==len(hparams_encoder.kss) and len(hparams_encoder.strides)==len(hparams_encoder.features) and len(hparams_encoder.strides)==len(hparams_encoder.dilations))
        lst = []
        for i,(s,k,f,d) in enumerate(zip(hparams_encoder.strides,hparams_encoder.kss,hparams_encoder.features,hparams_encoder.dilations)):
            lst.append(_conv1d((hparams_input_shape.channels*hparams_input_shape.channels2 if hparams_input_shape.channels2>0 else hparams_input_shape.channels) if i==0 else hparams_encoder.features[i-1],f,kernel_size=k,stride=s,dilation=d,bn=hparams_encoder.normalization,layer_norm=hparams_encoder.layer_norm))
            if(hparams_encoder.multi_prediction and i==0):#local pool after first conv
                if(hparams_encoder.local_pool_max):
                    lst.append(torch.nn.MaxPool1d(kernel_size=hparams_encoder.local_pool_kernel_size,stride=hparams_encoder.local_pool_stride if hparams_encoder.local_pool_stride!=0 else hparams_encoder.local_pool_kernel_size,padding=(hparams_encoder.local_pool_kernel_size-1)//2))
                else:
                    lst.append(torch.nn.AvgPool1d(kernel_size=hparams_encoder.local_pool_kernel_size,stride=hparams_encoder.local_pool_stride if hparams_encoder.local_pool_stride!=0 else hparams_encoder.local_pool_kernel_size,padding=(hparams_encoder.local_pool_kernel_size-1)//2))        
        
        self.layers = nn.Sequential(*lst)
        self.downsampling_factor = (hparams_encoder.local_pool_stride if hparams_encoder.multi_prediction else 1)*np.prod(hparams_encoder.strides)
        
        self.timesteps_per_token = hparams_encoder.timesteps_per_token
        self.sequence_last = hparams_input_shape.sequence_last
        self.output_dim = hparams_encoder.features[-1]

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.output_dim
        self.output_shape.channels2 = 0
        self.output_shape.length = int(hparams_input_shape.length//self.downsampling_factor+ (1 if hparams_input_shape.length%self.downsampling_factor>0 else 0))
        self.output_shape.sequence_last = False


    def forward(self, **kwargs):
        seq = kwargs["seq"]
        if(not self.sequence_last):
            seq = torch.movedim(seq,1,-1)
        if(len(seq.size())==4):#spectrogram input
            seq = seq.view(seq.size(0),-1,seq.size(-1))#flatten
        if(self.timesteps_per_token > 1):#patches a la vision transformer
            assert(seq.size(2)%self.timesteps_per_token==0)
            size = seq.size()
            seq = seq.transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2) # output: bs, output_dim, seq//downsampling_factor
        return {"seq": self.layers(seq).transpose(1,2)}#bs,seq,feat
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class RNNEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.encoder.RNNEncoder"

    #local pool after first conv
    multi_prediction:bool = False #local_pool named like this for consistency with MLP heads etc
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    
    strides:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder strides (space-separated)")
    kss:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder kernel sizes (space-separated)")
    features:List[int]=field(default_factory=lambda: [512,512,512,512]) #help="encoder features (space-separated)")
    dilations:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder dilations (space-separated)")
    normalization:bool=True #help="disable encoder batch/layer normalization")
    layer_norm:bool=False#", action="store_true", help="encoder layer normalization")