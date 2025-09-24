__all__ = ['LearnableQueryAttentionPoolingHead', 'LearnableQueryAttentionPoolingHeadConfig']

import torch
import torch.nn as nn
import torch.nn.functional as F

import dataclasses
from dataclasses import dataclass, field
from typing import List

class HeadBase(nn.Module):
    '''Head base class'''
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''
        input shape: bs, seq, feat
        output shape: bs,seq,nc for multi_prediction else bs,nc for global_pool
        '''
        super().__init__()
        self.target_dim = target_dim
        self.multi_prediction = hparams_head.multi_prediction
        
    def get_output_shape(self):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())
    
@dataclass
class HeadBaseConfig:
    _target_: str = ""
    multi_prediction: bool = False # prediction for every token/set of pooled tokens

class LearnableQueryAttentionPool1d(nn.Module):
    '''V-JEPA (https://openreview.net/forum?id=WFYbBOEOtv) Learnable Query Attention Pooling '''
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None, batch_first: bool = True, bias: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, bias=bias)
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.bias = bias

    def forward(self, x):
        if(self.batch_first): # (B, Sx, E) otherwise (Sx, B, E) 
            x = x.permute(1, 0, 2)  # (Sx, B, E)
        
        x, _ = F.multi_head_attention_forward(
            query=self.query.repeat(1,x.shape[1],1), key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]) if self.bias else None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias if self.bias else None,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class LearnableQueryAttentionPoolingHead(HeadBase):
    #learnable query attention pool a la v-jepa
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        assert(hparams_head.multi_prediction is False)
        
        self.head = LearnableQueryAttentionPool1d(embed_dim=hparams_input_shape.channels,num_heads=hparams_head.heads, output_dim=target_dim, bias=hparams_head.bias)

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = 0

    def forward(self, **kwargs):
        return {"seq": self.head(kwargs["seq"])}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class LearnableQueryAttentionPoolingHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.LearnableQueryAttentionPoolingHead"
    multi_prediction:bool=False

    heads:int = 16
    bias:bool = False