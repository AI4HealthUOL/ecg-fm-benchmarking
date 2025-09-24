__all__ = ['S4Predictor','S4PredictorConfig']

from typing import Any
from dataclasses import dataclass

from clinical_ts.ts.s4_modules.s4_model import S4Model
from clinical_ts.template_modules import PredictorBase, PredictorBaseConfig

class S4Predictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)
        self.predictor = S4Model(
            d_input = hparams_input_shape.channels if hparams_input_shape.channels!=hparams_predictor.model_dim else None,#modified
            d_output = None,
            d_state = hparams_predictor.state_dim,
            d_model = hparams_predictor.model_dim,
            n_layers = hparams_predictor.layers,
            dropout = hparams_predictor.dropout,
            tie_dropout = hparams_predictor.tie_dropout,
            prenorm = hparams_predictor.prenorm,
            l_max = hparams_input_shape.length,
            transposed_input = False,
            bidirectional=not(hparams_predictor.causal),
            layer_norm=not(hparams_predictor.batchnorm),
            pooling = False,
            backbone = hparams_predictor.backbone) #note: only apply linear layer before if feature dimensions do not match

    def forward(self, **kwargs):   
        return {"seq": self.predictor(kwargs["seq"])}

@dataclass
class S4PredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.s4.S4Predictor"
    model_dim:int = 512 
    causal: bool = True #use bidirectional predictor
    state_dim:int = 64 #help="S4: N")
    layers:int = 4
    dropout:float=0.2
    tie_dropout:bool=True
    prenorm:bool=False
    batchnorm:bool=False
    backbone:str="s42" #help="s4original/s4new/s4d")  