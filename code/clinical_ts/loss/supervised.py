__all__ = ['SupervisedLossConfig', 'BCELossConfig', 'BinaryCrossEntropyLoss', 'CELossConfig', 'CrossEntropyLoss', 'QuantileRegressionLoss', 'QuantileRegressionLossConfig', 'CrossEntropyFocalLoss', 'CEFLossConfig', 'BinaryCrossEntropyFocalLoss', 'BCEFLossConfig', 'MSELoss', 'MSELossConfig']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass, field
from typing import List

from clinical_ts.template_modules import LossConfig

####################################################################################
# BASIC supervised losses
###################################################################################
@dataclass
class SupervisedLossConfig(LossConfig):
    _target_:str = "" #insert appropriate loss class
    loss_type:str ="supervised"
    supervised_type:str="classification_single"#"classification_multi","regression_quantile"

#https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
class QuantileRegressionLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.quantiles = hparams_loss.quantiles
        self.register_buffer("weight",torch.from_numpy(np.array(hparams_loss.weight,dtype=np.float32)).unsqueeze(0).unsqueeze(2) if len(hparams_loss.weight)>0 else torch.ones(1),persistent=True)
        
    def forward(self, preds, target):
        assert not target.requires_grad
        preds = preds.view(preds.size(0),-1,len(self.quantiles))#bs, len(lbl_itos), quantiles
        assert(preds.size(1)==target.size(1)) 
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:,:, i]
            losses.append(
                torch.max(
                (q-1) * errors, 
                q * errors
            ).unsqueeze(2))#bs,labels,1
        loss =  self.weight*torch.sum(torch.cat(losses, dim=2), dim=2).view(-1) #bs*labels
        loss = torch.mean(loss[~torch.isnan(loss)]) # to deal with nans
        return loss

@dataclass
class QuantileRegressionLossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.QuantileRegressionLoss"
    loss_type:str="supervised"
    quantiles:List[float]= field(default_factory=lambda: [0.5,0.025,0.975])
    supervised_type:str="regression_quantile"
    weight:List[float]=field(default_factory=lambda: [])#class weights e.g. target medians

class CrossEntropyLoss(nn.Module):
    #standard CE loss that just passes the class_weights correctly
    def __init__(self, hparams_loss):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(hparams_loss.weight,dtype=np.float32)) if len(hparams_loss.weight)>0 else None)
        
    def forward(self, preds, targs):
        return self.ce(preds,targs)

@dataclass
class CELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.CrossEntropyLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_single"
    weight:List[float]=field(default_factory=lambda: [])#class weights e.g. inverse class prevalences

class CrossEntropyFocalLoss(nn.Module):
    """
    Focal CE loss for multiclass classification with integer labels
    Reference: https://github.com/artemmavrin/focal-loss/blob/7a1810a968051b6acfedf2052123eb76ba3128c4/src/focal_loss/_categorical_focal_loss.py#L162
    """
    def __init__(self, hparams_loss):
        super().__init__()
        self.gamma = hparams_loss.gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(hparams_loss.weight, dtype=np.float32)) if len(hparams_loss.weight)>0 else None, reduction='none')

    def forward(self, preds, targs):
        probs = F.softmax(preds, dim=-1).squeeze(-1)
        probs = torch.gather(probs, -1, targs.unsqueeze(-1)).squeeze(-1)
        focal_modulation = torch.pow((1 - probs), self.gamma if type(self.gamma)==float else self.gamma.index_select(dim=0, index=preds.argmax(dim=-1)))
        # mean aggregation
        return (focal_modulation*self.ce(input=preds, target=targs)).mean()
        
@dataclass
class CEFLossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.CrossEntropyFocalLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_single"
    weight:List[float]=field(default_factory=lambda: []) #ignored if empty list is passed
    gamma:float=2.


class BinaryCrossEntropyLoss(nn.Module):
    #standard BCE loss that just passes the pos_weight correctly
    def __init__(self, hparams_loss):
        super().__init__()
        self.ignore_nans = hparams_loss.ignore_nans
        self.pos_weight_set = len(hparams_loss.pos_weight)>0

        if(not self.ignore_nans):
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(hparams_loss.pos_weight,dtype=np.float32)) if len(hparams_loss.pos_weight)>0 else None)
        else:
            if(self.pos_weight_set):
                self.bce = torch.nn.ModuleList([torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array([hparams_loss.pos_weight[i]],dtype=np.float32))) for i in range(len(self.pos_weight_set))])
            else:
                self.bce = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targs):
        if(not self.ignore_nans):
            return self.bce(preds,targs)
        else:
            losses = []
            for i in range(preds.size(1)):
                predsi = preds[:,i]
                targsi = targs[:,i]
                maski = ~torch.isnan(targsi)
                predsi = predsi[maski]
                targsi = targsi[maski]
                if(len(predsi)>0):
                    if(self.pos_weight_set):
                        losses.append(self.bce[i](predsi,targsi))
                    else:
                        losses.append(self.bce(predsi,targsi))
                
            return torch.sum(torch.cat(losses)) if(len(losses)>0) else 0.
                    

@dataclass
class BCELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.BinaryCrossEntropyLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_multi"
    pos_weight:List[float]=field(default_factory=lambda: [])#class weights e.g. inverse class prevalences
    ignore_nans:bool=False #ignore nans- requires separate BCEs for each label

class BinaryCrossEntropyFocalLoss(nn.Module):
    """
    Focal BCE loss for binary classification with labels of 0 and 1
    """
    def __init__(self, hparams_loss):
        super().__init__()
        self.gamma = hparams_loss.gamma
        
        self.ignore_nans = hparams_loss.ignore_nans
        self.pos_weight_set = len(hparams_loss.pos_weight)>0

        if(not self.ignore_nans):
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.from_numpy(np.array(hparams_loss.pos_weight,dtype=np.float32)) if len(hparams_loss.pos_weight)>0 else None)
        else:
            if(self.pos_weight_set):
                self.bce = torch.nn.ModuleList([torch.nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.from_numpy(np.array([hparams_loss.pos_weight[i]],dtype=np.float32))) for i in range(len(self.pos_weight_set))])
            else:
                self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targs):
        if(not(self.ignore_nans)):
            probs = torch.sigmoid(preds)
            p_t = probs * targs + (1 - probs) * (1 - targs)
            focal_modulation = torch.pow((1 - p_t), self.gamma)
            # mean aggregation
            return (focal_modulation * self.bce(input=preds, target=targs.float())).sum(-1).mean()
        else:
            losses = []
            for i in range(preds.size(1)):
                predsi = preds[:,i]
                targsi = targs[:,i]
                maski = ~torch.isnan(targsi)
                predsi = predsi[maski]
                targsi = targsi[maski]
                if(len(predsi)>0):
                    probsi = torch.sigmoid(predsi)
                    p_ti = probsi * targsi + (1 - probsi) * (1 - targsi)
                    focal_modulationi = torch.pow((1 - p_ti), self.gamma)
                    if(self.pos_weight_set):
                        losses.append(torch.mean(focal_modulationi*self.bce[i](predsi,targsi)))
                    else:
                        losses.append(torch.mean(focal_modulationi*self.bce(predsi,targsi)))
                
            return torch.sum(torch.stack(losses)) if(len(losses)>0) else 0.
        
@dataclass
class BCEFLossConfig(BCELossConfig):
    _target_:str= "clinical_ts.loss.supervised.BinaryCrossEntropyFocalLoss"
    gamma:float=2.

#old version, which does not work around nans
#@dataclass
#class MSELossConfig(SupervisedLossConfig):
#    _target_:str= "torch.nn.functional.mse_loss"
#    loss_type:str="supervised"
#    supervised_type:str="regression"


class MSELoss(nn.Module):
    """
    MSE loss that ignores nan tokens
    """
    def __init__(self, hparams_loss):
        super().__init__()
        self.register_buffer("weight",torch.from_numpy(np.array(hparams_loss.weight,dtype=np.float32)).unsqueeze(0).unsqueeze(2) if len(hparams_loss.weight)>0 else torch.ones(1),persistent=True)

    def forward(self, preds, targs):
        #loss = self.weight*self.weight*torch.square(preds-targs).view(-1)#bs*labels or just bs
        #return torch.mean(loss[~torch.isnan(targs.view(-1))])
        
        preds = self.weight* preds
        targs = self.weight* targs

        mask = ~torch.isnan(targs)
        return torch.nn.functional.mse_loss(preds[mask],targs[mask])
        

@dataclass
class MSELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.MSELoss"
    loss_type:str="supervised"
    supervised_type:str="regression"
    weight:List[float]=field(default_factory=lambda: []) #ignored if empty list is passed