__all__ = ['CPCLoss', 'CPCLossConfig']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import List
from dataclasses import dataclass, field
from clinical_ts.template_modules import SSLLossConfig, MaskingBaseConfig

class CPCLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.steps_predicted = hparams_loss.steps_predicted
        self.n_false_negatives = hparams_loss.n_false_negatives
        self.negatives_from_same_seq_only = hparams_loss.negatives_from_same_seq_only
        self.negatives_selection_interval = hparams_loss.negatives_selection_interval
        assert(self.negatives_from_same_seq_only is False or self.negatives_selection_interval==0 or self.negatives_selection_interval*2>= self.n_false_negatives)#make sure to have enough negatives available
        self.t = hparams_loss.temperature
        self.normalize = hparams_loss.normalize
        
    def forward(self,input_predicted,input_encoded, **kwargs):
        #both: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1,input_encoded.size(2)) #for negatives below: -1, features
        
        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]
        
        tp_cnt = torch.tensor(0,dtype=torch.int64, device=input_predicted.device)
        loss = torch.tensor(0,dtype=torch.float32, device=input_predicted.device)

        for i in range(seq-self.steps_predicted):
            positives = input_encoded[:,i+self.steps_predicted].unsqueeze(1) #bs,1,encoder_output_dim
            
            start = max(0,i-self.negatives_selection_interval) if self.negatives_selection_interval>0 else 0
            end = min(i+self.negatives_selection_interval+1,seq if self.negatives_selection_interval<self.steps_predicted else seq-1) if self.negatives_selection_interval>0 else seq-1

            idxs_seq = torch.randint(start,end,(bs*self.n_false_negatives,), device=input_predicted.device)
            #make sure we don't pick the positive
            idxs_seq2 = idxs_seq * (idxs_seq<(i+self.steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+self.steps_predicted)).long()#bs*false_neg
            if(self.negatives_from_same_seq_only):
                idxs_batch = torch.arange(0,bs, device=input_predicted.device).repeat_interleave(self.n_false_negatives)
            else:
                idxs_batch = torch.randint(0,bs,(bs*self.n_false_negatives,), device=input_predicted.device)
            idxs2_flat = idxs_batch*seq+idxs_seq2

            #old
            #if(self.negatives_from_same_seq_only):
            #    idxs = torch.randint(0,(seq-1),(bs*self.n_false_negatives,)).to(input_predicted.device)
            #else:#negative from everywhere
            #    idxs = torch.randint(0,bs*(seq-1),(bs*self.n_false_negatives,)).to(input_predicted.device)
            #idxs_seq = torch.remainder(idxs,seq-1) #bs*false_neg
            #idxs_seq2 = idxs_seq * (idxs_seq<(i+self.steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+self.steps_predicted)).long()#bs*false_neg
            #if(self.negatives_from_same_seq_only):
            #    idxs_batch = torch.arange(0,bs).repeat_interleave(self.n_false_negatives).to(input_predicted.device)
            #else:
            #    idxs_batch = idxs//(seq-1)
            #idxs2_flat = idxs_batch*seq+idxs_seq2 #for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity
            
            negatives = input_encoded_flat[idxs2_flat].view(bs,self.n_false_negatives,-1) #bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives,negatives],dim=1)#bs,false_neg+1,encoder_output_dim
            preds = input_predicted[:,i]
            if(self.normalize):
                candidates = F.normalize(candidates, p=2.0, dim = -1)
                preds = F.normalize(preds, p=2.0, dim = -1)

            sim=torch.sum(preds.unsqueeze(1)*candidates,dim=-1)/self.t #bs,(false_neg+1)
            targs = torch.zeros(bs, dtype=torch.int64, device=input_predicted.device)
            
            #if(eval_acc):
            sim_argmax = torch.argmax(sim,dim=-1)
            tp_cnt += torch.sum(sim_argmax == targs)
                
            loss += F.cross_entropy(sim,targs)
        return {"loss":loss, "metric_acc":tp_cnt.float()/bs/(input_encoded.size()[1]-self.steps_predicted)}


@dataclass
class CPCLossConfig(SSLLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.CPCLoss"
    loss_type:str = "cpc"
    steps_predicted: int = 12
    n_false_negatives: int = 128
    negatives_from_same_seq_only:bool = False # help="only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    negatives_selection_interval: int = 0 #only draw negative from -x...x around the current index
    normalize: bool = False #normalize before calculating similarities
    temperature: float = 1.0 #temperature parameter dividing similarities