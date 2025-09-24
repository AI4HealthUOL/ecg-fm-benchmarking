__all__ = ['ForwardHook','TriggerQuantizerHyperparameterUpdate','UnfreezingFinetuningCallback','LRMonitorCallback', 'cos_anneal', 'DecayLR',"freeze_bn_stats","sanity_check"]

import torch
from torch import nn
import lightning.pytorch as lp
import math

from lightning.pytorch.callbacks import Callback, BaseFinetuning

class ForwardHook:
    "Create a forward hook on module `m` "

    def __init__(self, m, store_output=True):
        self.store_output = store_output
        self.hook = m.register_forward_hook(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "stores input/output"
        if self.store_output:
            self.stored = output
        else:
            self.stored = input

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

class TriggerQuantizerHyperparameterUpdate(Callback):
    def __init__(self,quantizer_modules):
        super(TriggerQuantizerHyperparameterUpdate, self).__init__()
        self.modules = quantizer_modules
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for m in self.modules:
            m.update_hyperparams(trainer.global_step)

class UnfreezingFinetuningCallback(BaseFinetuning):

    def __init__(self, unfreeze_epoch: int = 5, train_bn: bool = True):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: lp.LightningModule):
        modules = pl_module.get_params(modules=True)
        for mod in modules[1:]:
            self.freeze(mod["params"], train_bn=self.train_bn)
            
    def finetune_function(self, pl_module: lp.LightningModule, epoch: int, optimizer: torch.optim.Optimizer, opt_idx: int):
        """Called on every epoch starts."""
        if epoch == self.unfreeze_epoch:
            modules = pl_module.get_params(modules=True)
            for mod in modules[1:]:
                self.unfreeze_and_add_param_group(
                    mod["params"],
                    optimizer,
                    lr=mod["lr"]*optimizer.param_groups[0]["lr"]/modules[0]["lr"],
                    train_bn=self.train_bn,
                )

class LRMonitorCallback(Callback):
    def __init__(self,interval="epoch",start=True,end=True):
        super().__init__()
        self.interval = interval
        self.start = start
        self.end = end
        
    def on_train_batch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)
    
    def on_train_batch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

############################################################################################################
def freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if(isinstance(m,nn.BatchNorm1d)):
            if(freeze):
                m.eval()
            else:
                m.train()

############################################################################################################
def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'head.1.weight' in k or 'head.1.bias' in k or 'head.4.weight' in k or 'head.4.bias' in k:
            continue


        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

############################################################################################################
#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

class DecayLR(Callback):
    def __init__(self,num_steps=1200000,lrstart=3e-4,lrend=1.25e-6):
        super(DecayLR, self).__init__()
        self.num_steps = num_steps
        self.lrstart = lrstart
        self.lrend = lrend

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, self.num_steps, self.lrstart, self.lrend, trainer.global_step)
        for g in pl_module.model_cpc.optimizer.param_groups:
            g['lr'] = t