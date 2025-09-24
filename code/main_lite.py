import torch
from torch import nn
import lightning.pytorch as lp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import argparse
import dataclasses
import os
import subprocess
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from clinical_ts.models.xresnet1d import xresnet1d50,xresnet1d101
from clinical_ts.models.inception1d import inception1d
from clinical_ts.models.s4_model import S4Model
from clinical_ts.utils.misc_utils import LRMonitorCallback
#################
#specific
from clinical_ts.data.time_series_dataset import *
from clinical_ts.data.time_series_dataset_utils import *
from clinical_ts.data.time_series_dataset_transforms import *

from clinical_ts.utils.schedulers import *
from clinical_ts.utils.eval_utils_cafa import multiclass_roc_curve
from clinical_ts.utils.bootstrap_utils import *

from pathlib import Path
import numpy as np
import pandas as pd

from main_lite_ecg import Main_Lite_ECG
from main_lite_eeg import Main_Lite_EEG
from main_lite_movement import Main_Lite_Movement

#mlflow without autologging https://github.com/zjohn77/lightning-mlflow-hf/blob/74c30c784f719ea166941751bda24393946530b7/lightning_mlflow/train.py#L39
MLFLOW_AVAILABLE=True
try:
    import mlflow
    from lightning.pytorch.loggers import MLFlowLogger
    from omegaconf import DictConfig, ListConfig

    def log_params_from_namespace(hparams):
        
        for k in hparams.keys():
            mlflow.log_param(k," " if str(hparams[k])=="" else str(hparams[k]))

except ImportError:
    MLFLOW_AVAILABLE=False

def get_git_revision_short_hash():
    return ""#subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

############################################################################################################
#at this scope to avoid pickle issues
def mcrc_flat(targs,preds,classes):
    _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
    return np.array(list(res.values()))
    
    
class Main_Lite(lp.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr

        print(hparams)
        

    def forward(self, x, **kwargs):
        # QUICK FIX FOR NANS IN INPUT
        x[torch.isnan(x)]=0
        x = self.model(x, **kwargs)
        x = torch.nan_to_num(x, nan=0.0)
        return x
    
    def on_validation_epoch_end(self):
        for i in range(len(self.val_preds)):
            self.on_valtest_epoch_eval({"preds":self.val_preds[i], "targs":self.val_targs[i]}, dataloader_idx=i, test=False)
            self.val_preds[i].clear()
            self.val_targs[i].clear()
    
    def on_test_epoch_end(self):
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_eval({"preds":self.test_preds[i], "targs":self.test_targs[i]}, dataloader_idx=i, test=True)
            self.test_preds[i].clear()
            self.test_targs[i].clear()

    def eval_scores(self, targs,preds,classes=None,bootstrap=False):
        _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
        if(bootstrap):
            point,low,high,_ = empirical_bootstrap((targs,preds), mcrc_flat, n_iterations=1000,score_fn_kwargs={"classes":classes})
            res2={}
            for i,k in enumerate(res.keys()):
                res2[k]=point[i]
                res2[k+"_low"]=low[i]
                res2[k+"_high"]=high[i]
            return res2
        return res 

    def on_valtest_epoch_eval(self, outputs_all, dataloader_idx, test=False):
        #for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            preds_all = torch.cat(outputs_all["preds"]).cpu()
            targs_all = torch.cat(outputs_all["targs"]).cpu()
            # apply softmax/sigmoid to ensure that aggregated scores are calculated based on them
            if(self.hparams.finetune_dataset == "thew" or self.hparams.finetune_dataset.startswith("segrhythm")):
                preds_all = F.softmax(preds_all.float(),dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            else:
                preds_all = torch.sigmoid(preds_all.float())
            
            preds_all = preds_all.numpy()
            targs_all = targs_all.numpy()
            #instance level score
            res = self.eval_scores(targs_all,preds_all,classes=self.lbl_itos,bootstrap=test)
            res = {k+"_auc_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}#avoid () for mlflow
            self.log_dict(res)
            print("epoch",self.current_epoch,"test" if test else "val","noagg:",res["macro_auc_noagg_"+("test" if test else "val")+str(dataloader_idx)])#,"agg:",res_agg)
            
            preds_all_agg,targs_all_agg = self.val_datasets[0].aggregate_predictions(preds_all,targs_all,self.test_idmaps[dataloader_idx] if test else self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
            res_agg = self.eval_scores(targs_all_agg,preds_all_agg,classes=self.lbl_itos,bootstrap=test)
            res_agg = {k+"_auc_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}
            self.log_dict(res_agg)

            print("epoch",self.current_epoch,"test" if test else "val","agg:",res_agg["macro_auc_agg_"+("test" if test else "val")+str(dataloader_idx)])#,"agg:",res_agg)

    def setup_dataset(self, target_folder, df_mapped, lbl_itos, mean, std):
        '''dataset specific modification in derived classes'''
        return df_mapped, lbl_itos, mean, std
    
    def setup_transforms(self, tfms_lst):
        '''modifications of default transformations in derived classes'''
        return tfms_lst
            
    def setup(self, stage): 

        # configure dataset params
        input_size_data = int(self.hparams.input_size*self.hparams.fs_data)
        chunkify_train = self.hparams.chunkify_train
        chunk_length_train = int(self.hparams.chunk_length_train*input_size_data) if chunkify_train else 0
        stride_train = int(self.hparams.stride_fraction_train*input_size_data)
        
        chunkify_valtest = True
        chunk_length_valtest = input_size_data if chunkify_valtest else 0
        stride_valtest = int(self.hparams.stride_fraction_valtest*input_size_data)

        train_datasets = []
        val_datasets = []
        test_datasets = []

        self.ds_mean = None
        self.ds_std = None
        self.lbl_itos = None

        for i,target_folder in enumerate(list(self.hparams.data.split(","))):
            target_folder = Path(target_folder)           
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
            df_mapped, lbl_itos,  mean, std = self.setup_dataset(target_folder, df_mapped, lbl_itos, mean, std)
            
            print("Folder:",target_folder,"Samples:",len(df_mapped))

            if(self.lbl_itos is None):
                self.lbl_itos = lbl_itos
            
            # Set dataset statistics for normalization
            if self.ds_mean is None:
                self.ds_mean = mean
                self.ds_std = std
        
            tfms_lst = []
            
            memmap_meta = load_memmap_meta_dict(target_folder/"memmap.npy")
            if(len(memmap_meta)==0 and self.hparams.fs_model != memmap_meta["fs"]):
                tfms_lst.append(Resample(memmap_meta["fs"], self.hparams.fs_model))
            elif(len(memmap_meta)==0 and self.hparams.fs_model != self.hparams.fs_data):
                print("legacy mode: please update your memmap files")
                tfms_lst.append(Resample(self.hparams.fs_data, self.hparams.fs_model))
            
            if self.hparams.normalize:
                tfms_lst.append(Normalize(self.ds_mean, self.ds_std))
            
            if hasattr(self.model, 'get_model_transforms'):
                tfms_lst = self.model.get_model_transforms(tfms_lst)
            
            tfms_lst.append(ToTensor())
            
            tfms = tfms_lst[0] if len(tfms_lst)==1 else transforms.Compose(tfms_lst)
        
            max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            df_train = df_mapped[df_mapped.strat_fold<max_fold_id-1]
            df_val = df_mapped[df_mapped.strat_fold==max_fold_id-1]
            df_test = df_mapped[df_mapped.strat_fold==max_fold_id]

            dataset_config_train = TimeSeriesDatasetConfig(
                df=df_train,
                output_size=input_size_data,
                data_folder=target_folder,
                chunk_length=chunk_length_train,
                min_chunk_length=input_size_data,
                stride=stride_train,
                transforms=tfms,  # Now tfms is properly defined
                col_lbl="label",
                memmap_filename=target_folder/("memmap.npy"))
            train_datasets.append(TimeSeriesDataset(dataset_config_train))

            dataset_config_val = dataclasses.replace(dataset_config_train)
            dataset_config_val.df = df_val
            dataset_config_val.chunk_length= chunk_length_valtest
            dataset_config_val.stride= stride_valtest
            dataset_config_val.transforms= tfms
            val_datasets.append(TimeSeriesDataset(dataset_config_val))

            dataset_config_test = dataclasses.replace(dataset_config_val)
            dataset_config_test.df = df_test
            test_datasets.append(TimeSeriesDataset(dataset_config_test))
            
            print("\n",target_folder)
            if(i<len(self.hparams.data.split(","))):  # Fixed condition
                print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            print("test dataset:",len(test_datasets[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatTimeSeriesDataset(train_datasets)
            self.val_datasets = [ConcatTimeSeriesDataset(val_datasets)]+val_datasets
            print("train dataset:",len(self.train_dataset),"samples")
            print("val datasets (total):",len(self.val_datasets[0]),"samples")
            self.test_datasets = [ConcatTimeSeriesDataset(test_datasets)]+test_datasets
            print("test datasets (total):",len(self.test_datasets[0]),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_datasets = val_datasets
            self.test_datasets = test_datasets

        #create empty lists for results
        self.val_preds=[[] for _ in range(len(self.val_datasets))]
        self.val_targs=[[] for _ in range(len(self.val_datasets))]
        self.test_preds=[[] for _ in range(len(self.test_datasets))]
        self.test_targs=[[] for _ in range(len(self.test_datasets))]
        
        # store idmaps for aggregation
        self.val_idmaps = [ds.get_id_mapping() for ds in self.val_datasets]
        self.test_idmaps = [ds.get_id_mapping() for ds in self.test_datasets]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return [DataLoader(ds, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4) for ds in self.val_datasets]
    
    def test_dataloader(self):
        return [DataLoader(ds, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4) for ds in self.test_datasets]
        
    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0):
        #if(torch.sum(torch.isnan(data_batch[0])).item()>0):#debugging
        #    print("nans",torch.sum(torch.isnan(data_batch[0])).item())
        preds_all = self.forward(data_batch["seq"])

        loss = self.criterion(preds_all,data_batch["label"])
        self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
        
        if(not train and not test):
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch["label"])
        elif(not train and test):
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch["label"])
        
        return loss
    
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch,batch_idx,train=True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False, dataloader_idx=dataloader_idx)
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        return self._step(test_batch,batch_idx,train=False,test=True, dataloader_idx=dataloader_idx)
    
    def configure_optimizers(self):
        
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
            
        params = self.parameters()

        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if(self.hparams.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif(self.hparams.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)
        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def load_state_dict(self, state_dict, strict=True):
        #S4-compatible load_state_dict
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Key {name} not found in state_dict")
        
        for name, param in self.named_buffers():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Buffer {name} not found in state_dict")




####################################################################################################
# FMWrapper Base Class
####################################################################################################
class FMWrapperBase(nn.Module):
    def __init__(self, num_classes, num_output_tokens):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_tokens = num_output_tokens

    def get_model_transforms(self, tfms_lst):
        '''should return model-specific transforms to bring data from its standard format into the format expected by the foundation model
        Note: before ToTensor shape is typically ts,ch afterwards ch,ts
        '''
        return tfms_lst

####################################################################################################
# MISC
######################################################################################################
def load_from_checkpoint(pl_model, checkpoint_path):
    """ load from checkpoint function that is compatible with S4
    """
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data


    
#####################################################################################################
#ARGPARSER
#####################################################################################################

def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
    parser.add_argument('--data', metavar='DIR',type=str,
                        help='path(s) to dataset (comma-separated)')#,action='append')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')#was sgd
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=str, default="16-mixed", help="32,16-mixed,bf16-mixed")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--input-size", dest="input_size", help="input size (in seconds)", type=float, default=2.5)
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument("--finetune", action="store_true", help="finetuning (downstream classification task)",  default=False )
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )
    
    parser.add_argument("--lr-schedule", type=str, help="const/warmup-const/warmup-cos/warmup-cos-restart/warmup-poly", default="const")
    parser.add_argument("--lr-num-warmup-steps", type=int, help="number of linear lr warmup steps", default=1000)
    
    parser.add_argument("--discriminative-lr-factor", type=float, help="factor by which the lr decreases per layer group during finetuning", default=0.1)
    parser.add_argument("--lr-find", action="store_true", help="run lr finder before training run", default=False)
    parser.add_argument("--auto-batch-size", action='store_true')

    parser.add_argument("--auc-maximization", action="store_true", help="direct auc maximization",  default=False)
    parser.add_argument("--refresh-rate", type=int, help="progress bar refresh rate (0 to disable)", default=0)

    parser.add_argument("--mlflow", action="store_true", help="also log to mlflow")
    parser.add_argument("--mlflow-experiment-name", type=str, default="fm-benchmarking", help="Name of the mlflow experiment")
    parser.add_argument("--mlflow-run-name", type=str, default="run", help="Give a meaningful name to distinguish")

    parser.add_argument("--fs-model", type=float, help="sampling frequency of the model", default=100)
    parser.add_argument("--fs-data", type=float, help="sampling frequency of the dataset", default=100)
    
    return parser

def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--architecture", type=str, help="xresnet1d50/xresnet1d101/inception1d/s4", default="xresnet1d50")
    
    parser.add_argument("--s4-n", type=int, default=8, help='S4: N (Sashimi default:64)')
    parser.add_argument("--s4-h", type=int, default=512, help='S4: H (Sashimi default:64)')
    parser.add_argument("--s4-layers", type=int, default=4, help='S4: number of layers (Sashimi default:8)')
    parser.add_argument("--s4-batchnorm", action='store_true', help='S4: use BN instead of LN')
    parser.add_argument("--s4-prenorm", action='store_true', help='S4: use prenorm')

    # MerlWrapper
    parser.add_argument("--merl-backbone", type=str, default="resnet", help="resnet/vit")
     
    return parser

def add_application_specific_args(parser):
    parser.add_argument("--modality", type=str, help="ecg/eeg/movement", default="ecg")
    
    parser.add_argument("--normalize", action='store_true', help='Normalize input using dataset stats')
    parser.add_argument("--finetune-dataset", type=str, help="...", default="ptbxl_all")
    parser.add_argument("--chunk-length-train", type=float, default=1.,help="training chunk length in multiples of input size")
    parser.add_argument("--stride-fraction-train", type=float, default=1.,help="training stride in multiples of input size")
    parser.add_argument("--stride-fraction-valtest", type=float, default=1.,help="val/test stride in multiples of input size")
    parser.add_argument("--chunkify-train", action='store_true')
    
    parser.add_argument("--segmentation", action='store_true')
    
    parser.add_argument("--eval-only", type=str, help="path to model checkpoint for evaluation", default="")

    parser.add_argument("--export-predictions", action="store_true", help="Export predictions in npz format")
    parser.add_argument('--prediction-path', default='.', type=str, dest="prediction_path", help='prediction path')
    parser.add_argument("--eval-mode", type=str, help="finetuning_linear/finetuning_nonlinear/frozen/linear", default="finetuning_linear")

    # Label Efficiency hyperparamters
    parser.add_argument("--label-ratio", type=float, default=1, help="ratio forl label efficiency")
    
    return parser
            
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    parser = add_application_specific_args(parser)

    hparams = parser.parse_args()
    hparams.executable = "main_lite_"+hparams.modality
    hparams.revision = get_git_revision_short_hash()

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)

    if(hparams.modality=="ecg"):    
        model = Main_Lite_ECG(hparams)
    elif(hparams.modality=="eeg"):    
        model = Main_Lite_EEG(hparams)
    elif(hparams.modality=="movement"):    
        model = Main_Lite_Movement(hparams)
    else:
        assert(False)

    logger = [TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")]
    print("Output directory:",logger[0].log_dir)

    if(MLFLOW_AVAILABLE):
        mlflow.set_experiment(hparams.mlflow_experiment_name)
        run = mlflow.start_run(run_name=hparams.mlflow_run_name)              

        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=False,
        )
        mlf_logger._run_id = run.info.run_id
        mlf_logger.log_hyperparams = log_params_from_namespace       
        logger.append(mlf_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger[0].log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor="composite_score_agg_val0" if hparams.finetune_dataset == "mimic" else "macro_auc_agg_val0",
        mode="min" if hparams.finetune_dataset == "mimic" else "max")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    #lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")

    callbacks = [checkpoint_callback,lr_monitor]#,lr_monitor2]

    if(hparams.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.refresh_rate))

    trainer = lp.Trainer(
        num_sanity_val_steps=0,#no debugging
        #overfit_batches=50,#debugging

        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        accelerator="gpu" if hparams.gpus>0 else "cpu",
        devices=hparams.gpus if hparams.gpus>0 else 1,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        #distributed_backend=hparams.distributed_backend,
        
        enable_progress_bar=hparams.refresh_rate>0)
        
    if(hparams.auto_batch_size):#auto tune batch size batch size
        tuner=Tuner(trainer)
        tuner.scale_batch_size(model, mode="binsearch")

    if(hparams.lr_find):# lr find
        tuner=Tuner(trainer)
        lr_finder = tuner.lr_find(model)
        new_lr = lr_finder.suggestion()
        print("Suggested lr:",new_lr)

    if(hparams.epochs > 0 and hparams.eval_only == ""):
        trainer.fit(model, ckpt_path=None if hparams.resume == "" else hparams.resume)
        trainer.test(model, ckpt_path="best")

    elif(hparams.eval_only!=""):#eval only
        trainer.test(model,ckpt_path=hparams.eval_only)
    
    if(MLFLOW_AVAILABLE):
        mlflow.end_run()