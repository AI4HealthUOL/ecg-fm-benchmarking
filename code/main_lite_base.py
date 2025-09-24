"""
Base classes and utilities for main_lite: Main_Lite, FMWrapperBase, and multihot_encode.
"""
import torch
from torch import nn
import lightning.pytorch as lp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import dataclasses
import numpy as np
from pathlib import Path

from clinical_ts.utils.eval_utils_cafa import multiclass_roc_curve
from clinical_ts.utils.eval_utils_regression import regression_metrics
from clinical_ts.data.time_series_dataset import *
from clinical_ts.data.time_series_dataset_utils import *
from clinical_ts.data.time_series_dataset_transforms import *
from clinical_ts.utils.bootstrap_utils import empirical_bootstrap
from clinical_ts.utils.schedulers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_invsqrt_decay_schedule_with_warmup,
)
# Utility function

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res

# at this scope to avoid pickle issues
def mcrc_flat(targs,preds,classes):
    _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
    return np.array(list(res.values()))

def regression_flat(targs, preds, metrics=["mae"], target_names=None):
    res = regression_metrics(targs, preds, metrics=metrics, target_names=target_names)

    n_targets = targs.shape[1]
    if target_names is None:
        target_names = [str(i) for i in range(n_targets)]
    
    ordered_values = []

    for metric in metrics:
        ordered_values.append(res[metric])
    
    for target_name in target_names:
        for metric in metrics:
            ordered_values.append(res[f"{target_name}_{metric}"])

    return np.array(ordered_values)

# FMWrapperBase
class FMWrapperBase(nn.Module):
    def __init__(self, num_classes, num_output_tokens):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_tokens = num_output_tokens

    def get_model_transforms(self, tfms_lst):
        return tfms_lst

# Main_Lite base class
class Main_Lite(lp.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr
        print(hparams)
    # ... rest of Main_Lite definition ... 

    def forward(self, x, **kwargs):
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

    def eval_scores(self, targs, preds, classes=None, bootstrap=False):
        if self.task == "classification_and_regression":
            cls_preds = preds[:, :-35]
            reg_preds = preds[:, -35:]
                
            cls_targs = targs[:, :-35]
            reg_targs = targs[:, -35:]

            cls_metrics = {}
            reg_metrics = {}
        
            _, _, cls_res = multiclass_roc_curve(cls_targs, cls_preds, classes=classes[:-35])
            cls_metrics = {f"{k}": v for k, v in cls_res.items()}

            reg_res = regression_metrics(reg_targs, reg_preds, metrics=["mae"], target_names=classes[-35:])
            reg_metrics = {f"{k}": v for k, v in reg_res.items()}

            cls_metrics["composite_score"] = (1.0 - cls_res["macro"]) + reg_res["mae"]

            if bootstrap:
                cls_point, cls_low, cls_high, _ = empirical_bootstrap(
                    input_tuple=(cls_targs, cls_preds),
                    score_fn=mcrc_flat,
                    n_iterations=1000,
                    score_fn_kwargs={"classes": classes[:-35]}
                )
                reg_point, reg_low, reg_high, _ = empirical_bootstrap(
                    input_tuple=(reg_targs, reg_preds),
                    score_fn=regression_flat,
                    n_iterations=1000,
                    score_fn_kwargs={"metrics": ["mae"], "target_names": classes[-35:]}
                )

                cls_bootstrap = {}
                for i, k in enumerate(cls_res.keys()):
                    cls_bootstrap[f"{k}"] = cls_point[i]
                    cls_bootstrap[f"{k}_low"] = cls_low[i]
                    cls_bootstrap[f"{k}_high"] = cls_high[i]
                
                reg_bootstrap = {}
                for i, k in enumerate(reg_res.keys()):
                    reg_bootstrap[f"{k}"] = reg_point[i]
                    reg_bootstrap[f"{k}_low"] = reg_low[i]
                    reg_bootstrap[f"{k}_high"] = reg_high[i]

                composite_point = (1.0 - cls_bootstrap["macro"]) + reg_bootstrap["mae"]
                composite_low = (1.0 - cls_bootstrap["macro_low"]) + reg_bootstrap["mae_low"]
                composite_high = (1.0 - cls_bootstrap["macro_high"]) + reg_bootstrap["mae_high"]

                cls_bootstrap["composite_score"] = composite_point
                cls_bootstrap["composite_score_low"] = composite_low
                cls_bootstrap["composite_score_high"] = composite_high
                
                return {**cls_bootstrap, **reg_bootstrap}
            else:
                return {**cls_metrics, **reg_metrics}
        else:
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
        preds_all = torch.cat(outputs_all["preds"]).cpu()
        targs_all = torch.cat(outputs_all["targs"]).cpu()

        if self.task != "classification_and_regression":
            if(self.hparams.finetune_dataset == "thew" or self.hparams.finetune_dataset.startswith("segrhythm")):
                preds_all = F.softmax(preds_all.float(),dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            else:
                preds_all = torch.sigmoid(preds_all.float())

        if self.task == "classification_and_regression":
            cls_preds = torch.sigmoid(preds_all[:, :-35]).float()
            reg_preds = preds_all[:, -35:].float()
            preds_all = torch.cat([cls_preds, reg_preds], dim=1)

        preds_all = preds_all.numpy()
        targs_all = targs_all.numpy()

        # Non-aggregated

        res = self.eval_scores(targs_all, preds_all, self.lbl_itos, bootstrap=test)
        if self.task == "classification_and_regression":
            res = {k+"_noagg_"+("test" if test else "val")+str(dataloader_idx): v for k, v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} composite score noagg: {res['composite_score_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} macro auroc noagg: {res['macro_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} mae noagg: {res['mae_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
        else:
            res = {k+"_auc_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}
            print("epoch",self.current_epoch,"test" if test else "val","noagg:",res["macro_auc_noagg_"+("test" if test else "val")+str(dataloader_idx)])
        self.log_dict(res)

        # Aggregated
        preds_all_agg,targs_all_agg = self.val_datasets[0].aggregate_predictions(preds_all,targs_all,self.test_idmaps[dataloader_idx] if test else self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
        res_agg = self.eval_scores(targs_all_agg, preds_all_agg, self.lbl_itos, bootstrap=test)
        if self.task == "classification_and_regression":
            res_agg = {k+"_agg_"+("test" if test else "val")+str(dataloader_idx): v for k, v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} composite score agg: {res_agg['composite_score_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} macro auroc agg: {res_agg['macro_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} mae agg: {res_agg['mae_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
        else:            
            res_agg = {k+"_auc_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}            
            print("epoch",self.current_epoch,"test" if test else "val","agg:",res_agg["macro_auc_agg_"+("test" if test else "val")+str(dataloader_idx)])
        self.log_dict(res_agg)
        
        
        if self.hparams.export_predictions:
            # Find version number of the prediction folder

            prediction_dir = Path(self.hparams.prediction_path)
            prediction_dir.mkdir(parents=True, exist_ok=True)

            existing_versions = []
            for folder in prediction_dir.glob(f"{self.hparams.finetune_dataset}_version_*"):
                if folder.is_dir():
                    try:
                        ver_num = int(folder.name.split('_')[-1])
                        existing_versions.append(ver_num)
                    except ValueError:
                        raise ValueError(f"Unexpected folder name format: '{folder.name}'. Expected a suffix with an integer version number.")
            
            current_version = max(existing_versions) + 1 if existing_versions else 0

            if test:
                self._export_predictions(
                    preds=preds_all,
                    targs=targs_all,
                    dataloader_idx=dataloader_idx,
                    epoch=self.current_epoch,
                    agg=False,
                    version=current_version
                )

                self._export_predictions(
                    preds=preds_all_agg,
                    targs=targs_all_agg,
                    dataloader_idx = dataloader_idx,
                    epoch=self.current_epoch,
                    agg=True,
                    version=current_version
                )

    
    def _export_predictions(self, preds, targs, dataloader_idx, epoch, agg, version):
        prediction_dir = Path(self.hparams.prediction_path)        
        version_folder_name = f"{self.hparams.finetune_dataset}_version_{version}"
        full_prediction_dir = prediction_dir / version_folder_name / ("agg" if agg else "noagg")
        full_prediction_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            full_prediction_dir / f"test_{dataloader_idx}_epoch_{epoch}_{'agg' if agg else 'noagg'}.npz",
            preds=preds,
            targs=targs,
            lbl_itos=np.array(self.lbl_itos),
            epoch=epoch
        )    
    
    def setup_dataset(self, target_folder, df_mapped, lbl_itos, mean, std):
        return df_mapped, lbl_itos, mean, std

    def setup_transforms(self, tfms_lst):
        return tfms_lst

    def setup(self, stage):
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
            if self.ds_mean is None:
                self.ds_mean = mean
                self.ds_std = std
            tfms_lst = []
            if(self.hparams.fs_model != self.hparams.fs_data):
                tfms_lst.append(Resample(self.hparams.fs_data, self.hparams.fs_model))
            if self.hparams.normalize:
                tfms_lst.append(Normalize(self.ds_mean, self.ds_std))
            if hasattr(self.model, 'get_model_transforms'):
                tfms_lst = self.model.get_model_transforms(tfms_lst)
            tfms_lst.append(ToTensor())
            tfms = tfms_lst[0] if len(tfms_lst)==1 else transforms.Compose(tfms_lst)
            max_fold_id = df_mapped.strat_fold.max()
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
                transforms=tfms,
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
            if(i<len(self.hparams.data.split(","))):
                print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            print("test dataset:",len(test_datasets[-1]),"samples")
        if(len(train_datasets)>1):
            print("\nCombined:")
            self.train_dataset = ConcatTimeSeriesDataset(train_datasets)
            self.val_datasets = [ConcatTimeSeriesDataset(val_datasets)]+val_datasets
            print("train dataset:",len(self.train_dataset),"samples")
            print("val datasets (total):",len(self.val_datasets[0]),"samples")
            self.test_datasets = [ConcatTimeSeriesDataset(test_datasets)]+test_datasets
            print("test datasets (total):",len(self.test_datasets[0]),"samples")
        else:
            self.train_dataset = train_datasets[0]
            self.val_datasets = val_datasets
            self.test_datasets = test_datasets
        self.val_preds=[[] for _ in range(len(self.val_datasets))]
        self.val_targs=[[] for _ in range(len(self.val_datasets))]
        self.test_preds=[[] for _ in range(len(self.test_datasets))]
        self.test_targs=[[] for _ in range(len(self.test_datasets))]
        self.val_idmaps = [ds.get_id_mapping() for ds in self.val_datasets]
        self.test_idmaps = [ds.get_id_mapping() for ds in self.test_datasets]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last = True)
    def val_dataloader(self):
        return [DataLoader(ds, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4) for ds in self.val_datasets]
    def test_dataloader(self):
        return [DataLoader(ds, collate_fn=tsdata_collate_fn, batch_size=self.hparams.batch_size, num_workers=4) for ds in self.test_datasets]
    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0):
        preds_all = self.forward(data_batch["seq"])
        loss = self.criterion(preds_all, data_batch["label"].float())
        
        if(not train and not test):
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch["label"])
        elif(not train and test):
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch["label"])
        return loss
    # def _step(self, data_batch, batch_idx, train, test=False, dataloader_idx=0):
    #     preds_all = self.forward(data_batch["seq"])
    #     targets = data_batch["label"]

    #     print(f"Step Pred shape: {preds_all.shape}")
    #     print(f"Step Targ shape: {targets.shape}")
        
    #     nan_mask = torch.isnan(targets)
    #     special_value_mask = (targets == -999)
        
    #     invalid_mask = nan_mask | special_value_mask
        
    #     if torch.any(invalid_mask):
    #         row_has_invalid = invalid_mask.any(dim=1)
    #         completely_valid_rows = ~row_has_invalid
            
    #         targets_clean = targets[completely_valid_rows]
    #         preds_clean = preds_all[completely_valid_rows]
    #         loss = self.criterion(preds_clean, targets_clean.float())
    #     else:
    #         loss = self.criterion(preds_all, targets.float())
        
    #     self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
        
    #     if(not train and not test):
    #         self.val_preds[dataloader_idx].append(preds_all.detach())
    #         self.val_targs[dataloader_idx].append(data_batch["label"])
    #     elif(not train and test):
    #         self.test_preds[dataloader_idx].append(preds_all.detach())
    #         self.test_targs[dataloader_idx].append(data_batch["label"])
    #     return loss
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
        elif(self.hparams.lr_schedule=="linear"):
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
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    def load_state_dict(self, state_dict, strict=True):
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
