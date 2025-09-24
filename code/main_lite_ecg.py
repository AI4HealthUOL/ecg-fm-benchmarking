"""
ECG-specific Main_Lite_ECG class for the fm-benchmarking project.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from main_lite_base import Main_Lite, multihot_encode
from clinical_ts.models.s4_model import S4Model
from clinical_ts.utils.schedulers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_invsqrt_decay_schedule_with_warmup,
)
from clinical_ts.models.ecg_foundation_models.ecg_founder import Net1D
from clinical_ts.models.fm_ecg import ECGFounderWrapper, ECGJEPAWrapper, StMemWrapper, MerlWrapper, EcgFmKEDWrapper, HubertEcgWrapper, CPCWrapper, ECG_JEPA_Scratch
from clinical_ts.utils.stratify import stratified_subset


class Main_Lite_ECG(Main_Lite):
    def __init__(self,hparams):
        super().__init__(hparams)

        if(hparams.finetune_dataset == "ptb"):
            num_classes = 18
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "ningbo"):
            num_classes = 68
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "cpsc2018"):
            num_classes = 9
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "cpsc_extra"):
            num_classes = 33
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "georgia"):
            num_classes = 50
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "chapman"):
            num_classes = 42
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "sph"):
            num_classes = 35
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "code15_diag"):
            num_classes = 6
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "code_test"):
            num_classes = 6
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "ptbxl_super"):
            num_classes = 5
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "ptbxl_sub"):
            num_classes = 23
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "ptbxl_all"):
            num_classes = 71
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "echonext"):
            num_classes = 11
            self.task = "classification_multi"
        elif(hparams.finetune_dataset == "mimic"):
            num_classes = 1127
            self.task = "classification_and_regression"
        elif(hparams.finetune_dataset == "zzu_pecg"):
            num_classes = 58
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ptbxl_label_efficiency":
            num_classes = 5
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "echonext_label_efficiency":
            num_classes = 6
            self.task = "classification_multi"
        elif self.hparams.finetune_dataset == "ptbxl_all_label_efficiency":
            num_classes = 13
            self.task = "classification_multi"
            
        # elif(hparams.finetune_dataset.startswith("mimic")): #default setting: mimic_ed_all_edfirst_all_2000_5A
            #prefetch dataframe
        #    df_mapped, lbl_itos,  mean, std = load_dataset(Path(list(hparams.data.split(","))[0]))
        #    _, lbl_itos, _, _ = self.setup_dataset(hparams.data, df_mapped, lbl_itos, mean, std)
        #    num_classes = len(lbl_itos)
        #    self.task = "classification_multi"

        ###################################################################################################
        # prepare loss
        ###################################################################################################
        if(self.task == "classification_single"):
            self.criterion = F.cross_entropy 
        elif(self.task == "classification_multi"):
            self.criterion = F.binary_cross_entropy_with_logits
        elif(self.task == "regression"):
            self.criterion = F.l1_loss
        elif(self.task == "classification_and_regression"):
            def classification_and_regression_loss(preds, targs):
                cls_preds = preds[:, :-35]
                reg_preds = preds[:, -35:]                
                cls_targs = targs[:, :-35]
                reg_targs = targs[:, -35:]

                cls_losses=[]
                for i in range(cls_targs.size(1)):
                    predsi = cls_preds[:, i]
                    targsi = cls_targs[:, i]
                    maski = ~torch.isnan(targsi)

                    if torch.any(maski):
                        loss_i = F.binary_cross_entropy_with_logits(
                            predsi[maski],
                            targsi[maski]
                        )
                        cls_losses.append(loss_i)
                    
                cls_loss = torch.mean(torch.stack(cls_losses))

                reg_losses = []
                for i in range(reg_targs.size(1)):
                    predsi = reg_preds[:, i]
                    targsi = reg_targs[:, i]
                    maski = ~torch.isnan(targsi)

                    if torch.any(maski):
                        loss_i = F.l1_loss(
                            predsi[maski],
                            targsi[maski]
                        )
                        reg_losses.append(loss_i)                    

                reg_loss = torch.mean(torch.stack(reg_losses))
    
                return cls_loss + reg_loss
            
            self.criterion = classification_and_regression_loss

        ###################################################################################################
        # prepare model
        ###################################################################################################
        input_size_model = int(self.hparams.input_size*self.hparams.fs_model )
        if(hparams.architecture=="ecg_founder"):
            self.model = ECGFounderWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif(hparams.architecture=="ecg_jepa"):
            self.model = ECGJEPAWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif(hparams.architecture=="st_mem"):
            self.model = StMemWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif(hparams.architecture=="merl"):
            self.model = MerlWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                backbone=hparams.merl_backbone,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif(hparams.architecture=="ecgfm_ked"):
            self.model = EcgFmKEDWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif(hparams.architecture=="s4"):
            self.model = S4Model(
                d_input=hparams.input_channels,
                d_output=num_classes,
                l_max=input_size_model,
                d_state=self.hparams.s4_n,
                d_model=self.hparams.s4_h,
                n_layers = self.hparams.s4_layers,
                bidirectional=True
            )
        elif(hparams.architecture=="net1d"):
            self.model = Net1D(
                in_channels=hparams.input_channels, 
                base_filters=64,
                ratio=1, 
                filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                kernel_size=16, 
                stride=2, 
                groups_width=16,
                verbose=False, 
                use_bn=False,
                use_do=False,
                n_classes=num_classes)
        elif(hparams.architecture=="cpc"):
            self.model = CPCWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                config_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
        )
        elif hparams.architecture == "hubert_ecg":
            self.model = HubertEcgWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor
            )
        elif hparams.architecture == "ecg_jepa_scratch":
            self.model = ECG_JEPA_Scratch(
                num_classes=num_classes
            )
        else:
            assert(False)

    def setup_dataset(self, target_folder, df_mapped, lbl_itos, mean, std):
        if(self.hparams.finetune_dataset == "ptb"):
            df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "ningbo"):
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "cpsc2018"):
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            df_mapped["label"] = df_mapped["labels"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "cpsc_extra"):
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "georgia"):
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "chapman"):
            lbl_itos = np.array(lbl_itos["label_all_filtered"])
            df_mapped["label"] = df_mapped["label_all_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "sph"):
            lbl_itos= np.array(lbl_itos["label_primary_filtered"])
            df_mapped["label"] = df_mapped["label_primary_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "code15_diag"):
            df_mapped = df_mapped[df_mapped.strat_fold >= 0].copy()
            df_mapped = df_mapped[df_mapped["data_length"] >= 4000]
            df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "code_test"):
            df_mapped = df_mapped[df_mapped["data_length"] >= 4000]
            df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset=="ptbxl_super"):
            lbl_itos = np.array(lbl_itos["label_diag_superclass"])
            df_mapped["label"] = df_mapped["label_diag_superclass_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset=="ptbxl_sub"):
            lbl_itos = np.array(lbl_itos["label_diag_subclass"])
            df_mapped["label"] = df_mapped["label_diag_subclass_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "ptbxl_all"):
            lbl_itos= np.array(lbl_itos["label_all"])
            df_mapped["label"] = df_mapped["label_all_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "echonext"):
            df_mapped = df_mapped[df_mapped["split"].isin(["train", "val", "test"])]
            df_mapped.loc[df_mapped["split"] == "val", "strat_fold"] = 8
            df_mapped.loc[df_mapped["split"] == "test", "strat_fold"] = 9
            df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif(self.hparams.finetune_dataset == "mimic"):
            df_mapped["label"] = df_mapped["label_all"]
        elif(self.hparams.finetune_dataset=="zzu_pecg"):
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["aha_description_filtered"])
            df_mapped["label"] = df_mapped["aha_description_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "ptbxl_label_efficiency":
            lbl_itos = np.array(lbl_itos["label_diag_superclass"])
            
            df_train = df_mapped[df_mapped["strat_fold"] <= 8]
            df_val = df_mapped[df_mapped["strat_fold"] == 9]
            df_test = df_mapped[df_mapped["strat_fold"] == 10]
            
            df_train_sub = stratified_subset(df_train, self.hparams.label_ratio)
            df_val_sub = stratified_subset(df_val, self.hparams.label_ratio)

            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped["label_diag_superclass_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "echonext_label_efficiency":
            label_filtered = np.array([
                "lvef_lte_45_flag",
                "lvwt_gte_13_flag",
                "tricuspid_regurgitation_moderate_or_greater_flag",
                "rv_systolic_dysfunction_moderate_or_greater_flag",
                "pasp_gte_45_flag",
                "tr_max_gte_32_flag"
            ])
            mapping = {i: np.where(label_filtered == lbl)[0][0] for i, lbl in enumerate(lbl_itos) if lbl in label_filtered}
            df_mapped["label_filtered_numeric"] = df_mapped['label'].apply(lambda x: [mapping[idx] for idx in x if idx in mapping])
            idx_to_label = {i: lbl for i, lbl in enumerate(label_filtered)}
            df_mapped["label_filtered"] = df_mapped["label_filtered_numeric"].apply(lambda x: [idx_to_label[idx] for idx in x])

            df_mapped = df_mapped[df_mapped["split"].isin(["train", "val", "test"])]
            df_mapped.loc[df_mapped["split"] == "val", "strat_fold"] = 8
            df_mapped.loc[df_mapped["split"] == "test", "strat_fold"] = 9

            df_train = df_mapped[df_mapped["strat_fold"] < 8]
            df_val = df_mapped[df_mapped["strat_fold"] == 8]
            df_test = df_mapped[df_mapped["strat_fold"] == 9]
                        
            df_train_sub = stratified_subset(
                df=df_train,
                ratio=self.hparams.label_ratio,
                col_label="label_filtered",
                col_age="age_at_ecg",
                value_male="male",
                col_patient="patient_key"
            )            
            df_val_sub = stratified_subset(
                df=df_val,
                ratio=self.hparams.label_ratio,
                col_label="label_filtered",
                col_age="age_at_ecg",
                value_male="male",
                col_patient="patient_key"
            )
            lbl_itos = [label for label in label_filtered]
            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "ptbxl_all_label_efficiency":
            labels = ["SR", "NORM", "ABQRS", "IMI", "ASMI", "LVH", "NDT", "LAFB", "AFIB", "ISC_", "PVC", "IRBBB", "STD_"]
            df_mapped["label"] = df_mapped["label_all_filtered"].apply(lambda x: [item for item in x if item in labels])
            df_mapped["label_numeric"] = df_mapped["label"].apply(lambda x: [labels.index(item) for item in x])
            df_mapped = df_mapped[df_mapped["label"].apply(len) != 0]

            df_train = df_mapped[df_mapped["strat_fold"] < 9]
            df_val = df_mapped[df_mapped["strat_fold"] == 9]
            df_test = df_mapped[df_mapped["strat_fold"] == 10]

            df_train_sub = stratified_subset(df=df_train, ratio=self.hparams.label_ratio, col_label="label")            
            df_val_sub = stratified_subset(df=df_val, ratio=self.hparams.label_ratio, col_label="label")

            lbl_itos = np.array(labels)
            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped["label_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        
        return df_mapped, lbl_itos, mean, std
        # elif(self.hparams.finetune_dataset.startswith("mimic")):
            #now request the actual labels (will load records_w_diag_icd10.csv to fetch the full labels)
        #    df_mapped, lbl_itos = prepare_mimic_ecg(self.hparams.finetune_dataset, Path(target_folder), df_mapped=df_mapped)        
        
    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            opt = torch.optim.SGD
        elif self.hparams.optimizer == "adam":
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        
        if hasattr(self.model, "get_params"):
            params = self.model.get_params()
        else:
            params = self.parameters()
        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_schedule=="const":
            scheduler = get_constant_schedule(optimizer)
        elif self.hparams.lr_schedule=="warmup-const":
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif self.hparams.lr_schedule=="warmup-cos":
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif self.hparams.lr_schedule=="warmup-cos-restart":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif self.hparams.lr_schedule=="warmup-poly":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif self.hparams.lr_schedule=="warmup-invsqrt":
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif self.hparams.lr_schedule=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        )
