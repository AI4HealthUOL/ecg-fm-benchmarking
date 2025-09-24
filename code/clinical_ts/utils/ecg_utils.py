__all__ = ['get_available_channels', 'channel_stoi_default', 'channel_stoi_canonical', 'resample_data', 'get_filename_out',
           'prepare_ptbv2', 'prepare_data_ningbo', 'prepare_data_cpsc2018', 'prepare_data_cpsc_extra',
           'prepare_data_georgia', 'prepare_data_chapman', 'prepare_data_sph', 'prepare_data_ribeiro_full',
           'prepare_data_ptb_xl',  'map_and_filter_labels', 'prepare_mimicecg','prepare_heedb', 'prepare_data_echonext', 'prepare_data_zzu_pecg']

import traceback
import wfdb

import scipy.io

#icentia
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

import resampy
from tqdm.auto import tqdm
from pathlib import Path

#thew
from ishneholterlib import Holter
from clinical_ts.utils.stratify import stratify, stratify_batched

#ribeiro
import h5py
import datetime

#from clinical_ts.misc_utils import *
from clinical_ts.data.time_series_dataset_utils import *

from sklearn.model_selection import StratifiedKFold

from concurrent.futures import ProcessPoolExecutor, as_completed

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}
channel_stoi_canonical = {"i": 0, "ii": 1, "iii":2, "avr":3, "avl":4, "avf":5, "v1":6, "v2":7, "v3":8, "v4":9, "v5":10, "v6":11, "vx":12, "vy":13, "vz":14}

def get_stratified_kfolds(labels,n_splits,random_state):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    return skf.split(np.zeros(len(labels)),labels)

def get_available_channels(channel_labels, channel_stoi):
    if(channel_stoi is None):
        return range(len(channel_labels))
    else:
        return sorted([channel_stoi[c] for c in channel_labels if c in channel_stoi.keys()])

def fix_nans_and_clip(signal,clip_amp=3):
    for i in range(signal.shape[1]):
        tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
        signal[:,i]= np.clip(tmp,a_max=clip_amp, a_min=-clip_amp) if clip_amp>0 else tmp
    return signal

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                #if(skimage_transform):
                #    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                #else:
                #    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
                data[:,channel_stoi[cl]] = resampy.resample(sigbufs[:,i], fs, target_fs).astype(np.float32)
    else:
        #if(skimage_transform):
        #    data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        #else:
        #    data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data


def get_filename_out(filename_in, target_folder=None, suffix=""):
    if target_folder is None:
        #absolute path here
        filename_out = filename_in.parent/(filename_in.stem+suffix+".npy")
        filename_out_relative = filename_out
    else:
        if("train" in filename_in.parts):
            target_folder_train = target_folder/"train"
            # relative path here
            filename_out = target_folder_train/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)

            target_folder_train.mkdir(parents=True, exist_ok=True)
        elif("eval" in filename_in.parts or "dev_test" in filename_in.parts or "valid" in filename_in.parts or "valtest" in filename_in.parts):
            target_folder_valid = target_folder/"valid"
            filename_out = target_folder_valid/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder_valid.mkdir(parents=True, exist_ok=True)
        else:
            filename_out = target_folder/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)
    return filename_out, filename_out_relative


def _age_to_categorical(age):
    if(np.isnan(age)):
        label_age = -1
    elif(age<30):
        label_age = 0
    elif(age<40):
        label_age = 1
    elif(age<50):
        label_age = 2
    elif(age<60):
        label_age = 3
    elif(age<70):
        label_age = 4
    elif(age<80):
        label_age = 5
    else:
        label_age = 6
    return label_age

def _sex_to_categorical(sex):
    sex_mapping = {"n/a":-1, "male":0, "female":1, "":-1}
    return sex_mapping[sex]

def map_and_filter_labels(df,min_cnt,lbl_cols):
    #filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in lbl_cols:
        if(min_cnt>0):
            label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
            df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
            lbl_itos_ptb_xl[selection+"_filtered"] = np.array(sorted(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist]))))
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection+"_filtered"])}
            df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
        #also lbl_itos and ..._numeric col for original label column
        lbl_itos_ptb_xl[selection]= np.array(sorted(list(set([x for sublist in df_ptb_xl[selection] for x in sublist]))))
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection+"_numeric"]=df_ptb_xl[selection].apply(lambda x:[lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl

###################################### PTB ##################################################
def prepare_ptbv2(data_path="", min_cnt=10, target_fs=1000, channels=12, strat_folds=5, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):

    if(recreate_data):
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        def extract_comments(file_path):
            # Read comments from HEA file using wfdb
            record = wfdb.rdrecord(file_path[:-4], physical=False)
            return record.comments

        df=[]
        # Loop through subfolders and extract comments
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith('.hea'):
                    file_path = os.path.join(root, file)
                    comments = extract_comments(file_path)

                    # Append data to the DataFrame
                    df += [{'filename': file_path, 'Comments': comments}]

        df=pd.DataFrame(df)

        for _,row in df.iterrows():
            sigbufs, header = wfdb.rdsamp(str(row["filename"])[:-4])
            if header['fs'] != 1000:
                print(f'frequency is {header['fs']}')
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            
            stem=Path(row["filename"]).stem
            assert(target_fs<=header['fs'])
            np.save(target_folder/stem,data)
        df["data"]=df["filename"].apply(lambda x: Path(x).stem+".npy")
        df["patient_id"]=df["filename"].apply(lambda x: int(Path(x).parent.stem.replace("patient","")))
        
        #extract diagnoses from comments
        df["Comments"]=df["Comments"].apply(lambda x: [l.split(":") for l in x])
        df["Comments"]=df["Comments"].apply(lambda x: {y[0]:y[1] for y in x})


        for k in df["Comments"].iloc[0].keys():
            df[k]=df["Comments"].apply(lambda x: x[k])


        df = df.drop(columns=['Comments'])
        df = df.drop(columns=['Diagnose'])

        df["Additional diagnoses"]=df["Additional diagnoses"].apply(lambda x: x.split(","))

        df["Combined diagnoses"]=df.apply(lambda row:[row["Reason for admission"]]+row["Additional diagnoses"],axis=1)

        def clean_diags(lst):
            lst=[x.strip() for x in lst]
            blacklist=["no","n/a","unknown", "M. Bechterew", "M. Basedow"]
            lst=[x for x in lst if not (x in blacklist)]
            return lst

        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: clean_diags(x))

        df['ECG date']=df['ECG date'].apply(lambda x: np.nan if x.strip()=="n/a" else x.strip())
        df['ECG date'] = pd.to_datetime(df['ECG date'], format='%d/%m/%Y')

        df['Infarction date (acute)']=df['Infarction date (acute)'].apply(lambda x: np.nan if x.strip()=="n/a" else x.strip())
        df['Infarction date (acute)'] = pd.to_datetime(df['Infarction date (acute)'], format='%d-%b-%y')

        df['Catheterization date']=df['Catheterization date'].apply(lambda x: np.nan if (x.strip()=="n/a" or x.strip()=="-") else x.strip())
        df['Catheterization date'] = pd.to_datetime(df['Catheterization date'], format='%d-%b-%y')

        df['days_since_infarction'] = (df['ECG date'] - df['Infarction date (acute)']).dt.days
        df['days_since_catheterization'] = (df['ECG date'] - df['Catheterization date']).dt.days

        #https://emedicine.medscape.com/article/1960472-overview?form=fpf
        df['acute_MI']= df['days_since_infarction'].apply(lambda x: x>=0 and x<=3)
        df["catheterized"]=df.apply(lambda x: x["days_since_infarction"]>=0 and x["days_since_catheterization"]<=x["days_since_infarction"],axis=1)

        df["MI"]=df["Combined diagnoses"].apply(lambda x: "Myocardial infarction" in x)

        cleanup_map={
            "Arterial Hypertension": "Arterial hypertension",
            "Hyperlipoproteinemia Type IIa":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Type IV":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Type IIb":"Hyperlipoproteinemia",
            "Hyperlipemia":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Typ IIa":"Hyperlipoproteinemia",
            "Postop. Thyriodectomy":"Postop. Thyroidectomy",
            "Coronary artery disease (PTCA+Stent) ":"Coronary artery disease",
            "Hyperlipoproteinemia Type IV b":"Hyperlipoproteinemia",
            "peripheral atherosclerosis":"Peripheral atherosclerosis",
            "Recurrent sustained ventricular tachycardia":"Ventricular tachycardia",
            "Ventricular tachycardias":"Ventricular tachycardia",
            "Recurrent ventricular tachycardias":"Ventricular tachycardia",
            "Recurrent ventricular tachycardia":"Ventricular tachycardia",
            "Hyperlipidemia":"Hyperlipoproteinemia",
            "Mitral regurgitation (grade 2)":"Mitral regurgitation",
            "Hypertrophic obstructive cardiomyopathy (intraventricular gradient 100-160mmHg)":"Hypertrophic obstructive cardiomyopathy",
            "Hypertrophic obstructive cardiomyopathy (intraventricular gradient 70-90mmHg)":"Hypertrophic obstructive cardiomyopathy"  
        }

        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: [y.strip() for y in x])
        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: [(cleanup_map[y] if y in cleanup_map.keys() else y) for y in x])

        def mistatus(mi,acute,catheterized):
            if(not mi):
                return ""
            res=""
            if(not acute):
                res+=" old"
            else:
                res+=" acute"
                if(catheterized):
                    res+=" catheterized"
            return res

        df["MI_status"]=df.apply(lambda row:mistatus(row["MI"],row["acute_MI"],row["catheterized"]), axis=1)
        df["Combined diagnoses"]=df.apply(lambda row: [(y+row["MI_status"] if y== "Myocardial infarction" else y) for y in row["Combined diagnoses"]] ,axis=1)

        diag,cnt = np.unique([x for xs in list(df["Combined diagnoses"]) for x in xs], return_counts=True)
        idx=np.argsort(cnt)[::-1]

        lbl_itos=[]
        res=[]
        for d,c in zip(diag[idx],cnt[idx]):
            if(c>=min_cnt):
                lbl_itos.append(d)
            res.append({"diag":d,"count":c})
            print(d,c)
        #df_stats=pd.DataFrame(res)
        #df_stats.to_excel("ptb_stats.xlsx")

        label_stoi={s:i for i,s in enumerate(lbl_itos)}
        df["label"]=df["Combined diagnoses"].apply(lambda x: [label_stoi[y] for y in x if y in lbl_itos])

        #random split by patients
        unique_patients = np.unique(df.patient_id)
        splits_patients = get_stratified_kfolds(np.zeros(len(unique_patients)),n_splits=strat_folds,random_state=42)
        df["strat_fold"]=-1
        for i,split in enumerate(splits_patients):
            df.loc[df.patient_id.isin(unique_patients[split[-1]]),"strat_fold"]=i

        
        #add means and std
        dataset_add_mean_col(df,data_folder=target_folder)
        dataset_add_std_col(df,data_folder=target_folder)
        dataset_add_length_col(df,data_folder=target_folder)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### Ningbo ###############################################
def prepare_data_ningbo(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if recreate_data is True:
        df_label_mappings = pd.read_excel(data_path/"Label mappings 2021.xlsx", sheet_name="Ningbo", dtype={"SNOMED code": str})
        df_label_mappings = df_label_mappings.dropna(subset=["SNOMED code"])
        
        dx_mapping_snomed = {}
        for code, diagnosis in zip(df_label_mappings["SNOMED code"], df_label_mappings["Diagnosis in the dataset"]):
            if code not in dx_mapping_snomed:
                dx_mapping_snomed[code] = diagnosis

        dx_mapping_snomed["106068003"] = "ARH"

        # for idx, item in enumerate(dx_mapping_snomed.items()):
        #     print(f'{idx+2} -> {item[0]}: {item[1]}')
        
        metadata = []
        for filename in tqdm(list(data_path.glob('*.hea'))):
            try:
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            except:
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename)[:-4]," data contains nans. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed[str(int(x))] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"ningbo"})
        df =pd.DataFrame(metadata)

        #lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        #lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        #df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label"])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### CPSC2018 #############################################
def prepare_data_cpsc2018(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if recreate_data is True:
        df_label_mappings = pd.read_excel(data_path/"Label mappings 2021.xlsx", sheet_name="CPSC", dtype={"SNOMED code": str})
        df_label_mappings = df_label_mappings.dropna(subset=["SNOMED code"])
        
        dx_mapping_snomed = {}
        for code, diagnosis in zip(df_label_mappings["SNOMED code"], df_label_mappings["Diagnosis in the dataset"]):
            if code not in dx_mapping_snomed:
                dx_mapping_snomed[code] = diagnosis

        dx_mapping_snomed["106068003"] = "ARH"

        # for idx, item in enumerate(dx_mapping_snomed.items()):
        #     print(f'{idx+2} -> {item[0]}: {item[1]}')
        
        metadata = []
        for filename in tqdm(list(data_path.glob('*.hea'))):
            try:
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            except:
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename)[:-4]," data contains nans. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed[str(int(x))] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"cpsc2018"})
        df =pd.DataFrame(metadata)

        #lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        #lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        #df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label"])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### CPSC-Extra ##########################################
def prepare_data_cpsc_extra(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if recreate_data is True:
        df_label_mappings = pd.read_excel(data_path/"Label mappings 2021.xlsx", sheet_name="CPSC-Extra", dtype={"SNOMED code": str})
        df_label_mappings = df_label_mappings.dropna(subset=["SNOMED code"])
        
        dx_mapping_snomed = {}
        for code, diagnosis in zip(df_label_mappings["SNOMED code"], df_label_mappings["Diagnosis in the dataset"]):
            if code not in dx_mapping_snomed:
                dx_mapping_snomed[code] = diagnosis

        # for idx, item in enumerate(dx_mapping_snomed.items()):
        #     print(f'{idx+2} -> {item[0]}: {item[1]}')
        
        metadata = []
        for filename in tqdm(list(data_path.glob('*.hea'))):
            try:
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            except:
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename)[:-4]," data contains nans. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed[str(int(x))] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"cpsc_extra"})
        df =pd.DataFrame(metadata)

        #lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        #lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        #df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label"])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### Georgia ############################################
def prepare_data_georgia(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if recreate_data is True:
        df_label_mappings = pd.read_excel(data_path/"Label mappings 2021.xlsx", sheet_name="G12EC", dtype={"SNOMED code": str})
        df_label_mappings = df_label_mappings.dropna(subset=["SNOMED code"])
        
        dx_mapping_snomed = {}
        for code, diagnosis in zip(df_label_mappings["SNOMED code"], df_label_mappings["Diagnosis in the dataset"]):
            if code not in dx_mapping_snomed:
                dx_mapping_snomed[code] = diagnosis

        # for idx, item in enumerate(dx_mapping_snomed.items()):
        #     print(f'{idx+2} -> {item[0]}: {item[1]}')
        
        metadata = []
        for filename in tqdm(list(data_path.glob('*.hea'))):
            try:
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            except:
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename)[:-4]," data contains nans. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed[str(int(x))] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"georgia"})
        df =pd.DataFrame(metadata)

        #lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        #lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        #df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label"])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### Chapman ###########################################
def prepare_data_chapman(data_path, min_cnt=10, denoised=False, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''prepares the Chapman dataset from Zheng et al 2020'''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        #df_attributes = pd.read_excel("./AttributesDictionary.xlsx")
        #df_conditions = pd.read_excel("./ConditionNames.xlsx")
        #df_rhythm = pd.read_excel("./RhythmNames.xlsx")
        df = pd.read_excel(data_path/"Diagnostics.xlsx")
        df["id"]=df.FileName
        df["data"]=df.FileName.apply(lambda x: x+".npy")
        df["label_condition"]=df.Beat.apply(lambda x: [y for y in x.split(" ") if x!="NONE"])
        df["label_rhythm"]=df.Rhythm.apply(lambda x: x.split(" "))
        df["label_all"]=df.apply(lambda row: row["label_condition"]+row["label_rhythm"],axis=1)
        df["sex"]=df.Gender.apply(lambda x:x.lower())
        df["age"]=df.PatientAge
        df.drop(["Gender","PatientAge","Rhythm","Beat","FileName"],inplace=True,axis=1)

        #map to numerical indices
        #lbl_itos={}
        #lbl_stoi={}
        #lbl_itos["all"] = np.unique([item for sublist in list(df.label_txt) for item in sublist])
        #lbl_stoi["all"] = {s:i for i,s in enumerate(lbl_itos["all"])}
        #df["label"] = df["label_txt"].apply(lambda x: [lbl_stoi["all"][y] for y in x])
        #lbl_itos["condition"] = np.unique([item for sublist in list(df.label_condition_txt) for item in sublist])
        #lbl_stoi["condition"] = {s:i for i,s in enumerate(lbl_itos["condition"])}
        #df["label_condition"] = df["label_condition_txt"].apply(lambda x: [lbl_stoi["condition"][y] for y in x])
        #lbl_itos["rhythm"] = np.unique([item for sublist in list(df.label_rhythm_txt) for item in sublist])
        #lbl_stoi["rhythm"] = {s:i for i,s in enumerate(lbl_itos["rhythm"])}
        #df["label_rhythm"] = df["label_rhythm_txt"].apply(lambda x: [lbl_stoi["rhythm"][y] for y in x])
        
        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label_all","label_condition","label_rhythm"])
        
        df["dataset"]="Chapman"

        for id,row in tqdm(list(df.iterrows())):
            fs = 500.

            df_tmp = pd.read_csv(data_path/("ECGDataDenoised" if denoised else "ECGData")/(row["id"]+".csv"))
            channel_labels = list(df_tmp.columns)
            sigbufs = np.array(df_tmp)*0.001 #assuming data is given in muV

            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels)
            assert(target_fs<=fs)
            np.save(target_root/(row["id"]+".npy"),data)

        stratified_ids = stratify(list(df["label_all"]), lbl_itos["label_all"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


####################################### SPH #############################################
def prepare_data_sph(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root_sph = Path(".") if target_folder is None else target_folder
    target_root_sph.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        df_sph = pd.read_csv(data_path/'metadata.csv')
        df_sph.columns = ["ecg_id","aha_code","patient_id","age","sex","n","date"]
        df_sph = df_sph.drop("n",axis=1)
        
        #human readable labels
        #description = pd.read_csv(data_path/'code.csv')#descriptions not used here
        #def aha_dict(desc):
        #    res={}
        #    for _,row in desc.iterrows():
        #        res[str(row["Code"])]=row["Description"]
        #    return res
        #def aha_to_text(t):
        #    tsplit=[x.split("+") for x in t.split(";")]
        #    print(tsplit)
        #    return ";".join(["+".join([ahad[y] for y in x]) for x in tsplit])

        # preparing labels
        df_sph["label_primary_w_mod"]=df_sph.aha_code.apply(lambda x: x.split(";"))
        df_sph["label_primary"]= df_sph.label_primary_w_mod.apply(lambda x: [y.split("+")[0] for y in x])

        df_sph["dataset"]="sph"
        #filter and map (can be reapplied at any time)
        df_sph, lbl_itos_sph =map_and_filter_labels(df_sph,min_cnt=min_cnt,lbl_cols=["label_primary","label_primary_w_mod"])

        channel_labels = ["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
        channel_itos=   ["x"]*channels
        for k in channel_stoi.keys():
            if(channel_stoi[k]<channels):
                channel_itos[channel_stoi[k]]=k
        chidxs=np.array([np.where(np.array(channel_labels)==c.lower())[0][0] for c in np.array(channel_itos)[:channels]])

        filenames = []
        for index in tqdm(range(len(df_sph))):
            record = df_sph.ecg_id[index]
            filename = data_path/"records"/(str(record)+".h5")
            with h5py.File(filename, 'r') as f:
                signal = f['ecg'][()].astype('float32') # K,L
                resampled_signal = resampy.resample(signal, 500, target_fs, axis=1)
                ordered_signal = resampled_signal[chidxs].T #L,K
                np.save(target_root_sph/(filename.stem+".npy"),ordered_signal)
            filenames.append(Path(filename.stem+".npy"))
        df_sph["data"] = filenames

        #stratified splits
        lbl_itos = lbl_itos_sph["label_primary_w_mod"]
        df_labeled = df_sph.copy()
        df_labeled["label_sex"] = df_labeled.sex.apply(lambda x: len(lbl_itos)+1 if x=="M" else len(lbl_itos)+2)
        df_labeled["label_age"] = df_labeled.age.apply(lambda x: len(lbl_itos)+3 if x<20 else (len(lbl_itos)+4 if x<40 else (len(lbl_itos)+5 if x<60 else (len(lbl_itos)+6 if x<80 else len(lbl_itos)+7 ))))

        df_labeled["labelx"] = df_labeled.apply(lambda row: row["label_primary_w_mod_numeric"]+[len(lbl_itos)]+[row["label_sex"]]+[row["label_age"]],axis=1)

        df_patients = df_labeled.groupby("patient_id")["labelx"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify(patients_labels, range(len(lbl_itos)+8), [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True)
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        df_sph["strat_fold"]=-1 #unlabeled will end up in fold -1
        for i,split in enumerate(stratified_patient_ids):
            df_sph.loc[df_sph.patient_id.isin(split),"strat_fold"]=i

        #original splits
        # 80%-20% split
        # put all records belonging to patients with
        # multiple records in the test set
        test1 = df_sph.patient_id.duplicated(keep=False)
        N = int(len(df_sph)*0.2) - sum(test1)
        # 73 is chosen such that all primary statements exist in both sets
        df_test = pd.concat([df_sph[test1], df_sph[~test1].sample(N, random_state=73)])
        df_sph["orig_fold"]=df_sph.ecg_id.apply(lambda x: int(x in np.array(df_test.ecg_id)))
        
        #add means and std
        dataset_add_mean_col(df_sph,data_folder=target_root_sph)
        dataset_add_std_col(df_sph,data_folder=target_root_sph)
        dataset_add_length_col(df_sph,data_folder=target_root_sph)
        
        #save means and stds
        mean_sph, std_sph = dataset_get_stats(df_sph)

        #save
        save_dataset(df_sph,lbl_itos_sph,mean_sph,std_sph,target_root_sph)
    else:
        df_sph, lbl_itos_sph, mean_sph, std_sph = load_dataset(target_root_sph,df_mapped=False)
    return df_sph, lbl_itos_sph, mean_sph, std_sph

###################################### CODE-15 #########################################

def prepare_data_ribeiro_full(data_path, code15=False, max_records_per_id_exam = 0, min_records_per_id_exam=0, overwrite_npy_if_exists=False, selected_folds=None, skip_strat_folds=False, subsample=None, target_fs=400, strat_folds=100, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    data_path = Path(data_path)
    target_root = Path(".") if target_folder is None else Path(target_folder)
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):

        #1 prepare filenames df
        print("Loading filenames...")
        if(code15):
            res=[]
            for f in data_path.glob('*.hdf5'):
                f5 = h5py.File(f, 'r')
                for i,x in enumerate(f5['exam_id']):
                    res.append({"filename":f, "id_row":i, "id_exam":x})
            df = pd.DataFrame(res)
        else:
            with open(data_path/"wfdb/RECORDS.txt", 'r') as file:
                filenames = file.readlines()

            res=[]
            for filename in tqdm(filenames,leave=False):
                filename = filename.strip()
                if(len(filename)>0):
                    filename = data_path/"wfdb"/filename
                    filename_stem = filename.stem

                    try:
                        id_exam = int(filename_stem.split("_N")[0][4:])
                        id_record = int(filename_stem.split("_N")[1])
                    except:
                        print("Error processing",filename,". Skipping.")
                        continue
                    res.append({"filename":filename.parent/filename.stem, "id_exam": id_exam, "id_record":id_record})
            #filename without suffix, id_exam matching annotations.csv id_record=1... (record within the same id_exam)
            df = pd.DataFrame(res)

        #2 load annotations
        print("Loading annotations...")
        if(code15):
            df_annotations=pd.read_csv(data_path/"exams.csv").set_index("exam_id")
        else:
            df_annotations=pd.read_csv(data_path/"annotations.csv").set_index("id_exam")

        #3 minor postprocessing
        df = df.join(df_annotations, how="left", on="id_exam")
        df["dataset"]="code15" if code15 else "code100"

        if(code15 is False):
            df["date_exam"]=pd.to_datetime(df["date_exam"])
        df["missing_label"] = df.ST.isna()
        if(code15):
            df["sex"] =df.is_male.apply(lambda x: "male" if x else "female")
        else:
            sex_map = {"M":"male", "F":"female"}
            df["sex"] = df.sex.apply(lambda x: sex_map[x] if x in sex_map.keys() else "unknown").astype("category")


        df["label"]=df[["1dAVb","RBBB","LBBB","SB","AF","ST"]].values.tolist()
        df["label"]=df.label.apply(lambda x: list(np.where(np.array(x)==1.)[0]))
        lbl_itos = ["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]
        if(code15):
            df.rename({'patient_id': 'id_patient'}, axis=1, inplace=True)
        print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients before filtering.")

        #4 filter if desired
        if(not code15 and (max_records_per_id_exam > 0 or min_records_per_id_exam > 0)):
            if(max_records_per_id_exam > 0):
                df = df[(df.id_record<=max_records_per_id_exam) & (df.id_record>=min_records_per_id_exam)].copy()
            else:
                df = df[df.id_record>=min_records_per_id_exam].copy()
            print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after filtering for max_records_per_id_exam.")

        #subsample based on patients if desired
        if(subsample is not None):
            df_labeled = df[df.missing_label==False].copy()
            patient_ids = np.array(df.id_patient.unique())
            patient_ids_selected = patient_ids[int(subsample[0]*len(patient_ids)):int(subsample[1]*len(patient_ids))]
            df_labeled = df_labeled[df_labeled.id_patient.isin(patient_ids_selected)]


            df_unlabeled = df[df.missing_label==True].copy()
            df_unlabeled=df_unlabeled.iloc[int(subsample[0]*len(df_unlabeled)):int(subsample[1]*len(df_unlabeled))]

            df = pd.concat([df_labeled,df_unlabeled])
            print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after subsampling.")


        #5 determine folds
        if(skip_strat_folds is False):
            print("Preparing labels for fold distribution...")
            df_labeled = df[df.missing_label==False].copy()
            df_labeled["label_sex"] = df_labeled.sex.apply(lambda x: len(lbl_itos)+1 if x=="male" else len(lbl_itos)+2)
            df_labeled["label_age"] = df_labeled.age.apply(lambda x: len(lbl_itos)+3 if x<20 else (len(lbl_itos)+4 if x<40 else (len(lbl_itos)+5 if x<60 else (len(lbl_itos)+6 if x<80 else len(lbl_itos)+7 ))))

            df_labeled["labelx"] = df_labeled.apply(lambda row: row["label"]+[len(lbl_itos)]+[row["label_sex"]]+[row["label_age"]],axis=1)

            df_patients = df_labeled.groupby("id_patient")["labelx"].apply(lambda x: list(x))
            patients_ids = list(df_patients.index)
            patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
            patients_num_ecgs = list(df_patients.apply(len))

            stratified_ids = stratify_batched(patients_labels, range(len(lbl_itos)+8), [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True,batch_size=10000)
            stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

            df["strat_fold"]=-1 #unlabeled will end up in fold -1
            for i,split in enumerate(stratified_patient_ids):
                df.loc[df.id_patient.isin(split),"strat_fold"]=i
            print("Fold distribution:",df.strat_fold.value_counts())
            if(selected_folds is not None):
                df=df[df.strat_fold.isin(selected_folds)].copy()
                print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after filtering for selected folds.")


        #6 actually prepare the files
        print("Reformating files...")
        filenames = []
        times=[]
        keeplist = []

        def process_single(i,df):
            filename = df.filename.iloc[i]
            keep = True
            if(code15):
                filename_out = "code15_"+str(df.id_exam.iloc[i])+".npy"
            else:
                filename_out = "code100_"+filename.stem+".npy"
            if(overwrite_npy_if_exists or not((target_root/filename_out).exists())):#recreate npy
                if(code15):
                    with h5py.File(filename, 'r') as f5:
                        sigbufs = np.array(f5['tracings'][df.id_row.iloc[i]])
                    start_idxs=np.where(np.sum(np.abs(sigbufs),axis=1)==0.)[0] #discard zeros at beginning/end
                    start_idx = len(start_idxs)//2
                    sigbufs = sigbufs[start_idx:-start_idx or None]
                    sig_names = "I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6".lower().split(",")
                    fs = 400
                else:
                    sigbufs, header = wfdb.rdsamp(str(filename))#str(row["filename"]))
                    sig_names = [x.lower().replace("d","") for x in header["sig_name"]]
                    fs = header['fs']
                if(len(sigbufs)==0):
                    print("Error processing id",i,"filename",filename.stem,"empty waveform. Will be dropped.")
                    keep = False
                elif(target_fs>fs):
                    print("Error processing id",i,"filename",filename.stem,"sampling frequency insufficient. Will be dropped.")
                    keep = False
                else:
                    data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=sig_names,fs=fs,target_fs=target_fs,channels=channels)
                    np.save(target_root/filename_out,data)
            else:# use existing file
                if(code15 is False):
                    record = wfdb.rdheader(str(filename))
                    header = {"base_date":record.base_date, "base_time":record.base_time}
            if(code15):
                time = pd.NaT
            else:    
                if(header["base_date"] is not None and header["base_time"] is not None):
                    time = datetime.datetime.combine(header["base_date"],header["base_time"])
                elif(header["base_date"] is not None):
                    time = header["base_date"]
                else:
                    time = pd.NaT
            return i, filename_out, time, keep

        #for id, row in tqdm(df.iterrows()):
        for i in tqdm(range(len(df))):
            filename = df.filename.iloc[i]
            _,filename_out,time,keep = process_single(i,df)
            times.append(time)
            filenames.append(filename_out)
            keeplist.append(keep)

        ####
        #import multiprocessing
        #from joblib import Parallel, delayed


        #num_cores = multiprocessing.cpu_count()
        #inputs = tqdm(range(len(df)))

        #processed_list = Parallel(n_jobs=num_cores)(delayed(process_single)(i,df) for i in inputs)
        #ids, filenames, times = zip(*processed_list)
        #df_result = pd.DataFrame({"id":ids,"data":filenames,"record_datetime":times}).set_index("id")
        #df["id"]= range(len(df))
        #df = df.join(df_result,on="id").drop("id",axis=1)


        df["data"] = filenames
        df["record_datetime"]=times
        df["keep"]=keeplist

        df= df[df["keep"]].copy()
        df.drop(["keep"],axis=1,inplace=True)

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)
        #dataset_add_median_col(df,data_folder=target_root)
        #dataset_add_iqr_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df,lbl_itos,mean,std,target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

###################################### PTB-XL ################################################
def prepare_data_ptb_xl(data_path, min_cnt=10, target_fs=500, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        ptb_xl_csv = data_path/"ptbxl_database.csv"
        df_ptb_xl=pd.read_csv(ptb_xl_csv,index_col="ecg_id")
        #print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path/"scp_statements.csv")
        ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic >0]
        ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form >0]
        ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm >0]

        diag_class_mapping={}
        diag_subclass_mapping={}
        for id,row in ptb_xl_label_diag.iterrows():
            if(isinstance(row["diagnostic_class"],str)):
                diag_class_mapping[id]=row["diagnostic_class"]
            if(isinstance(row["diagnostic_subclass"],str)):
                diag_subclass_mapping[id]=row["diagnostic_subclass"]

        df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"]="ptb_xl"
        #filter and map (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl =map_and_filter_labels(df_ptb_xl,min_cnt=min_cnt,lbl_cols=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"])

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            # always start from 500Hz and sample down
            filename = data_path/row["filename_hr"] #data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root_ptb_xl/(filename.stem+".npy"),data)
            filenames.append(Path(filename.stem+".npy"))
        df_ptb_xl["data"] = filenames

        #add means and std
        dataset_add_mean_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        #save
        save_dataset(df_ptb_xl,lbl_itos_ptb_xl,mean_ptb_xl,std_ptb_xl,target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl

####################################### ZZU pECG #############################################
def prepare_data_zzu_pecg(data_path, min_cnt=10, target_fs=500, strat_folds=10, channels=12, 
                         channel_stoi=channel_stoi_canonical, target_folder=None, 
                         recreate_data=True):
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if not recreate_data:
        return load_dataset(target_root, df_mapped=False)

    df = pd.read_csv(data_path/"AttributesDictionary.csv")
    df_disease = pd.read_csv(data_path/"DiseaseCode.csv")
    df_ecg = pd.read_csv(data_path/"ECGCode.csv")
    df.columns = df.columns.str.lower()
    df_disease.columns = df_disease.columns.str.lower()
    df_ecg.columns = df_ecg.columns.str.lower()

    df['age'] = df['age'].str.replace('d','').astype(int)
    df['gender'] = df['gender'].str.replace("'", "").str.lower().map(
        {'male':1,'female':0}).fillna(-1)
    for col in ['aha_code','chn_code','icd-10 code']:
        df[col] = df[col].apply(lambda x: [] if pd.isna(x) or x=='Null' 
                                else [c.strip().replace("'", "") for c in str(x).split(';') if c.strip()])

    type_map, cat_map = {}, {}
    for _, row in df_disease.iterrows():
        for code in str(row['icd-10 code']).split(';'):
            code = code.strip()
            if code: 
                type_map[code] = row['disease type']
                cat_map[code] = row['disease category']
    
    df['icd10_disease_type'] = df['icd-10 code'].apply(
        lambda codes: [type_map.get(c, 'Unknown') for c in codes if c in type_map]
    )
    df['icd10_disease_category'] = df['icd-10 code'].apply(
        lambda codes: [cat_map.get(c, 'Unknown') for c in codes if c in cat_map]
    )

    aha_map, chn_map = {}, {}
    for _, row in df_ecg.iterrows():
        desc = str(row['description']).strip()
        aha = str(row['aha(category&code)']).strip()
        chn = str(row['chn(category&code)']).strip()
        if aha not in ['N/A','nan']: aha_map[aha] = desc
        if chn not in ['N/A','nan']: chn_map[chn] = desc
    
    df['aha_description'] = df['aha_code'].apply(
        lambda codes: [aha_map.get(c) for c in codes if c in aha_map]
    )
    df['chn_description'] = df['chn_code'].apply(
        lambda codes: [chn_map.get(c) for c in codes if c in chn_map]
    )

    df_12 = df[df['lead']==12].copy()

    def process_ecg_file(row, data_path, target_root, channel_stoi, target_fs, channels):
        try:
            rec = (Path(data_path/"Child_ecg")/row['filename']).with_suffix('')
            sigs, header = wfdb.rdsamp(str(rec))
            data = resample_data(
                sigbufs=sigs,
                channel_labels=[l.lower() for l in header['sig_name']],
                fs=header['fs'],
                target_fs=target_fs,
                channels=channels,
                channel_stoi=channel_stoi
            )
            out_file = target_root/f"{rec.name}.npy"
            np.save(out_file, data)

            return out_file.name
        except Exception as e:
            print(f"Error {row['filename']}: {e}")
            return None

    files_12 = []
    for _, row in tqdm(df_12.iterrows(), total=len(df_12), desc="Processing 12-lead"):
        files_12.append(process_ecg_file(row, data_path, target_root, channel_stoi, target_fs, channels))
    df_12["data"] = files_12

    df_12["dataset"] = "zzu_pecg"

    dataset_add_mean_col(df_12, data_folder=target_root)
    dataset_add_std_col(df_12, data_folder=target_root)
    dataset_add_length_col(df_12, data_folder=target_root)

    lbl_cols = ["icd10_disease_category","aha_description","chn_description"]
    df_12, lbl_itos = map_and_filter_labels(df_12, min_cnt, lbl_cols)

    lbl_itos_main = lbl_itos["icd10_disease_category_filtered"]
    df_labeled = df_12.copy()
    
    df_labeled["label_gender"] = df_labeled.gender.apply(
        lambda x: len(lbl_itos_main) + 1 if x == 1 else len(lbl_itos_main) + 2
    )
    
    df_labeled["label_age"] = df_labeled.age.apply(
        lambda x: len(lbl_itos_main) + 3 if x < 365 else
                 len(lbl_itos_main) + 4 if x < 365*5 else
                 len(lbl_itos_main) + 5 if x < 365*12 else
                 len(lbl_itos_main) + 6 if x < 365*18 else
                 len(lbl_itos_main) + 7
    )
    
    df_labeled["labelx"] = df_labeled.apply(
        lambda row: row["icd10_disease_category_filtered_numeric"] + 
                   [len(lbl_itos_main)] +
                   [row["label_gender"]] + 
                   [row["label_age"]], 
        axis=1
    )

    df_patients = df_labeled.groupby("patient_id")["labelx"].apply(lambda x: list(x))
    patients_ids = list(df_patients.index)
    patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
    patients_num_ecgs = list(df_patients.apply(len))

    stratified_ids = stratify(
        patients_labels, 
        range(len(lbl_itos_main) + 8),
        [1./strat_folds] * strat_folds, 
        samples_per_group=patients_num_ecgs,
        verbose=True
    )
    
    stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

    df_12["strat_fold"] = -1
    for i, patient_split in enumerate(stratified_patient_ids):
        df_12.loc[df_12.patient_id.isin(patient_split), "strat_fold"] = i

    mean, std = dataset_get_stats(df_12)
    save_dataset(df_12, lbl_itos, mean, std, target_root)
    
    return df_12, lbl_itos, mean, std

###################################### EchoNext #############################################
def prepare_data_echonext(data_path, target_folder=None, channel_stoi=channel_stoi_canonical, 
                         channels=12, target_fs=250, strat_folds=10, cv_random_state=42, 
                         recreate_data=True):
    data_path = Path(data_path)
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)
    
    if recreate_data:
        print("Loading metadata...")
        metadata_file = data_path / "EchoNext_metadata_100k.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        df["dataset"] = "echonext"
        
        splits = ["train", "val", "test", "no_split"]
        waveform_data = {}
        tabular_data = {}
        
        for split in splits:
            waveform_file = data_path / f"EchoNext_{split}_waveforms.npy"
            tabular_file = data_path / f"EchoNext_{split}_tabular_features.npy"
            
            if waveform_file.exists():
                waveform_data[split] = np.load(waveform_file)
            if tabular_file.exists():
                tabular_data[split] = np.load(tabular_file)
        
        # Process each split
        df["data"] = None

        preprocessed_feature_names = [
            "sex_preprocessed",
            "ventricular_rate_preprocessed",
            "atrial_rate_preprocessed",
            "pr_interval_preprocessed",
            "qrs_duration_preprocessed",
            "qt_corrected_preprocessed",
            "age_at_ecg_preprocessed"
        ]

        current_idx = 0
        
        for split in splits:
            if split in waveform_data:
                split_waveforms = waveform_data[split]
                split_tabular = tabular_data.get(split, None)

                print(f"Processing {split} data...")
                
                for i in tqdm(range(len(split_waveforms))):
                    meta_idx = current_idx + i
                    if meta_idx < len(df):
                        meta_row = df.iloc[meta_idx]                        
                        filename_out = f"echonext_{meta_row['ecg_key']}.npy"
                        
                        sigbufs = split_waveforms[i][0]
                        data = resample_data(
                            sigbufs=sigbufs,
                            channel_stoi=channel_stoi,
                            channel_labels=["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"],
                            fs=250,
                            target_fs=target_fs,
                            channels=channels
                        )                        
                        np.save(target_root / filename_out, data)
                        
                        df.at[meta_idx, "data"] = filename_out

                        if split_tabular is not None:
                            for j, col in enumerate(preprocessed_feature_names):
                                df.at[meta_idx, col] = split_tabular[i, j]
                
                current_idx += len(split_waveforms)

        df = df[df["data"].notna()].copy()
        
        # Create label
        lbl_itos = [
            "lvef_lte_45_flag",
            "lvwt_gte_13_flag",
            "aortic_stenosis_moderate_or_greater_flag",
            "aortic_regurgitation_moderate_or_greater_flag",
            "mitral_regurgitation_moderate_or_greater_flag",
            "tricuspid_regurgitation_moderate_or_greater_flag",
            "pulmonary_regurgitation_moderate_or_greater_flag",
            "rv_systolic_dysfunction_moderate_or_greater_flag",
            "pericardial_effusion_moderate_large_flag",
            "pasp_gte_45_flag",
            "tr_max_gte_32_flag"
        ]
        df["label"]=df[lbl_itos].values.tolist()
        df["label"]=df.label.apply(lambda x: list(np.where(np.array(x)==1.)[0]))        
        
        # For create stratified folds
        df["strat_fold"] = -2
        df.loc[df["split"] == "val", "strat_fold"] = strat_folds - 2
        df.loc[df["split"] == "test", "strat_fold"] = strat_folds - 1
        empty_mask = df['label'].apply(lambda x: len(x) == 0)
        df.loc[empty_mask, "strat_fold"] = -1

        train_mask = df["split"].isin(["train"])
        df_train= df[train_mask].copy()        
        if len(df_train) >= strat_folds-2:
            df_patients = df_train.groupby("patient_key")["label"].apply(lambda x: list(x))
            patients_ids = list(df_patients.index)
            patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
            patients_num_ecgs = list(df_patients.apply(len))            
            
            stratified_ids = stratify(patients_labels, list(range(len(lbl_itos))), [1./(strat_folds-2)]*(strat_folds-2), 
                                    samples_per_group=patients_num_ecgs, random_seed=cv_random_state)
            stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]            
            

            for i, split in enumerate(stratified_patient_ids):
                df.loc[df.patient_key.isin(split), "strat_fold"] = i
        
        #add means and std
        dataset_add_mean_col(df, data_folder=target_root)
        dataset_add_std_col(df, data_folder=target_root)
        dataset_add_length_col(df, data_folder=target_root)
        
        # save means and stds
        mean, std = dataset_get_stats(df)
        
        # save
        save_dataset(df, lbl_itos, mean, std, target_root)
        
    else:
        df, lbl_itos, mean, std = load_dataset(target_root, df_filename="df_memmap.pkl")
    
    return df, lbl_itos, mean, std

################################## MIMIC-IV-ECG ##########################################
def prepare_mimicecg(data_path="", clip_amp=3, target_fs=500, channels=12, strat_folds=20, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    
    if(recreate_data):
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(Path(data_path)/"mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip", 'r') as archive:
            lst = archive.namelist()
            lst = [x for x in lst if x.endswith(".hea")]

            meta = []
            for l in tqdm(lst):
                archive.extract(l, path="tmp_dir/")
                archive.extract(l[:-3]+"dat", path="tmp_dir/")
                filename = Path("tmp_dir")/l
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            
                tmp={}
                tmp["data"]=filename.parent.parent.stem+"_"+filename.parent.stem+".npy" #patientid_study.npy
                tmp["study_id"]=int(filename.stem)
                tmp["subject_id"]=int(filename.parent.parent.stem[1:])
                tmp['ecg_time']= datetime.datetime.combine(header["base_date"],header["base_time"])
                tmp["nans"]= list(np.sum(np.isnan(sigbufs),axis=0))#save nans channel-dependent
                if(np.sum(tmp["nans"])>0):#fix nans
                    fix_nans_and_clip(sigbufs,clip_amp=clip_amp)
                elif(clip_amp>0):
                    sigbufs = np.clip(sigbufs,a_max=clip_amp,a_min=-clip_amp)

                data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
                
                assert(target_fs<=header['fs'])
                np.save(target_folder/tmp["data"],data)
                meta.append(tmp)
                
                os.unlink("tmp_dir/"+l)
                os.unlink("tmp_dir/"+l[:-3]+"dat")
                shutil.rmtree("tmp_dir")

        df = pd.DataFrame(meta)

        #random split by patients
        #unique_patients = np.unique(df.subject_id)
        #splits_patients = get_stratified_kfolds(np.zeros(len(unique_patients)),n_splits=strat_folds,random_state=42)
        #df["fold"]=-1
        #for i,split in enumerate(splits_patients):
        #    df.loc[df.subject_id.isin(unique_patients[split[-1]]),"fold"]=i
        
        #add means and std
        dataset_add_mean_col(df,data_folder=target_folder)
        dataset_add_std_col(df,data_folder=target_folder)
        dataset_add_length_col(df,data_folder=target_folder)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        lbl_itos=[]
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std
       
################################## HEEDB #################################################
def process_ecg_file_heedb(row, data_path, target_folder, target_fs, channels, channel_stoi, clip_amp, recreate_numpy):
    try:
        filename = Path(data_path) / "ECG/WFDB" / row["FileName"]
        filename_new = filename.stem + ".npy"
        
        # Check if we can skip processing
        if not recreate_numpy and (Path(target_folder) / filename_new).exists():
            return {"file_source": row["file_source"], "data": str(filename_new)}
        
        # Validate header file exists
        if not filename.exists():
            print(f"Header {filename} does not exist.")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Validate signal file exists
        signal_file = filename.parent / (filename.stem + ".mat")
        if not signal_file.exists():
            print(f"Signal file corresponding to {filename} does not exist.")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Read the signal
        try:
            sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
        except Exception as e:
            error_message = str(e) if str(e) else "Unknown error reading signal file"
            print(f"Warning: could not read {filename}. Error: {error_message}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Check if header contains required fields
        if 'fs' not in header or 'sig_name' not in header:
            print(f"Missing required fields in header for {filename}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Handle NaNs and clipping
        if np.sum(np.isnan(sigbufs)) > 0:
            try:
                fix_nans_and_clip(sigbufs, clip_amp=clip_amp)
            except Exception as e:
                print(f"Error fixing NaNs in {filename}: {str(e)}")
                return {"file_source": row["file_source"], "data": "nan"}
        elif clip_amp > 0:
            sigbufs = np.clip(sigbufs, a_max=clip_amp, a_min=-clip_amp)
        
        # Validate target frequency
        if target_fs > header['fs']:
            print(f"Target frequency {target_fs} is higher than original {header['fs']} for {filename}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Resample data
        try:
            data = resample_data(
                sigbufs=sigbufs,
                channel_stoi=channel_stoi,
                channel_labels=header['sig_name'],
                fs=header['fs'],
                target_fs=target_fs,
                channels=channels
            )
        except Exception as e:
            print(f"Error resampling data for {filename}: {str(e)}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Save processed data
        try:
            output_path = Path(target_folder) / filename_new
            np.save(output_path, data)
        except Exception as e:
            print(f"Error saving data for {filename}: {str(e)}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        return {"file_source": row["file_source"], "data": str(filename_new)}
        
    except Exception as e:
        # Make sure we get a string representation of the error
        error_message = str(e) if str(e) else "Empty exception message"
        traceback_info = traceback.format_exc()
        print(f"Error processing {row.get('FileName', 'unknown file')}: {error_message}")
        print(f"Traceback: {traceback_info}")
        return {"file_source": row.get("file_source", "unknown"), "data": "nan"}
    
#['S0001-1987', 'S0001-1988', 'S0001-1994', 'S0001-2006', 'S0001-2007', 'S0001-2011', 'S0001-2013', 'S0001-2015', 'S0001-2016', 'S0001-2017', 'S0001-2018', 'S0001-2019', 'S0001-2020', 'S0001-2021', 'S0002-1993', 'S0002-1997', 'S0002-1998', 'S0002-2000', 'S0002-2003', 'S0002-2004', 'S0002-2008', 'S0002-2009', 'S0002-2013', 'S0002-2015', 'S0002-2016', 'S0002-2018', 'S0002-2020', 'S0002-2022', 'S0003-1990', 'S0003-1994', 'S0003-2009', 'S0003-2010', 'S0003-2013', 'S0004-2019']
def prepare_heedb(data_path="", partition="S0002-1998",clip_amp=3, target_fs=240, channels=12, strat_folds=20, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True, max_workers=8, recreate_numpy=False, skip_metadata=False):

    def prepare_metadata(partition,data_path):
        print("Reading metadata table...")
        df_demo=pd.read_csv(Path(data_path)/"Metadata/demographics_ECG.csv", low_memory=False)
        df_demo["partition"]=df_demo.FileName.apply(lambda x: "-".join(x.split("/")[1:3]))
        df_demo=df_demo[df_demo.partition==partition]
        df_demo["file_source"]=df_demo.FileName.apply(lambda x:x.split("/")[-1][:-len(".hea")])
        print(len(df_demo),"samples in the selection.")
        print("Reading ECG interpretation statements...")
        df_diag= pd.read_csv(Path(data_path)/"Metadata/ECG_Interpretations.psv",delimiter="|")
        df_diag["diagnostic_codes"]=df_diag["diagnostic_codes"].apply(lambda x: [int(y) for y in x.split(",")])
        labels= [x for xs in list(df_diag["diagnostic_codes"]) for x in xs]
        lbls, cnts = np.unique(labels, return_counts=True)
        lbl_itos = lbls[np.argsort(cnts)[::-1]]
        print("Reading duplicated list...")
        df_dupl=pd.read_csv(Path(data_path)/"Metadata/duplicated.csv")
        dupl_lst = list(df_dupl.file_name.apply(lambda x:x.split("/")[-1]))
        print("Joining tables...")
        df_demo=df_demo.join(df_diag.set_index("file_source"),on="file_source",how="left")
        #remove nans (also due to non-matching entries)
        df_demo["diagnostic_codes"]=df_demo["diagnostic_codes"].apply(lambda x: x if isinstance(x,list) else [])
        df_demo["PatientRace"]=df_demo["PatientRace"].replace({np.nan: 'Unavailable'})
        df_demo["SexDSC"]=df_demo["SexDSC"].replace({np.nan: 'Unknown'})
        df_demo=df_demo[~df_demo.file_source.isin(dupl_lst)].copy()#remove duplicated
        del df_diag
        return df_demo, lbl_itos
    
    
    if(recreate_data):
        print("Processing partition",partition)
        data_path = Path(data_path)
        target_folder = Path(target_folder)/partition
        (target_folder).mkdir(parents=True, exist_ok=True)
        if(not skip_metadata):
            print("Preparing metadata...")
            df, lbl_itos = prepare_metadata(partition,data_path)
            print("\nDone. Preparing signals... ")
        else:
            print("Preparing file list...")
            extra_path = "ECG/WFDB/"+"/".join(partition.split("-"))
            hea_files = []
            for root, dirs, filenames in os.walk(str(data_path/extra_path)):
                for filename in filenames:
                    if filename.endswith('.hea'):
                        hea_files.append(os.path.join(root, filename))
            #hea_files = list((data_path/extra_path).rglob('*.hea'))
            df = pd.DataFrame(hea_files, columns=['FileName'])
            df['FileName'] = df['FileName'].astype(str)
            df["file_source"]=df.FileName.apply(lambda x:x.split("/")[-1][:-len(".hea")])
            
        filenames_new = []

        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing jobs
            futures = {
                executor.submit(
                    process_ecg_file_heedb, 
                    row, 
                    data_path, 
                    target_folder, 
                    target_fs, 
                    channels, 
                    channel_stoi, 
                    clip_amp,
                    recreate_numpy
                ): row for _, row in df.iterrows()
            }
            
            # Tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    filenames_new.append(result)
        
        df_files = pd.DataFrame(filenames_new)
        
        if(skip_metadata):
            return df_files, [], None, None
        
        df=df.join(df_files.set_index("file_source"),on="file_source",how="left")
        
        if(len(df[df["data"]=="nan"])>0):
            print("The following files could not be processed:",list(df[df["data"]=="nan"]["FileName"]),"\nThe files will be skipped.")
            df=df[df["data"]!="nan"].copy()

        #fold asignments
        print("Preparing stratified fold distribution...")
        lbl_itos_strat = [str(x) for x in lbl_itos]+["sex_"+x for x in df["SexDSC"].unique()]+["race_"+x for x in df["PatientRace"].unique()]+["age_unknown","age_18","age_30","age_40","age_50","age_60","age_70","age_80","age_90","age_100"]
        def age_to_str(age):
            if(np.isnan(age) or age<0):
                return "age_unknown"
            elif(age<18):
                return "age_18"
            elif(age<30):
                return "age_30"
            elif(age<40):
                return "age_40"
            elif(age<50):
                return "age_50"
            elif(age<60):
                return "age_60"
            elif(age<70):
                return "age_70"
            elif(age<80):
                return "age_80"
            elif(age<90):
                return "age_90"
            else:
                return "age_100"
           
        df["label_strat"] = df.apply(lambda row: [str(y) for y in row["diagnostic_codes"]] + 
                ["sex_" + row["SexDSC"]] + 
                ["race_" + row["PatientRace"]] + 
                [age_to_str(row["Age"])],
                axis=1)
        df_patients = df.groupby("BDSPPatientID")["label_strat"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify_batched(patients_labels, lbl_itos_strat, [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True,batch_size=min(10000,len(df_patients)))#todo fix this in stratify
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        print("Applying fold assignments...")
        df["strat_fold"]=-1 #unlabeled will end up in fold -1
        for i,split in enumerate(stratified_patient_ids):
            df.loc[df.BDSPPatientID.isin(split),"strat_fold"]=i
        print("Fold distribution:",df.strat_fold.value_counts())
        print("In total",len(df),"records from ",len(df.BDSPPatientID.unique()),"patients.")
        df.drop("label_strat",axis=1, inplace=True)

        #add means and std
        #print("Calculating dataset stats...")
        #dataset_add_mean_col(df,data_folder=target_folder)
        #dataset_add_std_col(df,data_folder=target_folder)
        #dataset_add_length_col(df,data_folder=target_folder)

        #save means and stds
        #mean, std = dataset_get_stats(df)
        #dummies for now
        mean = np.zeros((12),dtype=np.float32)
        std = np.ones((12),dtype=np.float32)
        

        #save
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std
