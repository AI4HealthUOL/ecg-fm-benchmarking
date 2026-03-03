import pandas as pd
import numpy as np
import icd10 as icd
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, mean_absolute_error

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

import subprocess
from pathlib import Path

from code.clinical_ts.utils.mimic_ecg_preprocessing import prepare_mimic_ecg
from code.clinical_ts.utils.stratify import *


# load
df = pd.read_csv('data/records_w_diag_icd10.csv')
df['data'] = df.index

# path
target_folder = Path('mimic')
for c in ["hosp_diag_hosp" ,"ed_diag_ed", "ed_diag_hosp", "all_diag_hosp", "all_diag_all"]:
    df[c]=df[c].apply(lambda x:eval(x))
    
finetune_dataset = 'mimic_ed_all_edfirst_all_2000_5A'

df_diags_cols = [
    'file_name', 'ed_stay_id', 'ed_hadm_id', 'hosp_hadm_id',
    'ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp',
    'all_diag_hosp', 'all_diag_all', 'gender', 'age',
    'anchor_year', 'anchor_age', 'dod',
    'ecg_no_within_stay', 'ecg_taken_in_ed',
    'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp',
    'fold', 'strat_fold'
]

overlap = df.columns.intersection(df_diags_cols)
df_clean = df.drop(columns=overlap)

df_scenario, lbl_itos_diagnostic = prepare_mimic_ecg(
    finetune_dataset,
    target_folder,
    df_mapped=df_clean
)

df["is_diagnostic"] = df["data"].isin(df_scenario["data"]).astype(int)
mapping = df_scenario.set_index("data")["label"]
df["label_diagnostic"] = df["data"].map(mapping)

cardiac_chapter = 'IX'
def get_chapter(icd_code):
    try:
        return icd.find(icd_code).chapter
    except:
        return 'unknown'

cardiac_labels = [code for code in lbl_itos_diagnostic if get_chapter(code) == cardiac_chapter]
noncardiac_labels = [code for code in lbl_itos_diagnostic if get_chapter(code) != cardiac_chapter]
np.save('lbl_itos_diagnostic.npy', lbl_itos_diagnostic)


# load
dfed = pd.read_csv('data/mds_ed.csv', 
                   low_memory=False)

deterioration_columns = [i for i in dfed.columns if 'deterioration' in i]
dfed = dfed[deterioration_columns+['general_subject_id','general_data','general_strat_fold']]
dfed.columns = deterioration_columns+['subject_id','data','strat_fold']

for c in deterioration_columns:
    dfed[c] = dfed[c].replace(-999., np.nan)

dfed["label_deterioration"]=dfed[deterioration_columns].values.tolist()
lbl_itos_deterioration = np.array(deterioration_columns)

df["is_deterioration"] = df["data"].isin(dfed["data"]).astype(int)
mapping = dfed.set_index("data")["label_deterioration"]
df["label_deterioration"] = df["data"].map(mapping)

np.save('lbl_itos_deterioration.npy', lbl_itos_deterioration)


# load
dfecgfeatures = pd.read_csv('data/machine_measurements.csv', 
                            low_memory=False)
dfecgfeatures['data'] = dfecgfeatures.index

dfecgfeatures.loc[(dfecgfeatures['qrs_axis'] < -360) | (dfecgfeatures['qrs_axis'] > 360), 'qrs_axis'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['t_axis'] < -360) | (dfecgfeatures['t_axis'] > 360), 't_axis'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['p_axis'] < -360) | (dfecgfeatures['p_axis'] > 360), 'p_axis'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['p_onset'] < 0) | (dfecgfeatures['p_onset'] > 5000), 'p_onset'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['p_end'] < 0) | (dfecgfeatures['p_end'] > 5000), 'p_end'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['qrs_onset'] < 0) | (dfecgfeatures['qrs_onset'] > 5000), 'qrs_onset'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['qrs_end'] < 0) | (dfecgfeatures['qrs_end'] > 5000), 'qrs_end'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['t_end'] < 0) | (dfecgfeatures['t_end'] > 5000), 't_end'] = np.nan
dfecgfeatures.loc[(dfecgfeatures['rr_interval'] < 0) | (dfecgfeatures['rr_interval'] > 5000), 'rr_interval'] = np.nan
dfecgfeatures = dfecgfeatures.rename(columns={'rr_interval':'RR',
                                   'p_axis':'P_wave_axis',
                                   'qrs_axis':'QRS_axis',
                                   't_axis':'T_wave_axis',
                                   'gender':'sex'})
dfecgfeatures['PR'] = dfecgfeatures['qrs_onset'] - dfecgfeatures['p_onset']
dfecgfeatures['QRS'] = dfecgfeatures['qrs_end'] - dfecgfeatures['qrs_onset'] 
dfecgfeatures['QT'] = dfecgfeatures['t_end'] - dfecgfeatures['qrs_onset']
dfecgfeatures['QTc'] = np.where(dfecgfeatures['RR'] != 0, dfecgfeatures['QT'] / np.sqrt(dfecgfeatures['RR'] / 1000), np.nan)

dfecgfeatures = dfecgfeatures[['data','RR','QRS','QT','QTc','P_wave_axis','QRS_axis','T_wave_axis']]
df = df.merge(dfecgfeatures, on="data", how="left")

# load
omr = pd.read_csv('data/omr.csv.gz')
omr = omr[omr['result_name'].isin(['Height (Inches)','Weight (Lbs)','BMI (kg/m2)'])]
omr = omr.dropna(subset=['result_value'])

# load
vital = pd.read_csv('data/vitalsign.csv.gz')
vital = vital[['subject_id', 'stay_id', 'charttime', 'temperature', 'heartrate','resprate', 'o2sat', 'sbp', 'dbp']]
vital['charttime'] = pd.to_datetime(vital['charttime'])

# load
dflabitems = pd.read_csv('data/d_labitems.csv.gz')
dflabitems = dflabitems[dflabitems['itemid'].isin([50963,51006,52647,50811,51222,51640,50912,52546,50924,50912,52546,51221,51480,51638,51639,52028,
         50862,53085,51006,52647,52172,50811,51222,51640,50868,52500,51277,50882,50885,53089,51221,51480,
         51638,51639,52028,51237,51675,51279,51274,52921,50910,51249,50893,51244])]

# load
dflabevents = pd.read_csv('data/labevents.csv.gz')
dflabevents = dflabevents[dflabevents['itemid'].isin(dflabitems['itemid'].unique())]
dflabevents = dflabevents[dflabevents['valuenum'].notna()]
dflabevents = dflabevents.merge(dflabitems[['itemid', 'label']], on='itemid', how='left')
pair_counts = dflabevents.groupby(['label', 'itemid']).size().reset_index(name='count')
idx = pair_counts.groupby('label')['count'].idxmax()
most_common_pairs = pair_counts.loc[idx, ['label', 'itemid']]
dflabevents = dflabevents[dflabevents.set_index(['label','itemid']).index.isin(most_common_pairs.set_index(['label','itemid']).index)]
filtered = []
for label in dflabevents['label'].unique():
    sub = dflabevents[dflabevents['label'] == label]
    lower = sub['valuenum'].quantile(0.01)
    upper = sub['valuenum'].quantile(0.99)
    sub_filtered = sub[(sub['valuenum'] >= lower) & (sub['valuenum'] <= upper)]
    filtered.append(sub_filtered)
dflabevents = pd.concat(filtered, ignore_index=True)
uom_counts = dflabevents.groupby(['itemid', 'valueuom']).size().reset_index(name='count')
idx = uom_counts.groupby('itemid')['count'].idxmax()
most_common_uom = uom_counts.loc[idx, ['itemid', 'valueuom']]
dflabevents = dflabevents.merge(most_common_uom, on=['itemid', 'valueuom'], how='inner')
dflabevents = dflabevents[dflabevents['subject_id'].isin(df['subject_id'].unique())]
dflabevents['storetime'] = pd.to_datetime(dflabevents['storetime'])

dflabevents = dflabevents[['labevent_id','subject_id','storetime','valuenum','valueuom','label']]



# load

d_items_file = 'data/d_items.csv.gz'
chartevents_file = 'data/chartevents.csv.gz'

output_file = 'data/filtered_chartevents.csv'

chunksize = 1_000_000
min_label_count = 1000
d_items = pd.read_csv(d_items_file, compression='gzip', low_memory=False)
d_items_subset = d_items[['itemid', 'label']]
relevant_subject_ids = set(df['subject_id'].unique())
label_counts = {}
chartevents_iter = pd.read_csv(
    chartevents_file,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize)
print("Counting label occurrences...")
for chunk in tqdm(chartevents_iter):
    chunk = chunk[chunk['subject_id'].isin(relevant_subject_ids)]
    if chunk.empty:
        continue
    chunk = chunk.merge(d_items_subset, on='itemid', how='left')
    chunk = chunk[chunk['label'] != 'Safety Measures']
    counts = chunk['label'].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count
labels_to_keep = {label for label, count in label_counts.items() if count >= min_label_count}
chartevents_iter = pd.read_csv(
    chartevents_file,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)
print("Filtering and writing chunks to disk...")
for chunk in tqdm(chartevents_iter):
    chunk = chunk[chunk['subject_id'].isin(relevant_subject_ids)]
    if chunk.empty:
        continue
    chunk = chunk.merge(d_items_subset, on='itemid', how='left')
    chunk = chunk[(chunk['label'] != 'Safety Measures') & (chunk['label'].isin(labels_to_keep))]
    if not chunk.empty:
        chunk.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),
            index=False)
        

to_extract = [
    'Height (cm)', 'Height', 'Daily Weight',
    'Admission Weight (lbs.)', 'Admission Weight (Kg)',
    'Temperature Celsius', 'Temperature Fahrenheit',
    'Heart Rate', 'Respiratory Rate',
    'PAR-Oxygen saturation', 'O2 saturation pulseoxymetry',
    'Albumin', 'Anion Gap', 'Total Bilirubin',
    'Creatinine (serum)', 'Hematocrit (serum)', 'Hemoglobin']


# load

chunksize = 1_000_000
filtered_iter = pd.read_csv(
    'data/filtered_chartevents.csv',
    chunksize=chunksize)

dfs = [] 
for chunk in tqdm(filtered_iter):
    filtered_chunk = chunk[chunk['label'].isin(to_extract)]
    dfs.append(filtered_chunk)
filtered_df = pd.concat(dfs, ignore_index=True)

mask = filtered_df['label'] == 'Admission Weight (Kg)'
filtered_df.loc[mask, 'valuenum'] = filtered_df.loc[mask, 'valuenum'] * 2.20462
filtered_df.loc[mask, 'label'] = 'Weight (lbs)'
filtered_df.loc[mask, 'valueuom'] = 'lbs'

mask = filtered_df['label'] == 'Height (cm)'
filtered_df.loc[mask, 'valuenum'] = filtered_df.loc[mask, 'valuenum'] * 0.393701
filtered_df.loc[mask, 'label'] = 'Height (Inches)'
filtered_df.loc[mask, 'valueuom'] = 'Inch'

mask = filtered_df['label'] == 'Daily Weight'
filtered_df.loc[mask, 'valuenum'] = filtered_df.loc[mask, 'valuenum'] * 2.20462
filtered_df.loc[mask, 'label'] = 'Weight (lbs)'
filtered_df.loc[mask, 'valueuom'] = 'lbs'

filtered_df = filtered_df[filtered_df['label'] != 'PAR-Oxygen saturation']

labels_to_clean = [
    'Albumin', 'Total Bilirubin', 'Hematocrit (serum)',
    'Creatinine (serum)', 'Weight (lbs)'
]
filtered_df = filtered_df[~((filtered_df['label'].isin(labels_to_clean)) &
                            (filtered_df['valueuom'].isna()))]

filtered_df = filtered_df.reset_index(drop=True)

rename_map = {
    'Height': 'Height (Inches)',
    'Temperature Fahrenheit': 'temperature',
    'Heart Rate': 'heartrate',
    'Respiratory Rate': 'resprate',
    'O2 saturation pulseoxymetry': 'o2sat'
}
filtered_df['label'] = filtered_df['label'].replace(rename_map)

filtered_df = filtered_df[filtered_df['label'] != 'Admission Weight (lbs.)']

mask = filtered_df['label'] == 'Temperature Celsius'
filtered_df.loc[mask, 'valuenum'] = (filtered_df.loc[mask, 'valuenum'] * 9/5) + 32
filtered_df.loc[mask, 'label'] = 'temperature'
filtered_df.loc[mask, 'valueuom'] = '°F'

filtered_df = filtered_df.reset_index(drop=True)

df['ecg_time'] = pd.to_datetime(df['ecg_time'])
omr['chartdate'] = pd.to_datetime(omr['chartdate'])



new_rows = []

for label in ['Weight (lbs)', 'Height (Inches)']:
    filtered_label = filtered_df[filtered_df['label'] == label]
    for _, row in tqdm(filtered_label.iterrows()):
        new_rows.append({
            'subject_id': row['subject_id'],
            'chartdate': row['storetime'],   # match omr time column
            'seq_num': 0,
            'result_value': row['valuenum'], # match omr value column
            'result_name': row['label']      # match omr name column
        })


new_rows_df = pd.DataFrame(new_rows)

omr_updated = pd.concat([omr, new_rows_df], ignore_index=True)

omr_updated['result_name'] = omr_updated['result_name'].replace({'Weight (lbs)': 'Weight (Lbs)'})
omr_updated['result_value'] = pd.to_numeric(omr_updated['result_value'], errors='coerce')

q99 = omr_updated.groupby("result_name")["result_value"].transform(lambda x: x.quantile(0.99))
omr_updated = omr_updated[omr_updated["result_value"] <= q99]

q01 = omr_updated.groupby("result_name")["result_value"].transform(lambda x: x.quantile(0.01))
omr_updated = omr_updated[omr_updated["result_value"] >= q01]

vital_long = vital.melt(
    id_vars=['subject_id', 'stay_id', 'charttime'],   # columns to keep
    value_vars=['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp'],  # columns to unpivot
    var_name='result_name',   # new column with names
    value_name='result_value' # new column with values
)

vital_long = vital_long.sort_values(['subject_id', 'charttime', 'result_name']).reset_index(drop=True)

new_rows = []

for label in ['temperature', 'heartrate', 'resprate', 'o2sat']:
    filtered_label = filtered_df[filtered_df['label'] == label]
    for _, row in tqdm(filtered_label.iterrows()):
        new_rows.append({
            'subject_id': row['subject_id'],
            'stay_id': 0,
            'charttime': row['storetime'],   # match omr time column
            'result_name': row['label'],     # match omr name column
            'result_value': row['valuenum'], # match omr value column
        })
        
new_rows_df = pd.DataFrame(new_rows)

vital_updated = pd.concat([vital_long, new_rows_df], ignore_index=True)
vital_updated = vital_updated.dropna(subset=['result_value']).reset_index(drop=True)
vital_updated['result_value'] = pd.to_numeric(vital_updated['result_value'], errors='coerce')

q01 = vital_updated.groupby("result_name")["result_value"].transform(lambda x: x.quantile(0.01))
q99 = vital_updated.groupby("result_name")["result_value"].transform(lambda x: x.quantile(0.99))

vital_updated = vital_updated[(vital_updated["result_value"] >= q01) & 
                               (vital_updated["result_value"] <= q99)]

dflabevents = dflabevents[['subject_id','storetime','valuenum','label','valueuom']]
dflabevents = dflabevents.dropna(subset=['valuenum']).reset_index(drop=True)

new_rows = []

for label in ['Creatinine (serum)', 'Hemoglobin','Hematocrit (serum)', 'Total Bilirubin', 'Albumin']:
    filtered_label = filtered_df[filtered_df['label'] == label]
    for _, row in tqdm(filtered_label.iterrows()):
        new_rows.append({
            'subject_id': row['subject_id'],
            'storetime': row['storetime'],   # match omr time column
            'valuenum': row['valuenum'],
            'label': row['label']})
        
new_rows_df = pd.DataFrame(new_rows)

dflabevents_updated = pd.concat([dflabevents, new_rows_df], ignore_index=True)

dflabevents_updated['label'] = dflabevents_updated['label'].replace({
    'Creatinine (serum)': 'Creatinine',
    'Hematocrit (serum)': 'Hematocrit',
    'Total Bilirubin': 'Bilirubin'
})

dflabevents_updated['label'] = dflabevents_updated['label'].replace({'Bilirubin': 'Bilirubin, Total'})

q01 = dflabevents_updated.groupby("label")["valuenum"].transform(lambda x: x.quantile(0.01))
q99 = dflabevents_updated.groupby("label")["valuenum"].transform(lambda x: x.quantile(0.99))

dflabevents_updated = dflabevents_updated[(dflabevents_updated["valuenum"] >= q01) & 
                               (dflabevents_updated["valuenum"] <= q99)]
dflabevents_updated['valuenum'] = pd.to_numeric(dflabevents_updated['valuenum'], errors='coerce')
df['ecg_time'] = pd.to_datetime(df['ecg_time'])
omr_updated['chartdate'] = pd.to_datetime(omr_updated['chartdate'])
vital_updated['charttime'] = pd.to_datetime(vital_updated['charttime'])
dflabevents_updated['storetime'] = pd.to_datetime(dflabevents_updated['storetime'])

omr_subset = omr_updated[omr_updated['result_name'].isin(['Height (Inches)','Weight (Lbs)','BMI (kg/m2)'])]
omr_merged = df[['subject_id','ecg_time']].merge(omr_subset, on='subject_id', how='left')
omr_merged['time_diff'] = (omr_merged['chartdate'] - omr_merged['ecg_time']).abs().dt.days
omr_merged = omr_merged[omr_merged['time_diff'] <= 30]
omr_closest_idx = omr_merged.groupby(['subject_id','ecg_time','result_name'])['time_diff'].idxmin()
omr_closest = omr_merged.loc[omr_closest_idx]
omr_wide = omr_closest.pivot_table(index=['subject_id','ecg_time'],
                                   columns='result_name',
                                   values='result_value').reset_index()

vitals_subset = vital_updated[vital_updated['result_name'].isin(['dbp','heartrate','o2sat','resprate','sbp','temperature'])]
vitals_merged = df[['subject_id','ecg_time']].merge(vitals_subset, on='subject_id', how='left')
vitals_merged['time_diff'] = (vitals_merged['charttime'] - vitals_merged['ecg_time']).abs().dt.total_seconds() / 3600
vitals_merged = vitals_merged[vitals_merged['time_diff'] <= 1]
vitals_closest_idx = vitals_merged.groupby(['subject_id','ecg_time','result_name'])['time_diff'].idxmin()
vitals_closest = vitals_merged.loc[vitals_closest_idx]
vitals_wide = vitals_closest.pivot_table(index=['subject_id','ecg_time'],
                                         columns='result_name',
                                         values='result_value').reset_index()

lab_labels = ['PT', 'Albumin', 'Anion Gap', 'Bicarbonate', 'Bilirubin, Total','Calcium, Total', 'Creatinine', 'Ferritin', 'Urea Nitrogen','Hematocrit', 'Hemoglobin', 'Lymphocytes', 'MCHC', 'RDW','Red Blood Cells', 'RDW-SD', 'Creatine Kinase (CK)', 'NTproBNP']
labs_subset = dflabevents_updated[dflabevents_updated['label'].isin(lab_labels)]
labs_merged = df[['subject_id','ecg_time']].merge(labs_subset, on='subject_id', how='left')
labs_merged['time_diff'] = (labs_merged['storetime'] - labs_merged['ecg_time']).abs().dt.total_seconds()/3600
labs_merged = labs_merged[labs_merged['time_diff'] <= 1]
labs_closest_idx = labs_merged.groupby(['subject_id','ecg_time','label'])['time_diff'].idxmin()
labs_closest = labs_merged.loc[labs_closest_idx]
labs_wide = labs_closest.pivot_table(index=['subject_id','ecg_time'],
                                    columns='label',
                                    values='valuenum').reset_index()

labels_metadata_df = df.merge(omr_wide, on=['subject_id','ecg_time'], how='left') \
                       .merge(vitals_wide, on=['subject_id','ecg_time'], how='left') \
                       .merge(labs_wide, on=['subject_id','ecg_time'], how='left')

metadata_cols = ['age',
                   'Height (Inches)','Weight (Lbs)','BMI (kg/m2)',
                    'RR','QRS','QT','QTc','P_wave_axis','QRS_axis','T_wave_axis',
                    'PT', 'Albumin', 'Anion Gap', 'Bicarbonate', 'Bilirubin, Total','Calcium, Total', 'Creatinine', 'Ferritin', 'Urea Nitrogen','Hematocrit', 'Hemoglobin', 'Lymphocytes', 'MCHC', 'RDW','Red Blood Cells', 'RDW-SD', 'Creatine Kinase (CK)', 'NTproBNP',
                   'dbp','heartrate','o2sat','resprate','sbp','temperature']

np.save('data/lbl_itos_metadata.npy', metadata_cols)
labels_metadata_df = labels_metadata_df[labels_metadata_df['is_diagnostic']==1]
labels_metadata_df['label_sex'] = labels_metadata_df['gender'].map({'F': 0, 'M': 1}).fillna(np.nan).apply(lambda x: [x])

train_df = labels_metadata_df[labels_metadata_df['strat_fold'].isin(range(0,18))].copy()
val_df   = labels_metadata_df[labels_metadata_df['strat_fold']==18].copy()
test_df  = labels_metadata_df[labels_metadata_df['strat_fold']==19].copy()

scalers = {}

# Fit one scaler per column using train only
for col in metadata_cols:
    scaler = StandardScaler()
    # Use only non-NaN values
    non_nan_values = train_df[[col]].dropna()
    scaler.fit(non_nan_values.values)
    scalers[col] = scaler

# Function to transform a column while preserving NaNs
def scale_column(col_values, scaler):
    col_array = col_values.values.reshape(-1,1)
    nan_mask = np.isnan(col_array)
    col_array_filled = np.where(nan_mask, 0, col_array)  # temporary fill
    col_scaled = scaler.transform(col_array_filled)
    col_scaled[nan_mask] = np.nan  # restore NaNs
    return col_scaled.flatten()

# Apply scaler per column for each fold
for df in [train_df, val_df, test_df]:
    for col in metadata_cols:
        df[col] = scale_column(df[col], scalers[col])

# Merge back
labels_metadata_df_std = pd.concat([train_df, val_df, test_df], ignore_index=True)

joblib_file = 'data/scalers_dict.pkl'
joblib.dump(scalers, joblib_file)

loaded_scalers = joblib.load(joblib_file)
labels_metadata_df_std['label_metadata'] = labels_metadata_df_std[metadata_cols].values.tolist()

lbl_itos_diags = np.load('data/lbl_itos_diagnostic.npy')
lbl_itos_det = np.load('data/lbl_itos_deterioration.npy')
lbl_itos_sex = np.array(['sex'])
lbl_itos_meta = np.load('data/lbl_itos_metadata.npy')

lbl_itos_mimic = np.concatenate([lbl_itos_diags, lbl_itos_det, lbl_itos_sex, lbl_itos_meta])

expected_lengths = {}
for col in ['label_diagnostic', 'label_deterioration', 'label_sex', 'label_metadata']:
    first_valid = labels_metadata_df_std[col].dropna().iloc[0]
    if isinstance(first_valid, list):
        expected_lengths[col] = len(first_valid)
    elif hasattr(first_valid, "tolist"):
        expected_lengths[col] = len(first_valid.tolist())
    else:
        print('fail')  

def ensure_list(x, expected_len):
    if isinstance(x, list):
        return x
    elif hasattr(x, "tolist"):  
        return x.tolist()
    else:
        return [np.nan] * expected_len

for col in ['label_diagnostic', 'label_deterioration', 'label_sex', 'label_metadata']:
    labels_metadata_df_std[col] = labels_metadata_df_std[col].apply(lambda x: ensure_list(x, expected_lengths[col]))
    
    
import numpy as np

d18 = labels_metadata_df_std[labels_metadata_df_std['strat_fold'] == 18]
d19 = labels_metadata_df_std[labels_metadata_df_std['strat_fold'] == 19]

binary_cols = ['label_diagnostic', 'label_deterioration', 'label_sex']
reg_cols = ['label_metadata']

def check_binary_column(df, col):
    n_positions = len(df[col].iloc[0])
    failed_indices = []
    for i in range(n_positions):
        values = np.array([row[i] for row in df[col]])
        n_pos = np.sum(values == 1)
        n_neg = np.sum(values == 0)
        if n_pos < 2 or n_neg < 2:
            failed_indices.append(i)
    return failed_indices

def check_reg_column(df, col):
    n_positions = len(df[col].iloc[0])
    failed_indices = []
    for i in range(n_positions):
        values = np.array([row[i] for row in df[col]])
        n_not_nan = np.sum(~np.isnan(values))
        if n_not_nan < 3:
            failed_indices.append(i)
    return failed_indices

for fold_name, df_fold in [('fold18', d18), ('fold19', d19)]:
    print(f"--- {fold_name} ---")
    for col in binary_cols:
        failed = check_binary_column(df_fold, col)
        if failed:  # only print if there are failing indices
            print(f"{col} fails at positions: {failed}")
    for col in reg_cols:
        failed = check_reg_column(df_fold, col)
        if failed:  # only print if there are failing indices
            print(f"{col} fails at positions: {failed}")
            
            
labels_metadata_df_std['label_all'] = (
    labels_metadata_df_std['label_diagnostic'] 
    + labels_metadata_df_std['label_deterioration'] 
    + labels_metadata_df_std['label_sex'] 
    + labels_metadata_df_std['label_metadata']
)

labels_metadata_df_std.to_pickle('mimic/df_mimic_benchmark.pkl')
np.save('mimic/lbl_itos_mimic.npy', lbl_itos_mimic)


