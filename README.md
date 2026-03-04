# ECG-FM-Benchmarking

The official repository for the paper [Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks](http://arxiv.org/abs/2509.25095)

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](http://arxiv.org/abs/2509.25095)



Here, we benchmark ECG foundation models across **12 public datasets** and **26 clinically relevant tasks** encompassing **1,650 regression and classification targets**.
We also proposed ECG-CPC, a new and outperforming ECG foundational model. We provide scripts, configurations, and checkpoints to evaluate models efficiently and reproducibly.

![Abstract](abstract.svg)

---

## 📂 Datasets

You can download the datasets from the following sources:

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)  
- [PTB](https://www.physionet.org/content/ptbdb/1.0.0/)  
- [SPH](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1)  
- [EchoNext](https://physionet.org/content/echonext/1.1.0/)  
- [ZZU pECG](https://doi.org/10.6084/m9.figshare.27078763)  
- [CODE-15%](https://zenodo.org/records/4916206)  
- [Chapman](https://figshare.com/collections/ChapmanECG/4560497)  
- [CPSC2018, CPSC-Extra, Georgia, Ningbo](https://physionet.org/content/challenge-2021/1.0.3/)  (Please include `Label mappings 2021.xlsx` in the respective dataset folder. The [original file](https://docs.google.com/spreadsheets/d/1Q4m9axOlE1rEb7Fi2t4fPbvpw8JPvikLBO_j-lQcuuE/edit?gid=1645151417#gid=1645151417) is linked on the CinC21 challenge website. )
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)  (save under data/ the following files from physionet:
records_w_diag_icd10.csv (MIMIC-IV-ECG-ICD), mds_ed.csv (MDS-ED), machine_measurements.csv (MIMIC-IV-ECG), omr.csv.gz (MIMIC-IV), vitalsign.csv.gz (MIMIC-IV), d_labitems.csv.gz (MIMIC-IV), labevents.csv.gz (MIMIC-IV), d_items.csv.gz (MIMIC-IV), chartevents.csv.gz (MIMIC-IV) )

---

## 📦 Checkpoints

Download pretrained checkpoints for evaluation:

- [ECG-CPC](https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604?file=58173919)  
- [ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)  
- [ECG-JEPA](https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view)  
- [ST-MEM](https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view)  
- [MERL](https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2)  
- [ECGFM-KED](https://zenodo.org/records/14881564)  
- [HuBERT-ECG](https://huggingface.co/Edoardo-BS/hubert-ecg-base/tree/main)  
- [ECG-FM](https://huggingface.co/wanglab/ecg-fm/tree/main)  

---

## ⚙️ Installation

Set up the Python environment using the provided YAML files:

```bash
# General environment
conda env create -f env.yaml

# For ECG-FM evaluation
conda env create -f ecg_fm_env.yaml
```

---

## 🚀 Quick Start

Follow these steps to set up and run the benchmark:

### 1. Data Preprocessing

First, preprocess all datasets using the provided `preprocess_ecg_dataset.ipynb` Jupyter notebook.

### 2. Configuration Setup

Before running the benchmark, configure the necessary paths:

#### Edit `run.sh` file

Open `run.sh` in your preferred text editor and update the following variables with your local paths:

```bash
# Set these paths according to your system
BASE_DIR="/path/to/your/fm-benchmarking"
CHECKPOINTS_DIR="/path/to/your/checkpoints"
DATASET_DIR="/path/to/your/datasets"
```

#### Update dataset path
Modify the dataset path in the `fm-benchmarking/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml`

#### Run `run.sh` file

```bash
sbatch run.sh
```

# Please cite our publication if you found our research to be helpful.

```bibtex
@misc{almasud2025benchmarkingecgfoundationalmodels,
      title={Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks}, 
      author={M A Al-Masud and Juan Miguel Lopez Alcaraz and Nils Strodthoff},
      year={2025},
      eprint={2509.25095},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2509.25095}, 
}
```







