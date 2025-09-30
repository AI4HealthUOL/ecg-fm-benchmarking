# ECG-FM-Benchmarking

The official repository for the paper [Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks](http://arxiv.org/abs/2509.25095)

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](http://arxiv.org/abs/2509.25095)


Here, we benchmark ECG foundation models across **12 public datasets** and **26 clinically relevant tasks** encompassing **1,650 regression and classification targets**.
We also proposed ECG-CPC, a new and outperforming ECG foundational model. We provide scripts, configurations, and checkpoints to evaluate models efficiently and reproducibly.

![Abstract](abstract.png)

---

## 📂 Datasets

You can download the datasets from the following sources:

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)  
- [PTB](https://www.physionet.org/content/ptbdb/1.0.0/)  
- [SPH](https://figshare.com/articles/figure/SPH/22199548?file=39453271)  
- [EchoNext](https://physionet.org/content/echonext/1.1.0/)  
- [ZZU pECG](https://doi.org/10.6084/m9.figshare.27078763)  
- [CODE-15%](https://zenodo.org/records/4916206)  
- [Chapman](https://figshare.com/articles/dataset/Chapman_ECG_dataset/25558926)  
- [CPSC2018, CPSC-Extra, Georgia, Ningbo](https://physionet.org/content/challenge-2021/1.0.3/)  
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)  

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
