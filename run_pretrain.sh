#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --nodelist=
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null

BASE_DIR=""

module load hpc-env/13.1 CUDA/12.4.0 Anaconda3 git
conda activate lightning3

LOGS_DIR="${BASE_DIR}/pretrain_logs"
mkdir -p "${LOGS_DIR}"

LOG_FILE="${LOGS_DIR}/${EVAL_MODE}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

python ${BASE_DIR}/code/pretrain.py --config-name=config_cpc_ecg_s4_heedb.yaml