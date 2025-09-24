#!/bin/bash
#SBATCH --job-name=cpc
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
CHECKPOINTS_DIR=""
DATASET_DIR=""

EVAL_MODE="frozen"    # finetuning_linear, frozen, linear
MODEL="cpc"           # ecg_founder, ecg_jepa_multiblock, st_mem, merl_resnet, ecgfm_ked, s4, net1d, cpc, hubert_ecg_base
DATASET="ptbxl_all"   # mimic, ptb, ptbxl_all, ptbxl_sub, ptbxl_super, chapman, ningbo, sph, cpsc2018, cpsc_extra, echonext, georgia, code15_diag, code_test, zzu_pecg
LEARNING_RATE=0.001
BATCH_SIZE=64


if [ "$EVAL_MODE" == "finetuning_linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/new_finetuning_linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/new_finetuning_linear/predictions"
    LOGS_DIR="${BASE_DIR}/new_finetuning_linear/logs"
elif [ "$EVAL_MODE" == "frozen" ]; then
    OUTPUT_DIR="${BASE_DIR}/frozen/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/frozen/predictions"
    LOGS_DIR="${BASE_DIR}/frozen/logs"
elif [ "$EVAL_MODE" == "linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/linear/predictions"
    LOGS_DIR="${BASE_DIR}/linear/logs"
else
    echo "Error: Unknown mode '$EVAL_MODE'. Choose from finetuning_linear, finetuning_nonlinear frozen, or linear."
    exit 1
fi

module load hpc-env/13.1 CUDA/12.4.0 Anaconda3 git
conda activate lightning3

mkdir -p "${LOGS_DIR}/${MODEL}"
mkdir -p "${OUTPUT_DIR}/${MODEL}_${DATASET}"
mkdir -p "${PREDICTIONS_DIR}/${MODEL}"

LOG_FILE="${LOGS_DIR}/${MODEL}/${DATASET}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Special handling per dataset
ARGS_DATASET=()
case $DATASET in
  "ptb")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb"
        "--fs-data 1000"
        "--finetune-dataset ptb"
    )
    ;;
  "ningbo")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ningbo"
        "--fs-data 500"
        "--finetune-dataset ningbo"
    )
    ;;
  "cpsc2018")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc2018"
        "--fs-data 500"
        "--finetune-dataset cpsc2018"
    )
    ;;
  "cpsc_extra")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc_extra"
        "--fs-data 500"
        "--finetune-dataset cpsc_extra"
    )
    ;;
  "georgia")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/georgia"
        "--fs-data 500"
        "--finetune-dataset georgia"
    )
    ;;
  "chapman")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/chapman/ECGData"
        "--fs-data 500"
        "--finetune-dataset chapman"
    )
    ;;
  "sph")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/sph"
        "--fs-data 500"
        "--finetune-dataset sph"
    )
    ;;
  "code15_diag")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/code15"
        "--fs-data 400"
        "--finetune-dataset code15_diag"
    )
    ;;
  "ptbxl_super")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_super"
    )
    ;;
  "ptbxl_sub")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_sub"
    )
    ;;
  "ptbxl_all")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptb-xl/records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_all"
    )
    ;;
  "echonext")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/echonext"
        "--fs-data 250"
        "--finetune-dataset echonext"
    )
    ;;
  "mimic")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/mimic"
        "--fs-data 500"
        "--finetune-dataset mimic"
    )
    ;;
  "zzu_pecg")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/zzu_pecg"
        "--fs-data 500"
        "--finetune-dataset zzu_pecg"
    )
    ;;      
esac

# Special handling per model
ARGS_MODEL=()
case $MODEL in
  "ecg_founder")
    ARGS_MODEL+=(
        "--architecture ecg_founder"
        "--input-size 2.5"
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_founder/12_lead_ECGFounder.pth"
    )
    ;;
  "ecg_jepa_multiblock")
    ARGS_MODEL+=(
        "--architecture ecg_jepa"
        "--input-size 10" 
        "--fs-model 250"
        "--input-channels 8"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/multiblock_epoch100.pth"
    )
    ;;
  "ecg_jepa_random")
    ARGS_MODEL+=(
        "--architecture ecg_jepa"
        "--input-size 10" 
        "--fs-model 250"
        "--input-channels 8"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/random_epoch100.pth"
    )
    ;;
  "st_mem")
    ARGS_MODEL+=(
        "--architecture st_mem"
        "--input-size 2.4"
        "--fs-model 250"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/st_mem/st_mem_vit_base_full.pth"
    )
    ;;
  "merl_resnet")
    ARGS_MODEL+=(
        "--architecture merl"
        "--merl-backbone resnet"
        "--input-size 2.5" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/merl/res18_best_encoder.pth"
    )
    ;;
  "ecgfm_ked")
    ARGS_MODEL+=(
        "--architecture ecgfm_ked"
        "--input-size 10" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    )
    ;;
  "s4")
    ARGS_MODEL+=(
        "--architecture s4"
        "--input-size 2.5"
        "--fs-model 100"
        "--input-channels 12"
        "--s4-n 8"
        "--s4-h 512"
        "--s4-layers 4"
        "--precision 32"
    )
    ;;
  "net1d")
    ARGS_MODEL+=(
        "--architecture net1d"
        "--input-size 2.5"
        "--fs-model 500"
        "--input-channels 12"
    )
    ;;
  "cpc")
    ARGS_MODEL+=(
        "--architecture cpc"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/cpc/config_last_11597276_ckpt.yaml"
    )
    ;;
  "hubert_ecg_base")
    ARGS_MODEL+=(
        "--architecture hubert_ecg"
        "--input-size 5" 
        "--fs-model 100"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_base.safetensors"
    )
    ;;
  "ecg_fm")
    ARGS_MODEL+=(
        "--architecture ecg_fm"
        "--input-size 5" 
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
    )
    ;;              
esac

# Run the experiment
python ${BASE_DIR}/code/main_lite.py \
  ${ARGS_DATASET[@]} \
  ${ARGS_MODEL[@]} \
  --epochs 100 \
  --modality ecg \
  --lr ${LEARNING_RATE} \
  --batch-size ${BATCH_SIZE} \
  --finetune \
  --eval-mode ${EVAL_MODE} \
  --output-path "${OUTPUT_DIR}/${MODEL}_${DATASET}" \
  --prediction-path "${PREDICTIONS_DIR}/${MODEL}" \
  --export-predictions \
