#!/bin/bash
#SBATCH --job-name=oct_detr_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/oct_detr_1gpu_%j.out
#SBATCH --error=logs/oct_detr_1gpu_%j.err
#SBATCH --mail-user=sk1019@nemours.org
#SBATCH --mail-type=END,FAIL

set -e

CONDA_ENV=deformable
CODE_DIR=$HOME/Git/Deformable-DETR
OCT_ROOT=/home/skumar/Git/CSAT/pickle
TRAIN_LIST=/path/to/oct_train_list.txt
VAL_LIST=/path/to/oct_val_list.txt

module load cuda/11.8 || true

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${CODE_DIR}"

OUT_DIR="outputs/oct_pkl_single_${SLURM_JOB_ID}"

python main.py \
  --dataset_file oct_pkl \
  --oct_pkl_root "${OCT_ROOT}" \
  --oct_train_list "${TRAIN_LIST}" \
  --oct_val_list "${VAL_LIST}" \
  --num_queries 100 \
  --batch_size 2 \
  --num_workers 8 \
  --with_box_refine \
  --two_stage \
  --output_dir "${OUT_DIR}"
