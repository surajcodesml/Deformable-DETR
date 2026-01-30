#!/bin/bash
#SBATCH --job-name=oct_detr
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/oct_detr_%j.out
#SBATCH --error=logs/oct_detr_%j.err

set -e

# Adjust to your environment
CONDA_ENV=deformable_detr
CODE_DIR=$HOME/Git/Deformable-DETR
OCT_ROOT=/home/skumar/Git/CSAT/pickle
TRAIN_LIST=/path/to/oct_train_list.txt
VAL_LIST=/path/to/oct_val_list.txt

module load cuda/11.8 || true

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${CODE_DIR}"

OUT_DIR="outputs/oct_pkl_slurm_${SLURM_JOB_ID}"

python -m torch.distributed.run --nproc_per_node=2 main.py \
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