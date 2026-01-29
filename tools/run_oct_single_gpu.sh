#!/bin/bash
set -e

CODE_DIR=$(dirname "$0")/..
CODE_DIR=$(cd "${CODE_DIR}" && pwd)

OCT_ROOT=/path/to/oct_pickles
TRAIN_LIST=/path/to/oct_train_list.txt
VAL_LIST=/path/to/oct_val_list.txt

OUT_DIR="${CODE_DIR}/outputs/oct_single_$(date +%Y%m%d_%H%M%S)"

cd "${CODE_DIR}"

python main.py \
  --dataset_file oct_pkl \
  --oct_pkl_root "${OCT_ROOT}" \
  --oct_train_list "${TRAIN_LIST}" \
  --oct_val_list "${VAL_LIST}" \
  --batch_size 2 \
  --num_workers 4 \
  --with_box_refine \
  --two_stage \
  --output_dir "${OUT_DIR}"