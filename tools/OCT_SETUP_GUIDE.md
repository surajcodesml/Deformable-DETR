# OCT Dataset Setup and Training Guide

This guide walks you through setting up and running training on the OCT `.pkl` dataset.

## Prerequisites

1. **OCT pickle files**: Your `.pkl` files should be organized in a directory structure.
   - Each `.pkl` file contains a dict with keys like `image`/`img`, `boxes`/`bboxes`, `labels`.
   - Boxes are normalized `[cx, cy, w, h]` in `[0, 1]`.
   - Labels in `.pkl`: **0 = fovea, 1 = SCR, 2 = no-object (ignore)**. The loader maps 0→1, 1→2 and drops 2, so the model sees **num_classes=2** (fovea=1, SCR=2) with “no-object” implicit (sigmoid, no extra class).

2. **Environment**: Ensure you have:
   - Python 3.7+
   - PyTorch with CUDA
   - All dependencies from `requirements.txt` installed
   - CUDA operators compiled (`cd models/ops && sh make.sh`)

## Step 1: Create Train/Val/Test Splits

### Option A: Automatic Split Generation (Recommended)

Use the helper script to automatically create splits by volume:

```bash
cd /home/suraj/Git/Deformable-DETR

python tools/create_oct_splits.py \
    --pkl_root /path/to/your/oct_pickle/directory \
    --output_dir /path/to/splits/directory \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

This will create:
- `train_split.txt` - list of volume IDs (or filenames if `--use_filenames` is used)
- `val_split.txt`
- `test_split.txt`

**Example:**
```bash
python tools/create_oct_splits.py \
    --pkl_root /home/suraj/Data/Nemours/pickle \
    --output_dir /home/suraj/Git/Deformable-DETR/splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Option B: Manual Split Creation

Create text files manually. Each line can be either:
- A **volume ID** (e.g., `3_L_3`) - the script will expand it to all 31 B-scans
- A **filename** (e.g., `3_L_3004.pkl`) - specific file only

**Example `train_split.txt`:**
```
3_L_3
4_R_5
5_L_2
```

**Example `val_split.txt`:**
```
6_R_1
7_L_4
```

**Important:** Ensure no volume ID appears in multiple splits (the code will check and raise an error if there's leakage).

## Step 2: Update Training Scripts

### For Single GPU Training

Edit `tools/run_oct_single_gpu.sh`:

```bash
#!/bin/bash
set -e

CODE_DIR=$(dirname "$0")/..
CODE_DIR=$(cd "${CODE_DIR}" && pwd)

# UPDATE THESE PATHS:
OCT_ROOT=/home/suraj/Data/Nemours/pickle  # Your pickle directory
TRAIN_LIST=/home/suraj/Git/Deformable-DETR/splits/train_split.txt
VAL_LIST=/home/suraj/Git/Deformable-DETR/splits/val_split.txt

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
  --output_dir "${OUT_DIR}" \
  --best_metric_key pr_macro_f1  # Use F1 for OCT instead of COCO AP
```

### For Slurm Multi-GPU Training

Edit `tools/slurm_oct_2gpu.sh`:

```bash
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

# UPDATE THESE:
CONDA_ENV=deformable_detr  # Your conda environment name
CODE_DIR=$HOME/Git/Deformable-DETR
OCT_ROOT=/home/suraj/Data/Nemours/pickle
TRAIN_LIST=/home/suraj/Git/Deformable-DETR/splits/train_split.txt
VAL_LIST=/home/suraj/Git/Deformable-DETR/splits/val_split.txt

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
  --batch_size 2 \
  --num_workers 8 \
  --with_box_refine \
  --two_stage \
  --output_dir "${OUT_DIR}" \
  --best_metric_key pr_macro_f1
```

## Step 3: Run Training

### Single GPU (Local or SSH)

```bash
cd /home/suraj/Git/Deformable-DETR
bash tools/run_oct_single_gpu.sh
```

### Multi-GPU via Slurm

```bash
cd /home/suraj/Git/Deformable-DETR
mkdir -p logs  # Create logs directory if it doesn't exist
sbatch tools/slurm_oct_2gpu.sh
```

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f logs/oct_detr_<JOB_ID>.out
```

## Step 4: Monitor Training

Training outputs will be written to `output_dir` (e.g., `outputs/oct_pkl_slurm_<JOB_ID>/`):

- **`log.txt`**: Legacy text log (one JSON line per epoch)
- **`metrics.jsonl`**: Structured JSON-lines log with all metrics
- **`hparams.json`**: Hyperparameters and environment info
- **`checkpoint_best.pth`**: Best model (by `--best_metric_key`)
- **`checkpoint_last.pth`**: Latest checkpoint
- **`checkpoint.pth`**: Periodic checkpoints

### Key Metrics in `metrics.jsonl`

For OCT dataset, you'll see:
- **COCO-style metrics**: `coco_ap`, `coco_ap_50`, `coco_ap_75`, etc.
- **OCT-specific PR/F1**:
  - `pr_fixed_fovea_precision`, `pr_fixed_fovea_recall`, `pr_fixed_fovea_f1`
  - `pr_fixed_SCR_precision`, `pr_fixed_SCR_recall`, `pr_fixed_SCR_f1`
  - `pr_macro_f1`, `pr_micro_f1` (recommended for tracking best model)
  - `pr_best_fovea_f1`, `pr_best_SCR_f1` (best-F1 over score thresholds)
- **Confusion matrix**: `cm_global_TP`, `cm_global_FP`, `cm_global_FN`

### View Metrics

```bash
# View latest metrics
tail -1 outputs/oct_pkl_slurm_*/metrics.jsonl | python -m json.tool

# Extract F1 scores
grep -o '"pr_macro_f1":[0-9.]*' outputs/oct_pkl_slurm_*/metrics.jsonl
```

## Step 5: Visualization and prediction-count check (on server)

### How val images are used

Val images are **not** copied anywhere. The scripts build the same val dataset as training (`build_dataset('val', args)`), so they read from `--oct_pkl_root` and `--oct_val_list`. Use the **same** `--oct_pkl_root`, `--oct_train_list`, and `--oct_val_list` as in your training/Slurm script.

### 1) Visualize 10 val images (GT green, pred > 0.3 red with score)

On the server (after at least one training run has produced a checkpoint):

```bash
cd $HOME/Git/Deformable-DETR   # or your repo path

python tools/visualize_oct_samples.py \
  --dataset_file oct_pkl \
  --oct_pkl_root /home/skumar/Git/CSAT/pickle \
  --oct_train_list $HOME/Git/Deformable-DETR/splits/train_split.txt \
  --oct_val_list $HOME/Git/Deformable-DETR/splits/val_split.txt \
  --output_dir outputs/oct_pkl_slurm_<JOB_ID> \
  --vis_checkpoint outputs/oct_pkl_slurm_<JOB_ID>/checkpoint_best.pth \
  --vis_num_samples 10 \
  --vis_score_thresh 0.3
```

- Replace `<JOB_ID>` with your Slurm job ID (e.g. `outputs/oct_pkl_single_12345`).
- **GT boxes**: green. **Pred boxes**: red, only if score ≥ 0.3; score text shown next to each pred box.
- Outputs: `output_dir/sanity/val_idxXXXXX_imgYYYYYY.png`. Download the `sanity/` folder to view locally if needed.

### 2) Check prediction counts (no-object sanity)

Run on 20 val images; reports mean/min/max number of predicted boxes above 0.05, 0.1, 0.3, 0.5. If counts are 100–300 even at 0.3/0.5, no-object handling may be broken.

```bash
cd $HOME/Git/Deformable-DETR

python tools/check_prediction_counts.py \
  --dataset_file oct_pkl \
  --oct_pkl_root /home/skumar/Git/CSAT/pickle \
  --oct_train_list $HOME/Git/Deformable-DETR/splits/train_split.txt \
  --oct_val_list $HOME/Git/Deformable-DETR/splits/val_split.txt \
  --resume outputs/oct_pkl_slurm_<JOB_ID>/checkpoint_best.pth \
  --num_images 20
```

Healthy model: low mean at 0.3 and 0.5 (e.g. &lt; 20). GT has typically 0–2 objects per OCT image.

## Step 6: Evaluation Only

To evaluate a trained model:

```bash
python main.py \
  --dataset_file oct_pkl \
  --oct_pkl_root /path/to/pickle/directory \
  --oct_train_list /path/to/train_split.txt \
  --oct_val_list /path/to/val_split.txt \
  --resume outputs/oct_pkl_slurm_<JOB_ID>/checkpoint_best.pth \
  --eval \
  --output_dir outputs/eval_run
```

## Why predictions look wrong early in training

If you see **red boxes in similar/wrong places every image**, **only "fovea" (no SCR)**, and **very low scores (e.g. max 0.02)**:

1. **Same spot every image**  
   In non–two-stage Deformable DETR, initial reference points come from **query_embed only** (no image). The decoder then refines them using the image. Early in training that refinement is weak, so box positions are almost the same for every image. This is expected; it improves as the decoder learns.

2. **Thin horizontal/vertical strips**  
   The model outputs (cx, cy, w, h); very small w or h give thin strips. Box regression usually needs many epochs to settle; early on you often get degenerate shapes.

3. **All fovea, no SCR**  
   The top-100 are taken over 300×2 scores. If class 0 (fovea) logits are higher than class 1 (SCR) everywhere, you only see fovea. With more training, SCR scores can rise. If your dataset has many more fovea than SCR, the model may learn fovea first.

**What to do**

- **Train longer** (e.g. 50+ epochs). Reference-point refinement and box regression need time to become content-dependent and accurate.
- **Use fewer queries for OCT**: you have 0–2 objects per image, so 300 slots make “no object” hard to learn. Try `--num_queries 100` when launching training (see run scripts below).
- Optionally **check class balance** (e.g. with a small script over your labels). If SCR is rare, consider a longer run or class-weighted loss in future.

## Troubleshooting

### Volume Leakage Error

If you see:
```
RuntimeError: Volume leakage detected across splits...
```

This means a volume ID appears in multiple splits. Fix your split files to ensure each volume is in exactly one split.

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 1  # Instead of 2
```

Or reduce number of workers:
```bash
--num_workers 2  # Instead of 4 or 8
```

### Missing Dependencies

```bash
pip install -r requirements.txt
cd models/ops && sh make.sh && python test.py
```

## Quick Reference

**Create splits:**
```bash
python tools/create_oct_splits.py --pkl_root <PKL_DIR> --output_dir <SPLITS_DIR>
```

**Single GPU training:**
```bash
bash tools/run_oct_single_gpu.sh
```

**Slurm training:**
```bash
sbatch tools/slurm_oct_2gpu.sh
```

**Visualize (10 val images, GT green, pred > 0.3 red with score):**
```bash
python tools/visualize_oct_samples.py --dataset_file oct_pkl --oct_pkl_root <ROOT> --oct_train_list <TRAIN> --oct_val_list <VAL> --output_dir <OUT> --vis_checkpoint <CHECKPOINT> --vis_num_samples 10 --vis_score_thresh 0.3
```

**Prediction-count check (20 val images, thresholds 0.05/0.1/0.3/0.5):**
```bash
python tools/check_prediction_counts.py --dataset_file oct_pkl --oct_pkl_root <ROOT> --oct_train_list <TRAIN> --oct_val_list <VAL> --resume <CHECKPOINT> --num_images 20
```

**Evaluate:**
```bash
python main.py --resume <CHECKPOINT> --eval ...
```
