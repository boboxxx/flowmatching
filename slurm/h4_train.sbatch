#!/bin/bash
#SBATCH --job-name=flowharq-h4-denoiser
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/flowharq-h4-train-%j.out
#SBATCH --error=logs/flowharq-h4-train-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
backbone=/mnt/nfs2/engdes/wc296/ugp_deepjscc_vtc2027/results/stage1/checkpoint.pt
cd "$project"
mkdir -p logs results/h4/one_shot_t0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.train \
  --data "$data" \
  --output results/h4/one_shot_t0 \
  --stage flow \
  --backbone "$backbone" \
  --batch-size 2 \
  --workers 8 \
  --epochs 40 \
  --flow-steps 1 \
  --mask-fraction 0.35 \
  --fm-loss normalized_huber \
  --fm-time zero \
  --lambda-reconstruction 1.0 \
  --seed 2030
