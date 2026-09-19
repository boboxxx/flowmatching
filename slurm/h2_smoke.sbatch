#!/bin/bash
#SBATCH --job-name=flowharq-h2-smoke
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/flowharq-h2-smoke-%j.out
#SBATCH --error=logs/flowharq-h2-smoke-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
cd "$project"
mkdir -p logs results/h2/smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.train \
  --data "$data" \
  --output results/h2/smoke \
  --stage decision \
  --initialize results/h1/huber_rec1/checkpoint.pt \
  --batch-size 2 \
  --workers 2 \
  --epochs 1 \
  --max-steps 2 \
  --flow-steps 4 \
  --fm-loss normalized_huber \
  --lambda-quality 1.0 \
  --lambda-decision 1.0 \
  --decision-target-psnr 24 \
  --decision-temperature-db 1.0 \
  --lr 1e-4 \
  --seed 2030
