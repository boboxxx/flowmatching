#!/bin/bash
#SBATCH --job-name=flowharq-v2-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-v2-smoke-%j.out
#SBATCH --error=logs/flowharq-v2-smoke-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
backbone=/mnt/nfs2/engdes/wc296/ugp_deepjscc_vtc2027/results/stage1/checkpoint.pt
cd "$project"
mkdir -p logs results/v2/smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m pytest -q
"$python" -m flowharq.train \
  --upstream upstream/SwinJSCC \
  --data "$data" \
  --output results/v2/smoke \
  --stage flow \
  --backbone "$backbone" \
  --batch-size 1 \
  --workers 2 \
  --limit 2 \
  --max-steps 1 \
  --flow-hidden-dim 64 \
  --flow-depth 1 \
  --flow-steps 2

