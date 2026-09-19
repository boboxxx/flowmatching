#!/bin/bash
#SBATCH --job-name=flowharq-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-smoke-%j.out
#SBATCH --error=logs/flowharq-smoke-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
cd "$project"
mkdir -p logs results/smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m pytest -q tests/test_flowharq_modules.py
"$python" -m flowharq.train \
  --upstream upstream/SwinJSCC \
  --data "$data" \
  --output results/smoke \
  --stage flow \
  --batch-size 1 \
  --workers 2 \
  --limit 2 \
  --max-steps 1 \
  --flow-hidden-dim 64 \
  --flow-depth 1 \
  --flow-steps 2

