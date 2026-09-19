#!/bin/bash
#SBATCH --job-name=flowharq-v2-train
#SBATCH --partition=long
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=2027-2029
#SBATCH --output=logs/flowharq-v2-train-%A_%a.out
#SBATCH --error=logs/flowharq-v2-train-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
backbone=/mnt/nfs2/engdes/wc296/ugp_deepjscc_vtc2027/results/stage1/checkpoint.pt
seed="$SLURM_ARRAY_TASK_ID"
cd "$project"
mkdir -p logs "results/v2/seed_${seed}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.train \
  --upstream upstream/SwinJSCC \
  --data "$data" \
  --output "results/v2/seed_${seed}" \
  --stage flow \
  --backbone "$backbone" \
  --batch-size 2 \
  --workers 8 \
  --epochs 40 \
  --flow-steps 4 \
  --mask-fraction 0.35 \
  --seed "$seed"

