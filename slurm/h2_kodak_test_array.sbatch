#!/bin/bash
#SBATCH --job-name=flowharq-h2-kodak
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/flowharq-h2-kodak-%A_%a.out
#SBATCH --error=logs/flowharq-h2-kodak-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/Kodak-Lossless-True-Color-Image-Suite/PhotoCD_PCD0992
training_seeds=(2030 2031 2032)
channel_seeds=(9001 9002 9003)
training_index=$((SLURM_ARRAY_TASK_ID / 3))
channel_index=$((SLURM_ARRAY_TASK_ID % 3))
training_seed="${training_seeds[$training_index]}"
channel_seed="${channel_seeds[$channel_index]}"
cd "$project"
mkdir -p logs results/h2/kodak
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "results/h2/seed_${training_seed}/checkpoint.pt" \
  --output "results/h2/kodak/train_${training_seed}_channel_${channel_seed}.csv" \
  --split all \
  --calibration-json "results/h2/seed_${training_seed}/decision.json" \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed "$channel_seed"
