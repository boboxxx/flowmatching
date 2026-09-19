#!/bin/bash
#SBATCH --job-name=flowharq-h1-test
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/flowharq-h1-test-%A_%a.out
#SBATCH --error=logs/flowharq-h1-test-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
channel_seeds=(7001 7002 7003)
channel_seed="${channel_seeds[$SLURM_ARRAY_TASK_ID]}"
cd "$project"
mkdir -p logs results/h1/test
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint results/h1/huber_rec1/checkpoint.pt \
  --output "results/h1/test/train_2030_channel_${channel_seed}.csv" \
  --split test \
  --calibration-json results/h1/huber_rec1/decision.json \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed "$channel_seed"
