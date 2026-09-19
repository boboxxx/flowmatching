#!/bin/bash
#SBATCH --job-name=flowharq-v2-cal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/flowharq-v2-cal-%A_%a.out
#SBATCH --error=logs/flowharq-v2-cal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
checkpoint="$project/results/v2/seed_2027/checkpoint.pt"
steps=(1 2 4 8)
thresholds=(0.00 0.50 0.65 0.80 0.90)
step_index=$((SLURM_ARRAY_TASK_ID / 5))
threshold_index=$((SLURM_ARRAY_TASK_ID % 5))
k="${steps[$step_index]}"
tau="${thresholds[$threshold_index]}"
tag="k${k}_t${tau/./}"
cd "$project"
mkdir -p logs results/v2/calibration
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "$checkpoint" \
  --output "results/v2/calibration/${tag}.csv" \
  --split calibration \
  --flow-steps "$k" \
  --mask-threshold "$tau" \
  --target-psnr 24 \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 4040
