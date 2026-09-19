#!/bin/bash
#SBATCH --job-name=flowharq-h1-cal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/flowharq-h1-cal-%A_%a.out
#SBATCH --error=logs/flowharq-h1-cal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
variants=(baseline rec1 huber huber_rec1)
steps=(1 2 4)
thresholds=(0.80 0.90)
variant_index=$((SLURM_ARRAY_TASK_ID / 6))
setting_index=$((SLURM_ARRAY_TASK_ID % 6))
step_index=$((setting_index / 2))
threshold_index=$((setting_index % 2))
variant="${variants[$variant_index]}"
k="${steps[$step_index]}"
tau="${thresholds[$threshold_index]}"
tag="k${k}_t${tau/./}"
cd "$project"
mkdir -p logs "results/h1/${variant}"
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "results/h1/${variant}/checkpoint.pt" \
  --output "results/h1/${variant}/calibration_${tag}.csv" \
  --split calibration \
  --flow-steps "$k" \
  --mask-threshold "$tau" \
  --target-psnr 24 \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 6060
