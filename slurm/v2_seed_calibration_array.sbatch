#!/bin/bash
#SBATCH --job-name=flowharq-v2-seedcal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=2028-2029
#SBATCH --output=logs/flowharq-v2-seedcal-%A_%a.out
#SBATCH --error=logs/flowharq-v2-seedcal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
training_seed="$SLURM_ARRAY_TASK_ID"
cd "$project"
mkdir -p logs results/v2/calibration
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "results/v2/seed_${training_seed}/checkpoint.pt" \
  --output "results/v2/calibration/seed_${training_seed}.csv" \
  --split calibration \
  --calibration-json results/v2/calibration/selected.json \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 4040
