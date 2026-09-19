#!/bin/bash
#SBATCH --job-name=flowharq-h1-cal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/flowharq-h1-cal-%A_%a.out
#SBATCH --error=logs/flowharq-h1-cal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
variants=(baseline rec1 huber huber_rec1)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
cd "$project"
mkdir -p logs "results/h1/${variant}"
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "results/h1/${variant}/checkpoint.pt" \
  --output "results/h1/${variant}/calibration_k2_t090.csv" \
  --split calibration \
  --flow-steps 2 \
  --mask-threshold 0.90 \
  --target-psnr 24 \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 6060
