#!/bin/bash
#SBATCH --job-name=flowharq-h2-cal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=2030-2032
#SBATCH --output=logs/flowharq-h2-cal-%A_%a.out
#SBATCH --error=logs/flowharq-h2-cal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
seed="$SLURM_ARRAY_TASK_ID"
cd "$project"
mkdir -p logs "results/h2/seed_${seed}"
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint "results/h2/seed_${seed}/checkpoint.pt" \
  --output "results/h2/seed_${seed}/calibration.csv" \
  --split calibration \
  --flow-steps 4 \
  --mask-threshold 0.80 \
  --target-psnr 24 \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 6060
"$python" -m flowharq.calibrate \
  --csv "results/h2/seed_${seed}/calibration.csv" \
  --output "results/h2/seed_${seed}/decision.json" \
  --reference-config results/h1/huber_rec1/decision.json
