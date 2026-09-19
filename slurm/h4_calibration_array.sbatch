#!/bin/bash
#SBATCH --job-name=flowharq-h4-cal
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/flowharq-h4-cal-%A_%a.out
#SBATCH --error=logs/flowharq-h4-cal-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
thresholds=(0.80 0.90)
tau="${thresholds[$SLURM_ARRAY_TASK_ID]}"
tag="t${tau/./}"
cd "$project"
mkdir -p logs results/h4/one_shot
export PYTHONPATH="$project/vendor:${PYTHONPATH:-}"
"$python" -m flowharq.evaluate \
  --data "$data" \
  --checkpoint results/h4/one_shot/checkpoint.pt \
  --output "results/h4/one_shot/calibration_k1_${tag}.csv" \
  --split calibration \
  --flow-steps 1 \
  --mask-threshold "$tau" \
  --target-psnr 24 \
  --batch-size 4 \
  --workers 6 \
  --lpips \
  --vendor vendor \
  --seed 6060
