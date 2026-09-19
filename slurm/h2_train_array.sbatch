#!/bin/bash
#SBATCH --job-name=flowharq-h2-train
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=2030-2032
#SBATCH --output=logs/flowharq-h2-train-%A_%a.out
#SBATCH --error=logs/flowharq-h2-train-%A_%a.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_train_HR
seed="$SLURM_ARRAY_TASK_ID"
if [[ "$seed" == "2030" ]]; then
  initialize="$project/results/h1/huber_rec1/checkpoint.pt"
else
  initialize="$project/results/h1/seed_${seed}/checkpoint.pt"
fi
cd "$project"
mkdir -p logs "results/h2/seed_${seed}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python" -m flowharq.train \
  --upstream upstream/SwinJSCC \
  --data "$data" \
  --output "results/h2/seed_${seed}" \
  --stage decision \
  --initialize "$initialize" \
  --batch-size 2 \
  --workers 8 \
  --epochs 10 \
  --flow-steps 4 \
  --mask-fraction 0.35 \
  --fm-loss normalized_huber \
  --lambda-quality 1.0 \
  --lambda-decision 1.0 \
  --decision-target-psnr 24 \
  --decision-temperature-db 1.0 \
  --lr 1e-4 \
  --seed "$seed"
