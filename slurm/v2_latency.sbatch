#!/bin/bash
#SBATCH --job-name=flowharq-v2-latency
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-v2-latency-%j.out
#SBATCH --error=logs/flowharq-v2-latency-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
cd "$project"
mkdir -p logs results/v2/latency
"$python" -m flowharq.benchmark_latency \
  --data "$data" \
  --checkpoint results/v2/seed_2027/checkpoint.pt \
  --calibration-json results/v2/calibration/selected.json \
  --output results/v2/latency/rtx_pro_6000.json \
  --warmup 30 \
  --repetitions 200
