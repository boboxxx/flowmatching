#!/bin/bash
#SBATCH --job-name=flowharq-eval
#SBATCH --partition=short
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/flowharq-eval-%j.out
#SBATCH --error=logs/flowharq-eval-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
cd "$project"
mkdir -p logs results/eval
"$python" -m flowharq.evaluate \
  --upstream upstream/SwinJSCC \
  --data "$data" \
  --checkpoint results/flow/checkpoint.pt \
  --output results/eval/flowharq.csv \
  --snrs 0,3,6,9,12,15 \
  --speeds 30,60,90,120 \
  --target-psnr 24 \
  --flow-steps 4 \
  --mask-threshold 0.80
"$python" -m flowharq.plot \
  --csv results/eval/flowharq.csv \
  --output-dir results/eval/figures
