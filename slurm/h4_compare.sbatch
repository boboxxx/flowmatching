#!/bin/bash
#SBATCH --job-name=flowharq-h4-compare
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/flowharq-h4-compare-%j.out
#SBATCH --error=logs/flowharq-h4-compare-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
"$python" -m flowharq.compare_repair \
  --csv results/h4/one_shot_t0/calibration_k1_t*.csv \
  --output results/h4/comparison.json \
  --min-psnr-gain 0.0
