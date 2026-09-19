#!/bin/bash
#SBATCH --job-name=flowharq-h1-compare
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/flowharq-h1-compare-%j.out
#SBATCH --error=logs/flowharq-h1-compare-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
"$python" -m flowharq.compare_repair \
  --csv results/h1/*/calibration_k*_t*.csv \
  --output results/h1/comparison.json
