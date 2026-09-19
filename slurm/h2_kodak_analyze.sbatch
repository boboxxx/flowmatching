#!/bin/bash
#SBATCH --job-name=flowharq-h2-kodak-analysis
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-h2-kodak-analysis-%j.out
#SBATCH --error=logs/flowharq-h2-kodak-analysis-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
mkdir -p results/h2/kodak_analysis
"$python" -m flowharq.analyze \
  --csv results/h2/kodak/*.csv \
  --split all \
  --output-dir results/h2/kodak_analysis
