#!/bin/bash
#SBATCH --job-name=flowharq-h1-analyze
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-h1-analyze-%j.out
#SBATCH --error=logs/flowharq-h1-analyze-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
mkdir -p results/h1/analysis
"$python" -m flowharq.analyze \
  --csv results/h1/test/*.csv \
  --output-dir results/h1/analysis
