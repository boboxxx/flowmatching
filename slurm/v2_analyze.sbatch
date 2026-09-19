#!/bin/bash
#SBATCH --job-name=flowharq-v2-analysis
#SBATCH --partition=enginf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-v2-analysis-%j.out
#SBATCH --error=logs/flowharq-v2-analysis-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
mkdir -p results/v2/analysis
"$python" -m flowharq.analyze \
  --csv results/v2/test/*.csv \
  --output-dir results/v2/analysis
