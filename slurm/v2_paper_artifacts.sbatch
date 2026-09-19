#!/bin/bash
#SBATCH --job-name=flowharq-v2-paper
#SBATCH --partition=enginf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowharq-v2-paper-%j.out
#SBATCH --error=logs/flowharq-v2-paper-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
"$python" -m flowharq.paper_artifacts \
  --analysis-dir results/v2/analysis \
  --calibration-json results/v2/calibration/selected.json \
  --latency-json results/v2/latency/rtx_pro_6000.json \
  --paper-dir paper
