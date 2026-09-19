#!/bin/bash
#SBATCH --job-name=flowharq-v2-biases
#SBATCH --partition=enginf
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/flowharq-v2-biases-%j.out
#SBATCH --error=logs/flowharq-v2-biases-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
"$python" -m flowharq.calibrate \
  --csv results/v2/calibration/seed_2028.csv \
  --output results/v2/calibration/seed_2028.json \
  --reference-config results/v2/calibration/selected.json
"$python" -m flowharq.calibrate \
  --csv results/v2/calibration/seed_2029.csv \
  --output results/v2/calibration/seed_2029.json \
  --reference-config results/v2/calibration/selected.json
