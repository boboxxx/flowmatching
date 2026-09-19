#!/bin/bash
#SBATCH --job-name=flowharq-v2-select
#SBATCH --partition=enginf
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/flowharq-v2-select-%j.out
#SBATCH --error=logs/flowharq-v2-select-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
cd "$project"
"$python" -m flowharq.calibrate \
  --csv results/v2/calibration/*.csv \
  --output results/v2/calibration/selected.json \
  --target-psnrs 22,24,26 \
  --psnr-tolerance 0.02 \
  --lpips-tolerance 0.003
