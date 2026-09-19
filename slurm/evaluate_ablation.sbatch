#!/bin/bash
#SBATCH --job-name=flowharq-ablate
#SBATCH --partition=short
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/flowharq-ablate-%j.out
#SBATCH --error=logs/flowharq-ablate-%j.err

set -euo pipefail
project=/mnt/nfs2/engdes/wc296/flowmatching
python=/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3
data=/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR
cd "$project"
mkdir -p logs results/ablations
nvidia-smi --query-gpu=name,memory.total --format=csv

for spec in \
  "k1_t050 1 0.50" \
  "k2_t050 2 0.50" \
  "k8_t050 8 0.50" \
  "k2_t065 2 0.65" \
  "k4_t065 4 0.65" \
  "k4_t080 4 0.80"
do
  read -r name steps threshold <<< "$spec"
  "$python" -m flowharq.evaluate \
    --upstream upstream/SwinJSCC \
    --data "$data" \
    --checkpoint results/flow/checkpoint.pt \
    --output "results/ablations/${name}.csv" \
    --snrs 0,3,6,9,12,15 \
    --speeds 30,60,90,120 \
    --target-psnr 24 \
    --flow-steps "$steps" \
    --mask-threshold "$threshold"
done

