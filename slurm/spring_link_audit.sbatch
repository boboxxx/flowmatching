#!/bin/bash
#SBATCH --job-name=flowharq-spring-audit
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/spring-audit-%j.out
#SBATCH --error=logs/spring-audit-%j.err
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/mnt/nfs2/engdes/wc296/flowmatching}"
python_bin="${FLOWHARQ_PYTHON:-/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3}"
div2k="${FLOWHARQ_DIV2K_VALID:-/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR}"
kodak="${FLOWHARQ_KODAK:-/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/Kodak-Lossless-True-Color-Image-Suite/PhotoCD_PCD0992}"
output="${FLOWHARQ_AUDIT_OUTPUT:-${project}/results/spring_final_20260921}"
cd "$project"
export PYTHONPATH="${project}/vendor:${project}/environment/spring_audit_deps:${PYTHONPATH:-}"
"$python_bin" scripts/audit_spring_link.py --checkpoint results/h2/seed_2030/checkpoint.pt \
  --calibration results/h2/seed_2030/decision.json --data "$div2k" --dataset div2k --seed 8001 --output "$output/div2k"
"$python_bin" scripts/audit_spring_link.py --checkpoint results/h2/seed_2030/checkpoint.pt \
  --calibration results/h2/seed_2030/decision.json --data "$kodak" --dataset kodak --seed 9001 --output "$output/kodak"
