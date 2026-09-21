#!/bin/bash
#SBATCH --job-name=flowharq-spring-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/spring-cpu-%A_%a.out
#SBATCH --error=logs/spring-cpu-%A_%a.err
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/mnt/nfs2/engdes/wc296/flowmatching}"
python_bin="${FLOWHARQ_PYTHON:-/mnt/nfs2/engdes/wc296/mambajscc_vtc2027/env/bin/python3}"
cd "$project"
export PYTHONPATH="${project}/vendor:${project}/environment/spring_audit_deps:${PYTHONPATH:-}"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  data="${FLOWHARQ_DIV2K_VALID:-/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/DIV2K_valid_HR}"
  dataset=div2k
  seed=8001
else
  data="${FLOWHARQ_KODAK:-/mnt/lustre/scratch/engdes/wc296/mambajscc_vtc2027/data/Kodak-Lossless-True-Color-Image-Suite/PhotoCD_PCD0992}"
  dataset=kodak
  seed=9001
fi
"$python_bin" scripts/audit_spring_link.py --checkpoint results/h2/seed_2030/checkpoint.pt \
  --calibration results/h2/seed_2030/decision.json --data "$data" --dataset "$dataset" --seed "$seed" \
  --device cpu --threads 8 --output "${FLOWHARQ_AUDIT_OUTPUT:-${project}/results/spring_final_cpu_20260921}/$dataset"
