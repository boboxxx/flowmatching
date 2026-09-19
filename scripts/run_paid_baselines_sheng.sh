#!/usr/bin/env bash
set -euo pipefail
project="${EVIDENCE_PROJECT:-/home/sheng/flowmatching_vtc2027}"
torch_python="${EVIDENCE_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}"
ntscc_python="${NTSCC_PYTHON:-${project}/.venvs/ntscc/bin/python}"
output="${EVIDENCE_BASELINE_OUTPUT:-${project}/results/evidence_harq_20260920/paid_baselines}"
cd "$project"
for name in swin_mse_c32 swin_mse_c64; do
  "$torch_python" scripts/evaluate_paid_jscc.py --family swin --upstream upstream/SwinJSCC \
    --checkpoint "artifacts/published_weights/${name}.pth" --data data/Kodak24 \
    --output "$output/$name.csv" --draws 10
done
for name in ntscc_quality1 ntscc_quality2; do
  "$ntscc_python" scripts/evaluate_paid_jscc.py --family ntscc --upstream upstream/ntscc_author \
    --checkpoint "artifacts/published_weights/${name}.pth" --data data/Kodak24 \
    --output "$output/$name.csv" --draws 10
done
