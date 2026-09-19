#!/usr/bin/env bash
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/home/sheng/flowmatching_vtc2027}"
python_bin="${FLOWHARQ_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}"
valid="${FLOWHARQ_VALID_DATA:-/home/sheng/ugp_deepjscc_v2_20260919/data/DIV2K_valid_HR}"
out="${FLOWHARQ_METRIC_OUTPUT:-${project}/results/published_baselines_20260919/metric_audit}"
cd "${project}"
mkdir -p "${out}"
"${python_bin}" -m pytest -q tests/test_metrics.py
for dataset in div2k kodak; do
  data="${valid}"
  split=test
  if [[ "${dataset}" == kodak ]]; then
    data="${project}/data/Kodak24"
    split=all
  fi
  for variant in h2 h3; do
    if [[ "${variant}" == h2 ]]; then
      ckpt="${project}/results/h2/seed_2030/checkpoint.pt"
      decision="${project}/results/h2/seed_2030/decision.json"
    else
      ckpt="${project}/results/h3/main_oracle_seed2027/checkpoint_epoch_010.pt"
      decision="${project}/results/h3/main_oracle_seed2027/decision.json"
    fi
    "${python_bin}" -m flowharq.evaluate --data "${data}" \
      --checkpoint "${ckpt}" --calibration-json "${decision}" \
      --output "${out}/${variant}_${dataset}.csv" --split "${split}" \
      --calibration-count 20 --batch-size 4 --workers 4 --lpips --seed 7070
  done
done
