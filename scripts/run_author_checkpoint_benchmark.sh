#!/usr/bin/env bash
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/home/sheng/flowmatching_vtc2027}"
python_bin="${FLOWHARQ_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}"
ntscc_python="${NTSCC_PYTHON:-${project}/.venvs/ntscc/bin/python}"
out="${PUBLISHED_OUTPUT:-${project}/results/published_baselines_20260919/author_benchmark}"
cd "${project}"
mkdir -p "${out}"
for crop in 0 256; do
  for name in swin_mse_c32 swin_mse_c64 swin_msssim_c32 swin_msssim_c64 ntscc_quality1 ntscc_quality2; do
    if [[ -f "${out}/${name}_crop${crop}.summary.json" ]]; then
      continue
    fi
    family=swin
    upstream=upstream/SwinJSCC
    py="${python_bin}"
    if [[ "${name}" == ntscc* ]]; then
      family=ntscc
      upstream=upstream/ntscc_author
      py="${ntscc_python}"
    fi
    "${py}" scripts/evaluate_published_checkpoint.py --family "${family}" \
      --upstream "${upstream}" --checkpoint "artifacts/published_weights/${name}.pth" \
      --data "${project}/data/Kodak24" --output "${out}/${name}_crop${crop}.csv" \
      --crop "${crop}" --snrs 10 --draws 10 --seed 7070
  done
done
