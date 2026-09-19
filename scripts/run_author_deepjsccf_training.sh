#!/usr/bin/env bash
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/home/sheng/flowmatching_vtc2027}"
train="${FLOWHARQ_TRAIN_DATA:-/home/sheng/ugp_deepjscc_v2_20260919/data/DIV2K_train_HR}"
valid="${FLOWHARQ_VALID_DATA:-/home/sheng/ugp_deepjscc_v2_20260919/data/DIV2K_valid_HR}"
out="${DEEPJSCCF_OUTPUT:-${project}/results/published_baselines_20260919/deepjsccf_author_v2}"
cd "${project}"
bash scripts/deepjsccf_runtime.sh scripts/check_deepjsccf_serialization.py
bash scripts/deepjsccf_runtime.sh scripts/train_author_deepjsccf.py \
  --upstream upstream/deepjscc_feedback_author --train "${train}" --valid "${valid}" \
  --output "${out}/c2_l2" --channels 2 --layers 2 --epochs 100 --steps-per-epoch 100 \
  --batch-size 8 --snr 10 --seed 2027
# No-feedback same-total-bandwidth control using the SAME published architecture.
bash scripts/deepjsccf_runtime.sh scripts/train_author_deepjsccf.py \
  --upstream upstream/deepjscc_feedback_author --train "${train}" --valid "${valid}" \
  --output "${out}/c4_l1" --channels 4 --layers 1 --epochs 100 --steps-per-epoch 100 \
  --batch-size 8 --snr 10 --seed 2027
# Scoring happens only after both training jobs finish successfully.
for name in c2_l2 c4_l1; do
  bash scripts/deepjsccf_runtime.sh scripts/export_author_deepjsccf.py \
    --upstream upstream/deepjscc_feedback_author --run "${out}/${name}" \
    --data "${project}/data/Kodak24" --output "${out}/${name}/kodak_crop256" --crop 256
  "${FLOWHARQ_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}" scripts/score_author_deepjsccf.py \
    --input "${out}/${name}/kodak_crop256" --output "${out}/${name}/kodak_crop256.csv"
done
