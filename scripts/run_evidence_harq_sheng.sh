#!/usr/bin/env bash
set -euo pipefail
project="${EVIDENCE_PROJECT:-/home/sheng/flowmatching_vtc2027}"
python_bin="${EVIDENCE_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}"
data_root="${EVIDENCE_DIV2K:-/home/sheng/ugp_deepjscc_v2_20260919/data}"
output="${EVIDENCE_OUTPUT:-${project}/results/evidence_harq_20260920/main}"
cd "$project"
export PYTHONHASHSEED=2027
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# flock prevents duplicate launches; resume is deliberately an explicit separate action.
exec 9>"${project}/results/evidence_harq_20260920.pipeline.lock"
flock -n 9 || { echo 'An evidence-HARQ pipeline is already running.'; exit 1; }
"$python_bin" -m evidence_harq.train_codec \
  --upstream "${project}/upstream/SwinJSCC" \
  --base-checkpoint "${project}/artifacts/published_weights/swin_mse_c32.pth" \
  --train-data "$data_root/DIV2K_train_HR" \
  --valid-data "$data_root/DIV2K_valid_HR" --test-data "$project/data/Kodak24" \
  --output "$output" --steps 2000 --batch-size 8 --workers 4
"$python_bin" -m evidence_harq.evaluate_codec --run "$output" --split oracle_gate \
  --output "$output/oracle_gate.csv" --draws 4 --snrs 10
"$python_bin" -m evidence_harq.oracle_gate --csv "$output/oracle_gate.csv" \
  --output "$output/gate_decision.json"
# Regardless of the decision, retain the complete first candidate, including negative results.
# Posterior/policy work is activated only after reading the gate and independent review.
echo 'CODEC_AND_ORACLE_GATE_FINISHED'
