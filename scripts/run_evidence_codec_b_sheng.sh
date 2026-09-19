#!/usr/bin/env bash
set -euo pipefail
project="${EVIDENCE_PROJECT:-/home/sheng/flowmatching_vtc2027}"
python_bin="${EVIDENCE_PYTHON:-/home/sheng/anaconda3/envs/cv/bin/python}"
data_root="${EVIDENCE_DIV2K:-/home/sheng/ugp_deepjscc_v2_20260919/data}"
output="${EVIDENCE_OUTPUT:-${project}/results/evidence_harq_20260920/codec_b}"
cd "$project"
export PYTHONHASHSEED=2027
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>"${project}/results/evidence_harq_20260920.pipeline.lock"
flock -n 9 || { echo 'An evidence-HARQ pipeline is already running.'; exit 1; }
"$python_bin" -m evidence_harq.train_codec --upstream "$project/upstream/SwinJSCC" \
  --base-checkpoint "$project/artifacts/published_weights/swin_mse_c32.pth" \
  --train-data "$data_root/DIV2K_train_HR" --valid-data "$data_root/DIV2K_valid_HR" \
  --test-data "$project/data/Kodak24" --output "$output" --steps 8000 --minimum-steps 3000 \
  --validate-every 200 --patience 10 --batch-size 8 --workers 4 --precision fp32 --lr 1e-3
"$python_bin" -m evidence_harq.evaluate_codec --run "$output" --split oracle_gate \
  --output "$output/oracle_gate.csv" --draws 4 --snrs 10
"$python_bin" -m evidence_harq.oracle_gate --csv "$output/oracle_gate.csv" \
  --output "$output/gate_decision.json"
echo 'CODEC_B_AND_DEVELOPMENT_GATE_FINISHED'
