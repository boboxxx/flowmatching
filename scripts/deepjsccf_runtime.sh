#!/usr/bin/env bash
set -euo pipefail
project="${FLOWHARQ_PROJECT:-/home/sheng/flowmatching_vtc2027}"
venv="${DEEPJSCCF_VENV:-${project}/.venvs/deepjsccf_tf214}"
cuda_site="${CUDA11_SITE:-/home/sheng/anaconda3/envs/cv/lib/python3.10/site-packages/nvidia}"
export LD_LIBRARY_PATH="${venv}/lib/python3.10/site-packages/nvidia/cudnn/lib:${venv}/lib/python3.10/site-packages/nvidia/cublas/lib:${cuda_site}/cuda_runtime/lib:${cuda_site}/cufft/lib:${cuda_site}/curand/lib:${cuda_site}/cusolver/lib:${cuda_site}/cusparse/lib:${cuda_site}/cuda_cupti/lib:${cuda_site}/cuda_nvrtc/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PATH="${venv}/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${venv}/lib/python3.10/site-packages/nvidia/cuda_nvcc"
export PYTHONPATH="${project}/upstream/deepjscc_feedback_author:${project}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${venv}/bin/python" "$@"
