# Reproduce on sheng

Work from `/home/sheng/flowmatching_vtc2027`. Model/data/output paths are CLI options;
the shell entrypoints also accept `FLOWHARQ_PROJECT`, `FLOWHARQ_PYTHON`,
`FLOWHARQ_TRAIN_DATA`, `FLOWHARQ_VALID_DATA`, `PUBLISHED_OUTPUT`, `NTSCC_PYTHON`,
`DEEPJSCCF_VENV`, `DEEPJSCCF_OUTPUT`, and `CUDA11_SITE` where relevant.

## Upstream checkouts

Use dedicated checkouts; preserve author licenses and never substitute our old adapter.

```sh
git clone https://github.com/kurka/deepJSCC-feedback.git upstream/deepjscc_feedback_author
git -C upstream/deepjscc_feedback_author checkout 61a853baf019d589a7be09dc443c72985b891a74
git clone https://github.com/wsxtyrdd/NTSCC_JSAC22.git upstream/ntscc_author
git -C upstream/ntscc_author checkout dd3c745374023af6d01f465c379122be785d936f
git -C upstream/ntscc_author apply ../../environment/ntscc-compat.patch
git -C upstream/SwinJSCC rev-parse HEAD
# Verified SwinJSCC revision: a6d0e6da53548976acbe9317839a077ef31f190f
```

If paths already exist, inspect them; do not blindly clone over or reset them.
The only NTSCC source modification is uint8 mask -> bool. The author DeepJSCC-f
source remains unmodified; all data loading/training orchestration is separate.

## Environments

Existing PyTorch environment: `/home/sheng/anaconda3/envs/cv`, captured in
`environment/torch-pip-freeze.txt`. The four-metric dependency is
`pytorch-msssim==1.0.0`, in the project requirements and Sheng lock.

```sh
/home/sheng/anaconda3/envs/cv/bin/python -m venv .venvs/deepjsccf_tf214
.venvs/deepjsccf_tf214/bin/pip install -r environment/deepjsccf-tf214.requirements.txt
/home/sheng/anaconda3/envs/cv/bin/python -m venv --system-site-packages .venvs/ntscc
.venvs/ntscc/bin/pip install --no-deps compressai==1.2.0
```

Exact snapshots: `environment/deepjsccf-pip-freeze.txt`,
`environment/ntscc-pip-freeze.txt`, `environment/torch-pip-freeze.txt`.
The NTSCC overlay depends on the captured PyTorch base environment; it is not a
standalone full lock. DeepJSCC-f's launch wrapper explicitly loads cuDNN8 and CUDA11
instead of the existing PyTorch cuDNN9. It requires the CUDA11_SITE runtime libraries.
Do not import TensorFlow and PyTorch in one evaluation process.

## Evaluate official pretrained checkpoints

```sh
.venvs/deepjsccf_tf214/bin/python scripts/download_published_weights.py --output artifacts/published_weights
bash scripts/run_author_checkpoint_benchmark.sh
python scripts/summarize_published_baselines.py --input results/published_baselines_20260919/author_benchmark --output experiments/published_baselines_20260919/RESULTS.md
```

Downloads are only the six selected public files, not whole author folders. JSON
sidecars record source URL, objective, bytes and SHA-256; evaluation rechecks the hash.
Existing complete benchmark files are skipped, partial files are not silently replaced.
Use a fresh `PUBLISHED_OUTPUT` for an independent rerun.

## Train and evaluate original DeepJSCC-f architecture

```sh
bash scripts/run_author_deepjsccf_training.sh
# The wrapper now runs export/scoring automatically after both training jobs.
# The following individual commands are for a fresh manual evaluation directory:
bash scripts/deepjsccf_runtime.sh scripts/export_author_deepjsccf.py --upstream upstream/deepjscc_feedback_author --run results/published_baselines_20260919/deepjsccf_author_v2/c2_l2 --data data/Kodak24 --output results/published_baselines_20260919/deepjsccf_author_v2/c2_l2/manual_crop256 --crop 256
/home/sheng/anaconda3/envs/cv/bin/python scripts/score_author_deepjsccf.py --input results/published_baselines_20260919/deepjsccf_author_v2/c2_l2/manual_crop256 --output results/published_baselines_20260919/deepjsccf_author_v2/c2_l2/manual_crop256.csv
```

Repeat export/scoring with `c4_l1` for the equal-total-forward-budget, no-feedback
control; use `--crop 0` and a distinct output directory for native-resolution test.
Float reconstruction exports are deliberately distinct from scored CSVs. A smoke run
or incomplete training status is rejected by the exporter. No test score influences
early stopping, loss selection or checkpoint selection.

The initial `deepjsccf_author` attempt is invalid due to incomplete Keras
`.weights.h5` serialization of TFC parameters. Do not use its numbers or checkpoints.
V2 uses verified legacy `.h5`; see the serialization regression and protocol notes.
