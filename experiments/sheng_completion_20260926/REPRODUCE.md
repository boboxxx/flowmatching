# Reproducing the September 26 Sheng revision

## Decision-risk diagnostic (standard-library Python)

Extract `artifacts/sheng_completion_20260926/risk_inputs.tar.gz` into a **new**
directory. The archive contains only explicit `results/h2/...` input files.
Do not extract over unaccounted-for local results. Then run:

```sh
python3 scripts/audit_ack_risk.py --root /path/to/extracted-root --output /path/to/new-output
```

The script never reads test data before fixing both calibration margins.
It refuses existing output paths. All inputs and outputs are hashed.

## Published-codec transfer (GPU)

Use the exact project source snapshot at
`artifacts/spring_final_20260921/runtime/source/` as the import root, not the
possibly modified current `flowharq/` directory. Place
`scripts/evaluate_swin_fading_bridge.py` and
`scripts/evaluate_published_checkpoint.py` beneath its `scripts/` directory.
The `--base` path supplies the verified author Swin upstream, published
`swin_mse_c32.pth`/JSON, and the seed-2030 checkpoint/calibration. Model hashes
must match each dataset's `inputs.json`. The source snapshot and current
builder are needed because old and new metrics modules have different APIs.

```sh
/home/sheng/anaconda3/envs/cv/bin/python scripts/evaluate_swin_fading_bridge.py \
  --base /home/sheng/flowmatching_vtc2027 \
  --data /home/sheng/flowmatching_vtc2027/data/Kodak24 \
  --dataset kodak --output results/new_bridge_kodak
```

For DIV2K, use the full 100-image validation directory and `--dataset div2k`;
the first 20 images are excluded internally. Both invocations are recorded
verbatim in `runtime.json`. CUDA/Python/Torch and `pip-freeze.txt` are included.
The actual environment was Python 3.10.15, Torch 2.5.0+cu118 on an RTX 4090.
The exact downloaded model files must be available: this repository does not
pretend a hash manifest is a publicly downloadable checkpoint.

## Validate and regenerate manuscript tables

With the archived outputs under `artifacts/sheng_completion_20260926/` and
the original diagnostic CSV inputs available at their recorded paths:

```sh
python3 scripts/summarize_sheng_revision.py
python3 -m unittest discover -s tests -p 'test_ack_risk.py'
python3 -m unittest discover -s tests -p 'test_spring_evidence.py'
```

The summarizer checks prior image/checkpoint identities and generates all-method
results plus a compact manuscript table. Its validation status is not a claim
that scientific acceptance criteria succeeded. The one-shot dominance gate
failed on both datasets; the distribution-shift diagnostic also failed to
support reliable Kodak acceptance. These negative outcomes must be retained.

`paper/main.tex` includes generated tables and figures and uses two bibliography
files. Compile it as a multi-file IEEEtran project with TeX Live/latexmk. The
delivered PDF is `output/pdf/FlowHARQ_VTC2027.pdf`.
