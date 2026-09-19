# Reproduce the evidence-HARQ feasibility study on Sheng

Host used: DESKTOP-UGDDO8T, RTX 4090 24GB, Python 3.10.15, PyTorch 2.5.0+cu118.
Use the recorded per-run `pip-freeze.txt`; baseline NTSCC uses the existing compatible
overlay documented in `../published_baselines_20260919/REPRODUCE.md`.

The checkpoint downloaders retrieve only public author weights with SHA-256 sidecars.
Keep the source submodules at their recorded commits:

```sh
git submodule update --init upstream/SwinJSCC upstream/CDDM_author
python scripts/download_published_weights.py --output artifacts/published_weights
python scripts/download_cddm_weights.py --output artifacts/cddm_author_weights
```

Sheng's `cv` environment does not need gdown; its existing
`.venvs/deepjsccf_tf214/bin/python` is the downloader interpreter. Training and CDDM
evaluation use `/home/sheng/anaconda3/envs/cv/bin/python`. Do not mix TensorFlow and
PyTorch processes. Checkpoint loading verifies hashes and all neural weight keys.
The CDDM adapter imports the author's model/sampler files directly to avoid unrelated
MongoDB-writing training drivers, preserving numerical inference code.

Environment variables `EVIDENCE_PROJECT`, `EVIDENCE_PYTHON`, `EVIDENCE_DIV2K`,
`EVIDENCE_OUTPUT`, `NTSCC_PYTHON`, and `EVIDENCE_BASELINE_OUTPUT` parameterize paths.
Each fresh run needs a NEW output directory. Dataset defaults point to the existing
Sheng DIV2K train/valid images and project `data/Kodak24`.

```sh
python -m pytest -q tests/test_evidence_harq.py tests/test_metrics.py tests/test_channel.py
# Original pilot A; its failed gate is archived, not omitted:
bash scripts/run_evidence_harq_sheng.sh
# Original B is retained for audit, but failed with unbounded scales:
# bash scripts/run_evidence_codec_b_sheng.sh
# Numerically bounded B, with amendment and unchanged gate thresholds:
bash scripts/run_evidence_codec_b_fixed_sheng.sh
# Published references using charged side information:
bash scripts/run_paid_baselines_sheng.sh
python scripts/evaluate_author_cddm.py \
  --upstream upstream/CDDM_author --weights artifacts/cddm_author_weights \
  --data data/Kodak24 --output results/evidence_harq_20260920/cddm_kodak24.csv --draws 10
```

The codec scripts finish by enumerating all 16 subsets on 40 development images,
four paired noise draws, then emitting a gate decision. They do not launch posterior
or policy training when the gate fails. Full experiment logs distinguish completed,
failed, smoke-only and negative-gate runs. A source snapshot for B-fixed is verified
against launch-time hashes before archival; A/B retain launch source hashes and their
original argument configurations. The current code preserves their unbounded option.

The selected enhancement `best.pt` for completed A and B-fixed is included in the
corresponding archive directory (about 12 MB each), with optimizer and RNG states.
They are **failed feasibility candidates**, not recommended trained models. Only load
trusted checkpoints: their training state requires PyTorch's pickle-based loading.
Redundant `last.pt` and the failed B checkpoint remain on Sheng; per-run
`artifact_inventory.json` records their sizes and hashes. Public author CDDM weights
include a ~2GB file and are downloaded from the authors rather than redistributed.
Compact CSVs, split manifests, logs, environment and decisions are mirrored under
`artifacts/evidence_harq_20260920/`. Paths in original configurations are historical;
use the evaluation CLI path overrides when replaying on another host.

`provenance.json` is an immutable launch record, so its `status` can say `running`;
read `status.json` for the terminal execution state and `gate_decision.json` for the
scientific continuation decision. `runtime/` is a separately labelled post-run
environment/source snapshot, not a claim to have captured every launch environment.
The version-pinned dependency snapshots do not provide a hermetic wheel/build lock;
OS, CUDA build and driver details are retained to make that limitation explicit.

```sh
python scripts/audit_evidence_archive.py
python scripts/summarize_evidence_feasibility.py
```

For example, replay a downloaded B-fixed checkpoint on relocated datasets without
editing the archived configuration or split hashes:

```sh
python -m evidence_harq.evaluate_codec \
  --run artifacts/evidence_harq_20260920/candidate_b_fixed --split oracle_gate \
  --data-root /path/to/DIV2K_valid_HR --upstream upstream/SwinJSCC \
  --base-checkpoint artifacts/published_weights/swin_mse_c32.pth \
  --output /path/to/new_results/oracle_gate.csv --draws 4 --snrs 10
```

The control-link reliability model is an assumption (0.5 info bits/complex use), not
a physical FEC implementation. Existing DeepJSCC-f ideal output-feedback results are
kept in their original package; they are not merged into the finite-feedback frontier.
No 5% ACK-risk certificate can be asserted from the current 40-image pilot split.
