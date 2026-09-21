# FlowHARQ-JSCC

The VTC2027-Spring manuscript (2026-09-21 revision) studies **compute before
retransmit**: frozen H2 receiver repair and calibrated full-payload HARQ, supported
by H4's qualified recipe control. See the [paper](paper/main.tex),
[PDF](output/pdf/FlowHARQ_VTC2027.pdf), [current protocol](experiments/spring_final_20260921/protocol.md),
and [evidence audit](experiments/spring_final_20260921/RESULTS.md).
The original three-run study and existing ten-run extension remain distinct;
no new training or seed sweep is introduced. Explicit scale/control accounting
reduces payload-only savings to about 0.35–0.39% under the stated signaling model.

The [evidence-adaptive codec study](experiments/evidence_harq_20260920/RESULTS.md)
failed its feasibility gate (0% selective saving). Its code and negative results
remain archived, but it is not the manuscript's proposed method. H3's rate-aware
extension also remains exploratory after failing its cross-SNR acceptance rule.
These decisions do not erase or relabel either negative result.

Research prototype for **virtual retransmission via flow matching** in semantic
image communication. The receiver first repairs unreliable DeepJSCC latent
tokens with conditional flow matching and only requests a physical HARQ round
when predicted post-repair quality remains below the target.

The initial, submission-oriented scope is deliberately narrow:

1. SwinJSCC encoder/decoder and a time-varying Rayleigh channel;
2. token-wise reliability detection;
3. Reliability-Anchored Flow Matching (RAFM), which transports only unreliable
   tokens and leaves reliable tokens exactly unchanged;
4. a receiver-side quality predictor for ACK/NACK;
5. full second-round retransmission with instantaneous-CSI maximum-ratio
   combining under time-correlated Rayleigh fading.

Selective token retransmission is intentionally deferred until this core claim
is validated: at comparable reconstruction quality, FlowHARQ should reduce the
physical retransmission rate relative to adaptive DeepJSCC-HARQ.

## Scientific comparison

All five methods are evaluated from the same first-round reception:

| Method | FM repair | Physical retransmission policy |
|---|---:|---|
| `direct` | no | never |
| `full_harq` | no | always |
| `fm_only` | yes | never |
| `adaptive_harq` | no | predicted pre-FM quality |
| `flowharq` | yes | predicted post-FM quality |

Reported metrics include PSNR, SSIM, LPIPS, NACK/retransmission rate, mean
transmission rounds, paired confidence intervals, and receiver GPU latency.
Evaluation sweeps SNR and vehicle speed using paired first/second latent
receptions within every image-condition sample.

The 100 DIV2K validation images are deterministically split into 20 calibration
images and 80 untouched test images. ODE steps, reliability threshold, and
scalar quality-head biases are frozen on calibration data before the 3 training
seeds by 3 channel seeds test matrix is evaluated.

## Model and losses

For clean latent `z`, first-round reception `z_tilde`, and unreliable mask `m`,
training samples a straight conditional path

```text
z(t) = z_tilde + t * m * (z - z_tilde),   u(t) = z - z_tilde.
```

RAFM receives the current path state, the received latent, a pooled reliable
anchor, channel context, mask, and time embedding. Euler integration is wrapped
in an identity constraint:

```text
z_repaired = z_tilde + m * (integrated_z - z_tilde).
```

The training objective is

```text
L = lambda_fm L_fm + lambda_rel L_rel
  + lambda_q L_quality + lambda_rec L_reconstruction.
```

The H3 extension uses a constrained primal--dual objective to minimize expected
NACK rate under frozen PSNR and LPIPS budgets.  Only RAFM is fine-tuned by
default; keeping the quality head frozen prevents the soft policy from gaming
the rate term.  The locked single-run protocol is recorded in
`experiments/h3_cost_aware_training/protocol.md`.

The reliability target is the per-image top-error token subset. The quality
head learns log-MSE both before and after repair, allowing both adaptive HARQ
baselines to use the same type of receiver-available decision signal.

## Layout

- `flowharq/modules.py`: reliability head, RAFM velocity field, quality head.
- `flowharq/model.py`: SwinJSCC/channel wrapper and end-to-end losses.
- `flowharq/train.py`: backbone, flow-only, and joint training stages.
- `flowharq/evaluate.py`: paired baseline evaluation and summary generation.
- `flowharq/analyze.py`: run-level aggregation and paired 95% confidence intervals.
- `flowharq/paper_artifacts.py`: vector paper figures, LaTeX tables, and result macros.
- `flowharq/plot.py`: pilot PSNR, retransmission, and mobility plots.
- `flowharq/frontier.py`: threshold sweep for PSNR--retransmission Pareto curves.
- `tests/test_flowharq_modules.py`: invariants and gradient tests.
- `slurm/`: Artemis RTX GPU jobs.
- `upstream/SwinJSCC`: pinned upstream implementation.

## Local checks

```bash
git submodule update --init --recursive
python3 -m pip install -r requirements.txt
python3 -m compileall -q flowharq tests
python3 -m pytest -q
```

The confirmatory Artemis environment is locked in
`environment/artemis-cu128.requirements.txt` and
`environment/artemis-cu128.yml`.  It records Python 3.10, PyTorch 2.7.1 with
CUDA 12.8, and every evaluation dependency.  The lightweight root
`requirements.txt` remains convenient for CPU development.

The single-run H3 extension is executed on sheng's RTX 4090 with Python 3.10.15
and PyTorch 2.5.0+cu118.  Its direct dependency lock is
`environment/sheng-cu118.requirements.txt`; the run directory additionally
stores the full `pip freeze`, GPU identity, and SHA-256 hashes of code and the
H2 initialization checkpoint.

## Artemis workflow

The Artemis scheduler exposes RTX 6000-class nodes through the generic resource
request `--gres=gpu:RTX:1`. The jobs print the exact GPU model with `nvidia-smi`
at runtime, so hardware provenance is captured in the log.

The exact confirmatory jobs are retained under `slurm/`: H1 trains the robust
repair objective, H2 fine-tunes only the 24 dB decision head and evaluates the
3x3 seed matrix, and H4 trains the parameter-matched one-step control. Each
job writes into a hypothesis-specific result directory so exploratory and
confirmatory artifacts cannot overwrite one another.

The production defaults use DIV2K 256x256 crops, a 32-dimensional channel
bottleneck, a 35% oracle unreliable-token fraction for supervision, SNR in
[0, 15] dB, vehicle speed in [0, 120] km/h, and CSI age in [0, 5] ms. ODE steps
and the inference mask threshold are selected only by the calibration job; the
test job reads the resulting frozen JSON rather than accepting hand-tuned test
settings.

All Slurm paths can be overridden without editing job files.  See
`slurm/README.md` for `FLOWHARQ_PROJECT`, `FLOWHARQ_PYTHON`, dataset, and
backbone variables.  Jobs still record the resolved GPU identity at runtime.

See `RESULTS.md` for the final DIV2K, Kodak24, mechanism-control, and latency
results. The paired retransmission intervals exclude zero on both image sets,
so the core conference claim is supported within the documented limitations.
The later single-run H3 constrained extension is documented separately there:
it improves aggregate NACK/PSNR but remains a near-miss because it fails its
locked cross-SNR consistency rule.

## Paper artifacts

Regenerate every numeric macro, table, and result figure from stored analysis:

```bash
python3 -m flowharq.paper_artifacts \
  --analysis-dir results/h2/analysis \
  --external-analysis-dir results/h2/kodak_analysis \
  --calibration-json results/h2/seed_2030/decision.json \
  --repair-ablation-json results/h1/comparison.json \
  --one-shot-json results/h4/comparison.json \
  --latency-json results/h2/latency/rtx_pro_6000.json \
  --paper-dir paper
```

The VTC draft is `paper/main.tex`; the compiled PDF is
`paper/build/main.pdf`. Submission-specific checks and the current official
deadline/page rules are recorded in `paper/SUBMISSION_CHECKLIST.md`.

## Result integrity

Smoke outputs establish execution and gradient flow only; they are not paper
results. A result is reportable only after the full checkpoint, evaluation CSV,
summary JSON, Slurm logs, seed, and runtime GPU identity have all been retained.

`scripts/build_artifact_manifest.py` publishes a SHA-256 manifest containing
the exact code revision, upstream commits, runtime packages, dataset filename
manifests, checkpoints, calibration decisions, and aggregate tables.  The
Artemis publication job writes it to `artifacts/manifest.json` only after every
dependent training and analysis job succeeds.

## Published-paper baselines and external-method extensions

The completed author-checkpoint and author-architecture baseline package is in
`experiments/published_baselines_20260919/`. It contains SwinJSCC (IEEE TCCN
2025), NTSCC (IEEE JSAC 2022), and DeepJSCC-f (IEEE JSAIT 2020) evidence,
the shared PSNR/MSE/LPIPS/five-scale-MS-SSIM definitions, actual CBR accounting,
the selected public-weight hashes, original result CSVs, training histories,
environment locks, and reconstruction SHA-256 inventory. The completed
DeepJSCC-f result uses the authors' progressive CNN/fusion architecture rather
than the earlier mechanism adapter. See the [results table](experiments/published_baselines_20260919/RESULTS.md),
[protocol](experiments/published_baselines_20260919/protocol.md), and
[reproduction instructions](experiments/published_baselines_20260919/REPRODUCE.md).

The baseline package is an AWGN reference study, not a direct win/loss comparison
with the project's time-correlated Rayleigh/HARQ results. Its data-availability
record lists every raw CSV and all 48 reconstruction-array SHA-256 values. The
1.3-GB reconstruction arrays and TensorFlow checkpoints are deliberately not
Git objects; they can be regenerated from the supplied scripts and validated
against that inventory.

The locked seed-expansion protocol in
`experiments/h2_seed_expansion/protocol.md` adds seeds 2033--2039 without
reopening model selection, for ten independent training seeds in total.  The
completed DIV2K and Kodak intervals both retain a retransmission reduction and
pass the frozen quality guardrails; see the separate extension section in
`RESULTS.md`.

The legacy mechanism-level comparison in
`experiments/external_deepjscc_f/protocol.md` adapts the published DeepJSCC-f
feedback/incremental-redundancy mechanism to the same frozen first-round codec
and paired Rayleigh draws.  Its source provenance and limitations are recorded
explicitly; the adapter is not presented as the original authors' checkpoint.
The resulting comparison is deliberately retained even though it is not a
matched-quality FlowHARQ win. It remains a historical internal control, not the
published DeepJSCC-f reproduction; see `RESULTS.md`.
