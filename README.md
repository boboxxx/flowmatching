# FlowHARQ-JSCC

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
python3 -m compileall -q flowharq tests
python3 -m pytest -q
```

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

See `RESULTS.md` for the final DIV2K, Kodak24, mechanism-control, and latency
results. The paired retransmission intervals exclude zero on both image sets,
so the core conference claim is supported within the documented limitations.

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
