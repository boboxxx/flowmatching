# FlowHARQ pilot results — 2026-09-19

## Run provenance

- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Backbone job: `11396925`, completed in 15:11.
- FlowHARQ smoke job: `11396935`, completed in 1:10; 4/4 tests passed.
- FlowHARQ training job: `11396950`, completed in 9:13; exit code `0:0`.
- Evaluation job: `11396951`, completed in 1:12; exit code `0:0`.
- ODE/mask ablation job: `11397022`, completed in 6:51; exit code `0:0`.
- Training: 40 epochs, 16,000 steps, seed 2027, DIV2K train, 256x256 crops.
- Validation: 100 DIV2K validation images, 6 SNRs, 4 speeds, five paired methods.

The selected inference point is four Euler steps with an unreliable-token
probability threshold of 0.80. It was chosen from the recorded ablation rather
than assumed in advance.

## Selected-point result

Differences below are FlowHARQ minus adaptive HARQ at the same 24 dB receiver
quality threshold. Retransmission values are absolute percentage-point changes.

| SNR (dB) | PSNR change (dB) | Retransmission change | FM-only vs direct (dB) |
|---:|---:|---:|---:|
| 0  | +0.002 | -1.00 pp | -0.031 |
| 3  | +0.011 | -0.50 pp | -0.017 |
| 6  | +0.003 | -2.00 pp | -0.002 |
| 9  | -0.003 | -1.75 pp | +0.005 |
| 12 | +0.007 | -0.50 pp | +0.009 |
| 15 | +0.005 | -0.25 pp | +0.003 |
| Mean | **+0.0043** | **-1.00 pp** | **-0.0056** |

The pilot therefore supports a modest version of the central claim: conservative
partial flow repair reduces physical NACKs without a measurable mean PSNR loss.
It does **not** support a claim of large reconstruction gains.

## Threshold-sweep robustness

Across 49 ACK thresholds per SNR, a FlowHARQ operating point Pareto-dominates
an adaptive-HARQ point (no greater retransmission and no lower PSNR) in:

| SNR (dB) | Adaptive points dominated by FlowHARQ | FlowHARQ points dominated by adaptive HARQ |
|---:|---:|---:|
| 0  | 37/49 | 30/49 |
| 3  | 44/49 | 26/49 |
| 6  | 43/49 | 21/49 |
| 9  | 25/49 | 22/49 |
| 12 | 47/49 | 10/49 |
| 15 | 43/49 | 8/49 |

This is substantially stronger than the unconstrained 0.50 mask threshold,
whose curve was often dominated because it modified too many reliable tokens.

## Artifacts

- Main per-image evaluation: `results/eval/flowharq.csv`
- Main fixed-threshold summary: `results/eval/flowharq.summary.json`
- Selected configuration: `results/ablations/k4_t080.csv`
- Selected summary: `results/ablations/k4_t080.summary.json`
- Selected Pareto data/figure: `results/ablations/k4_t080_frontier.{csv,pdf,png}`
- Main figures: `results/eval/figures/`

## What remains before a paper claim

1. Run at least three seeds and report paired confidence intervals.
2. Separate threshold calibration from the final test split.
3. Add SSIM, LPIPS, wall-clock FM latency, and transmitted-symbol accounting.
4. Expose receiver channel magnitude/effective noise variance to the reliability
   head; the pilot currently uses SNR, speed, CSI age, correlation, and received
   token statistics.
5. Replace equal-gain second-round combining with CSI-weighted MRC.
6. Only after the core result survives those checks, add selective-token HARQ.
