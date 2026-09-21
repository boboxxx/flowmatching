# FlowHARQ confirmatory results — 2026-09-19

## Current manuscript audit — 2026-09-21

The Spring paper returns to the frozen H2 claim; H3 and the failed progressive
codec are not primary evidence. See [the new audit](experiments/spring_final_20260921/RESULTS.md)
for exact CSV-derived decision decomposition, side-information accounting, and
the separately reported Artemis binary16/four-metric check.

For the existing ten-run extension, NACK reduction is 1.319444 pp on DIV2K and
1.452546 pp on Kodak24. These correspond to 0.758430%/0.845972% payload savings,
or 0.352913%/0.390886% charged-use savings with 256 binary16 token scales sent
once and metadata/control efficiency 0.5 bit per complex use. The latter is
analytical re-accounting of original ideal-scale decisions, not measured air-interface
efficiency. LPIPS worsens within the original +0.003 tolerance; it does not improve.
The PSNR tolerance remains the original -0.02 dB, not H3's different tolerance.

Published AWGN references below and in the newer paid-metadata table are not
matched-channel superiority comparisons against the Rayleigh/HARQ experiment.

## Published-paper baseline package — completed

The completed published-baseline package is deliberately separate from the
Rayleigh/HARQ confirmatory table below. It evaluates author public checkpoints
for SwinJSCC and NTSCC at AWGN 10 dB, plus a from-scratch author-architecture
DeepJSCC-f reproduction trained on DIV2K. All use Kodak24 256x256 center crops
where noted, RGB [0,1] PSNR/MSE, standard five-scale MS-SSIM, and AlexNet LPIPS.
Each row averages 10 independent channel draws per Kodak image—not multiple
training seeds. Full raw CSVs, hashes, reconstruction inventory, protocol and
environment locks are under `experiments/published_baselines_20260919/` and
`artifacts/published_*_20260919/`.

| Method | Setting | CBR | PSNR | MS-SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| SwinJSCC MSE C32 | public weights, full Kodak, AWGN 10 dB | 0.02083 | 29.369 | 0.95673 | 0.22077 |
| SwinJSCC MS-SSIM C32 | public weights, full Kodak, AWGN 10 dB | 0.02083 | 27.620 | 0.96701 | 0.22267 |
| NTSCC w/o z Q1 | public weights, full Kodak, AWGN 10 dB | 0.02548* | 29.795 | 0.95710 | 0.22176 |
| DeepJSCC-f, 2 rounds | author CNN/fusion, Kodak crop, AWGN 10 dB | 0.04167 | 26.387 | 0.93746 | 0.30644 |
| DeepJSCC-f CNN control | same total forward CBR, Kodak crop, AWGN 10 dB | 0.04167 | 25.322 | 0.91345 | 0.33157 |

\*NTSCC total includes the authors' capacity-limit rate-map signalling model;
the payload-only CBR is 0.02398. The DeepJSCC-f feedback model assumes ideal
channel-output feedback, which is not equivalent to a 1-bit ACK/NACK. These
results therefore establish transparent external references, not a matched
channel/feedback-budget comparison with FlowHARQ. See
`experiments/published_baselines_20260919/RESULTS.md` for every operating point.

## Claim

Receiver-side four-step reliability-anchored flow matching (RAFM), followed by
a target-aware ACK/NACK head, replaces a small but statistically supported
fraction of physical HARQ rounds while keeping reconstruction quality inside
predeclared tolerances.

The tolerances were FlowHARQ minus adaptive HARQ PSNR >= -0.02 dB and LPIPS
<= +0.003. The service target was fixed at 24 dB. Calibration selected K=4
and mask threshold 0.80 before the confirmatory evaluations.

## Main DIV2K result

Protocol: three independent training seeds (2030--2032), three fresh channel
seeds (8001--8003), 80 untouched DIV2K validation images, six SNRs, four
speeds, paired channel draws, and Student-t confidence intervals over the three
training-seed means after averaging channel seeds.

| FlowHARQ minus adaptive HARQ | Mean | 95% CI |
|---|---:|---:|
| Physical retransmission rate | **-1.296 pp** | **[-2.570, -0.0225] pp** |
| PSNR | **+0.0206 dB** | **[+0.0012, +0.0400] dB** |
| LPIPS | +0.00132 | [+0.00051, +0.00214] |
| SSIM | +0.00056 | [-0.00027, +0.00138] |

The retransmission interval excludes zero and both quality guardrails pass.
This is the central paper result.

## External Kodak24 result

The full model, calibration, K, mask threshold, and service target were frozen
before evaluating all 24 Kodak images with channel seeds 9001--9003.

| FlowHARQ minus adaptive HARQ | Mean | 95% CI |
|---|---:|---:|
| Physical retransmission rate | **-2.025 pp** | **[-3.343, -0.708] pp** |
| PSNR | +0.0284 dB | [-0.00028, +0.0570] dB |
| LPIPS | +0.00128 | [+0.00034, +0.00221] |
| SSIM | -0.00009 | [-0.00050, +0.00033] |

No Kodak sample was used for model fitting or operating-point selection.

## Locked ten-seed robustness extension

After the three-seed paper result was frozen, seeds 2033--2039 were added under
the locked protocol without changing the model, threshold, target, split, or
channel seeds.  Confidence intervals again operate on independent
training-seed means after averaging the three channel seeds.

| FlowHARQ minus adaptive HARQ | DIV2K mean [95% CI] | Kodak24 mean [95% CI] |
|---|---:|---:|
| Physical retransmission rate | **-1.319 pp [-1.537, -1.102]** | **-1.453 pp [-1.825, -1.081]** |
| PSNR | +0.0144 dB [+0.00895, +0.0199] | +0.0253 dB [+0.0193, +0.0312] |
| LPIPS | +0.00143 [+0.00118, +0.00169] | +0.00131 [+0.00103, +0.00159] |
| SSIM | +0.000383 [+0.000136, +0.000630] | -0.000108 [-0.000240, +0.000023] |

Both datasets retain a retransmission interval strictly below zero and pass the
predeclared PSNR/LPIPS tolerances.  The DIV2K and Kodak analyses contain 30
training/channel pairs (403,200 and 120,960 paired rows, respectively).  This
is reported as a post-paper robustness extension rather than a replacement for
the original three-seed confirmatory analysis.

## External HARQ mechanism baseline

Three independently trained DeepJSCC-f adapters were evaluated on the same
80-image DIV2K test split, channel seeds, SNRs, speeds, and first-round channel
draws as the original three FlowHARQ seeds.  The adapter uses the published
noiseless-output-feedback/incremental-redundancy mechanism with the shared
frozen SwinJSCC codec; it is not the original authors' checkpoint.  Its
adaptive rule uses true first-round PSNR at the transmitter, making stopping
intentionally favorable to the external baseline.

| FlowHARQ minus DeepJSCC-f adaptive | Mean | 95% CI |
|---|---:|---:|
| Physical retransmission rate | -2.685 pp | [-6.147, +0.777] pp |
| PSNR | **-0.1954 dB** | **[-0.2153, -0.1755] dB** |
| LPIPS | +0.00223 | [-0.00204, +0.00650] |
| SSIM | -0.00873 | [-0.01227, -0.00520] |

The external adapter therefore improves the quality side of the tradeoff.  The
FlowHARQ retransmission reduction against it is not statistically conclusive
and is not a matched-quality win.  This result is a pressure test, not evidence
for replacing the external baseline.

## Repair and mechanism controls

On the calibration split at fixed K=4 and threshold 0.80:

| Training/control | FM-only minus direct PSNR | LPIPS delta |
|---|---:|---:|
| Squared velocity loss | -0.0619 dB | +0.00049 |
| Squared loss + reconstruction alignment | -0.0172 dB | +0.00067 |
| Normalized Huber velocity loss | -0.0129 dB | +0.00171 |
| **Normalized Huber + reconstruction alignment** | **+0.0714 dB** | +0.00203 |
| Parameter-matched one-step t=0 control (best threshold) | -0.0147 dB | +0.00093 |

The one-step control's PSNR interval is [-0.0232, -0.0055] dB. Four-step
RAFM exceeds it by 0.0861 dB, passing the preregistered 0.02 dB
flow-specificity gate. The robust/reconstruction ablation also shows that the
successful objective is an interaction, not a single-component threshold
effect.

## Runtime and model size

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Frozen codec: 21.83 M parameters.
- Receiver-only reliability, RAFM, and quality modules: 0.479 M trainable parameters.
- Direct decode: 3.11 ms/image.
- Virtual repair plus decode (K=4): 5.00 ms/image.
- Local second-round MRC combine plus decode: 3.06 ms/image.

The physical-round timing excludes wireless airtime, scheduling, propagation,
and queueing, so it is not an end-to-end latency comparison. Virtual repair
adds approximately 1.9 ms over direct decoding.

## H3 constrained rate--distortion extension (single run)

H3 was trained once from the frozen H2 seed-2030 checkpoint on sheng's RTX
4090. Training used seed 2027; checkpoints at epochs 2, 4, 6, 8, and 10 were
compared only on the 20-image calibration split. All five met the aggregate
PSNR/LPIPS constraints, and epoch 10 was selected before opening the 80-image
test split. The values below are exact paired means over one locked channel
realization (seed 7070), not a training-seed average.

| 80-image DIV2K test, 1,920 conditions | Adaptive HARQ | H2 FlowHARQ | H3 FlowHARQ |
|---|---:|---:|---:|
| Physical retransmission rate | 73.958% | 71.875% | **70.833%** |
| Mean transmission rounds | 1.7396 | 1.7188 | **1.7083** |
| PSNR | 21.9476 dB | 21.9792 dB | **22.0649 dB** |
| LPIPS | **0.485024** | 0.486666 | 0.487205 |

Relative to frozen H2, H3 reduces NACK rate by **1.042 percentage points**,
adds **0.0857 dB** PSNR, and changes LPIPS by **+0.000538**. Relative to
adaptive HARQ, H3 reduces NACK rate by **3.125 percentage points**, adds
**0.1174 dB** PSNR, and changes LPIPS by **+0.002180**, within the aggregate
`-0.02 dB / +0.003` quality budgets.

This is a preregistered **near-miss, not a confirmatory pass**. H3 improves
NACK rate at SNR 0, 3, and 6 dB, but ties H2 at 9, 12, and 15 dB. It therefore
passes the aggregate rate, PSNR, and LPIPS rules but fails the locked
requirement for a positive saving at at least four of six SNR points (3/6).
The earlier quality-head-gated formulation was stopped after its LPIPS
constraint remained violated and its quality-proxy error grew; its logs are
retained on sheng and are not reported as a paper result.

The already frozen epoch-10 checkpoint and DIV2K calibration bias were then
applied once to Kodak24, again with channel seed 7070 and no Kodak tuning:

| Kodak24, 576 conditions | Adaptive HARQ | H2 FlowHARQ | H3 FlowHARQ |
|---|---:|---:|---:|
| Physical retransmission rate | 69.792% | 68.576% | **66.493%** |
| PSNR | 22.0409 dB | 22.0804 dB | **22.1951 dB** |
| LPIPS | **0.522318** | 0.524500 | 0.525314 |

H3 reduces NACK rate by **2.083 percentage points** versus H2 and by **3.299
percentage points** versus adaptive HARQ. Relative to adaptive HARQ, PSNR is
`+0.1542 dB` and LPIPS is `+0.002996`; four of six SNR points have a positive
H2-relative saving. This external check passes the aggregate locked rules, but
the LPIPS margin is only about `0.000004`, and the 0/3 dB LPIPS deltas exceed
the budget when inspected separately. It supports H3 as an exploratory
extension; it does not override the primary DIV2K cross-SNR rejection.

## Scientific limitations

- The NACK saving is modest (about 1--2 percentage points).
- The channel is simulated flat Rayleigh fading with Jakes temporal correlation.
- The SwinJSCC codec is frozen; the receiver is trained around it.
- A NACK triggers a full second round rather than selective-token retransmission.
- The external DeepJSCC-f comparison is a mechanism-level adapter on the shared
  codec/channel, not a reproduction of the original CIFAR/AWGN checkpoint.
- The separate published-paper baseline package includes a true author-code
  DeepJSCC-f reproduction, but it uses AWGN and ideal channel-output feedback;
  it must not be merged into the Rayleigh/HARQ table above as a direct comparison.
- Boundary balanced accuracy and ECE do not improve for every training seed;
  the supported decision claim is the paired held-out outcome, not universal
  calibration superiority.
- The single-run H3 extension improves aggregate rate and quality, but misses
  its predeclared cross-SNR consistency rule and is not yet a confirmatory
  paper claim.
- H4 is a paired calibration mechanism test, not a second held-out benchmark.

## Reproducibility pointers

- Paper macros/tables: `paper/generated/`
- Main analysis: `results/h2/analysis/`
- Kodak analysis: `results/h2/kodak_analysis/`
- Ten-seed analyses: `results/h2/analysis_10seeds/` and
  `results/h2/kodak_analysis_10seeds/`
- H1 objective ablation: `results/h1/comparison.json`
- H4 one-step control: `results/h4/comparison.json`
- Latency: `results/h2/latency/rtx_pro_6000.json`
- External DeepJSCC-f analysis: `results/external_deepjscc_f/analysis/`
- H3 checkpoint, calibration, paired test CSVs, and decision:
  `results/h3/main_oracle_seed2027/`
- H3 checksummed inventory: `artifacts/h3_manifest.json`
- Frozen Kodak protocol: `experiments/h2_decision_calibration/kodak_protocol.md`
- Slurm job definitions: `slurm/`
- Checksummed artifact inventory: `artifacts/manifest.json`
- Paper: `paper/main.tex` and `paper/build/main.pdf`
- Published baselines, raw CSVs, and result documents:
  `experiments/published_baselines_20260919/` and `artifacts/published_*_20260919/`
