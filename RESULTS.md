# FlowHARQ confirmatory results — 2026-09-19

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

## Scientific limitations

- The NACK saving is modest (about 1--2 percentage points).
- The channel is simulated flat Rayleigh fading with Jakes temporal correlation.
- The SwinJSCC codec is frozen; the receiver is trained around it.
- A NACK triggers a full second round rather than selective-token retransmission.
- Boundary balanced accuracy and ECE do not improve for every training seed;
  the supported decision claim is the paired held-out outcome, not universal
  calibration superiority.
- H4 is a paired calibration mechanism test, not a second held-out benchmark.

## Reproducibility pointers

- Paper macros/tables: `paper/generated/`
- Main analysis: `results/h2/analysis/`
- Kodak analysis: `results/h2/kodak_analysis/`
- H1 objective ablation: `results/h1/comparison.json`
- H4 one-step control: `results/h4/comparison.json`
- Latency: `results/h2/latency/rtx_pro_6000.json`
- Frozen Kodak protocol: `experiments/h2_decision_calibration/kodak_protocol.md`
- Slurm job definitions: `slurm/`
- Paper: `paper/main.tex` and `paper/build/main.pdf`
