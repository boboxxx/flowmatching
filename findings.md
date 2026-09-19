# Research Findings

## Research Question

Can receiver-side reliability-anchored flow matching act as a zero-airtime HARQ round, reducing physical retransmissions at matched image quality?

## Current Understanding

The architectural proposition is feasible and inexpensive: a 0.479 M-parameter receiver module can preserve reliable tokens exactly, use instantaneous receiver CSI, and add about 1.18 ms over direct decoding on an RTX PRO 6000.  The present bottleneck is not implementation or runtime.  It is statistical effect size.

The leakage-free v2 protocol shows essentially matched PSNR but no significant retransmission saving.  Flow repair changes reconstruction quality by only hundredths of a dB, while the quality head's calibration error is close to one dB.  Consequently the post-flow ACK/NACK decision cannot reliably exploit the small repair effect.  Inference-threshold sweeps cannot fix this mechanism.

## Key Results

- v2 uses 3 independent training seeds, 3 channel seeds, 80 untouched test images, paired channel draws, receiver CSI, correlated HARQ rounds, MRC, SSIM, LPIPS, and latency.
- Retransmission-rate delta: -0.10 percentage points, 95% CI [-0.41, 0.20]; inconclusive.
- PSNR delta: -0.004 dB, 95% CI [-0.023, 0.016]; matched quality.
- Virtual repair plus decode: 4.23 ms; direct decode: 3.05 ms.
- Conservative masks prevent hallucination, but also leave too little repair benefit to change many ACK decisions.

## Patterns and Insights

1. More ODE steps are not automatically better. Strong integration often over-repairs; high mask thresholds or fewer steps are safer.
2. The current straight paired path makes the velocity target an unnormalized residual. Deep Rayleigh fades create large ZF outliers that dominate squared loss.
3. The system trains reconstruction, flow, reliability, and quality, but not the actual communication objective: matched final quality with fewer physical rounds.
4. A global scalar calibration bias cannot compensate for sample-dependent uncertainty around the service threshold.
5. A paired held-out stratification over all 3 training seeds and 3 channel seeds localizes the v2 failure to low SNR: FM-only minus direct PSNR is -0.0416 dB at 0 dB, -0.0135 dB at 3 dB, and approximately zero from 6--15 dB. Speed has little effect on this gap.
6. Training-seed variability is material: the paired FM-only PSNR delta is -0.0321, -0.0016, and +0.0058 dB for seeds 2027--2029. A positive claim cannot rest on the best checkpoint.
7. On the v2 calibration split at frozen K=2, tau=0.9, FM-only minus direct is -0.0121 dB, with bootstrap 95% CI [-0.0179, -0.0058]. Decision-threshold tuning alone cannot rescue the current repair module.

## Lessons and Constraints

- Never tune and report on the same 100 DIV2K images; retain deterministic calibration/test separation.
- Report confidence intervals over training seeds, not over correlated images or channel draws.
- Do not claim that FM itself helps until a parameter-matched one-shot denoiser control is beaten.
- Preserve reliable-token identity as a hard architectural invariant.
- Treat a confidence interval crossing zero as inconclusive, even if the mean has the desired sign.
- Avoid further inference-only threshold sweeps until actual repair quality improves.

## Open Questions

- Can robust/normalized velocity training produce a measurable image-level repair gain?
- Can a target-aware decision objective cut boundary errors enough to expose that gain?
- Does a differentiable retransmission Lagrangian improve the true rate-quality frontier?
- Is the multi-step flow mechanism better than a simpler residual denoiser?
- Does the effect become stronger under mobility or bursty/selective corruption?

## Optimization Trajectory

The exploratory pilot reached a nominal 1 pp saving but did not survive leakage-free replication.  The current held-out baseline is 0.10 pp with a CI crossing zero.  The next accepted advance must improve the held-out constrained saving, not merely training PSNR or calibration performance.
