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
- H1 calibration succeeds: normalized Huber velocity supervision and reconstruction weight 1.0 yield +0.0714 dB FM-only PSNR over direct (bootstrap 95% CI [+0.0570, +0.0866]) at K=4, tau=0.8, with LPIPS delta +0.00203. Four of six conservative inference settings pass the preregistered +0.05 dB / +0.003 LPIPS gate.
- At that frozen H1 point, calibration-only FlowHARQ saves 1.04 percentage points of physical retransmissions while improving final PSNR by +0.022 dB; this is a gate result, not held-out paper evidence.
- The frozen single-training-seed held-out gate over three fresh channel seeds saves 2.1875 percentage points of retransmissions versus adaptive HARQ, with +0.0327 dB PSNR and +0.00145 LPIPS. This licenses the preregistered three-training-seed expansion but is not yet the final confidence-interval claim.
- H1's full 3x3 held-out matrix establishes repair but not the retransmission claim. FlowHARQ minus adaptive HARQ is +0.0271 dB PSNR (95% CI [+0.0120,+0.0422]), +0.00098 LPIPS, and -1.389 pp NACK (95% CI [-3.240,+0.463] pp). All three training seeds reduce NACK individually, but n=3 and decision variability leave the interval crossing zero.
- H2's frozen 3x3 DIV2K result supports the central system claim: FlowHARQ minus adaptive HARQ is -1.296 pp NACK (95% CI [-2.570,-0.0225]), +0.0206 dB PSNR (95% CI [+0.0012,+0.0400]), and +0.00132 LPIPS.
- External Kodak24 validation strengthens the result: -2.025 pp NACK (95% CI [-3.343,-0.708]), +0.0284 dB PSNR (95% CI [-0.00028,+0.0570]), and +0.00128 LPIPS, using no Kodak fitting or selection.

## Patterns and Insights

1. More ODE steps are not automatically better. Strong integration often over-repairs; high mask thresholds or fewer steps are safer.
2. The current straight paired path makes the velocity target an unnormalized residual. Deep Rayleigh fades create large ZF outliers that dominate squared loss.
3. The system trains reconstruction, flow, reliability, and quality, but not the actual communication objective: matched final quality with fewer physical rounds.
4. A global scalar calibration bias cannot compensate for sample-dependent uncertainty around the service threshold.
5. A paired held-out stratification over all 3 training seeds and 3 channel seeds localizes the v2 failure to low SNR: FM-only minus direct PSNR is -0.0416 dB at 0 dB, -0.0135 dB at 3 dB, and approximately zero from 6--15 dB. Speed has little effect on this gap.
6. Training-seed variability is material: the paired FM-only PSNR delta is -0.0321, -0.0016, and +0.0058 dB for seeds 2027--2029. A positive claim cannot rest on the best checkpoint.
7. On the v2 calibration split at frozen K=2, tau=0.9, FM-only minus direct is -0.0121 dB, with bootstrap 95% CI [-0.0179, -0.0058]. Decision-threshold tuning alone cannot rescue the current repair module.
8. The final-epoch v2 FM loss is extremely heavy-tailed. Across seeds, median batch losses are 0.26--0.28 and 90th percentiles are 2.2--3.2, but 99th percentiles reach 203--2,209 and maxima reach 1,703--158,505. This directly supports H1's outlier-dominance mechanism rather than treating robust loss as an arbitrary hyperparameter sweep.
9. An oracle-decision audit proves that calibration is not the only current bottleneck. At the 24 dB target, v2 FM moves 37 held-out cases from NACK to ACK but moves 39 in the opposite direction, for an oracle NACK saving of -0.012 percentage points. The learned head's nominal +0.104-point saving is therefore a calibration artifact, not evidence of useful repair. H1 must succeed before H2 is scientifically meaningful.
10. H1 exhibits a real interaction: reconstruction weight alone reaches -0.0008 dB and normalized Huber alone +0.0021 dB, while their combination reaches +0.0714 dB. The improvement therefore comes from jointly suppressing outlier gradients and aligning the residual predictor with decoded image quality.
11. H1 FM-only PSNR improves for every training seed (+0.0739, +0.0358, +0.0574 dB). Final NACK changes are also directionally consistent (-2.19, -1.27, -0.71 pp), so H2 targets the magnitude and stability of the decision boundary rather than the repair network.
12. H2's mechanism metrics are mixed: calibration balanced accuracy changes by +0.0220, -0.0236, and -0.0015 across seeds, while ECE improves for only one seed. The paper may claim that target-boundary fine-tuning stabilizes held-out retransmission savings, but not that it universally improves calibration accuracy.

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
