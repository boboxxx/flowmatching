# Research Findings

## Current evidence boundary — 2026-09-26

This update supersedes stronger interpretations in the historical narrative
below; those entries remain as a record of the research trajectory.

- The completed 2026-09-21/22 audit supports **mean-quality-constrained cost
  reduction**, not reliable target delivery. Charged-use savings are 0.353%
  on DIV2K and 0.391% on Kodak at the stated metadata/control efficiency.
- Conditional false ACK is about 14% on DIV2K and 51% on Kodak; final outage
  is substantial. A global quality bias is not a risk certificate.
- H4 is calibration evidence with different time supervision and threshold
  selection, not a held-out isolation of integration depth or proof that flow
  matching is necessary. The historical image-condition bootstrap interval
  should not be presented as an independent-image confidence interval.
- Author-model AWGN results are published references, not matched Rayleigh
  HARQ superiority evidence. Existing GPU microbenchmarks omit the quality
  head and radio system; they do not prove end-to-end latency reduction.
- The September 26 rewrite follows a problem/design/evidence structure informed
  by FlowIE and ResFlow. It adds an exact paired quality decomposition, not
  a fabricated theoretical guarantee. See `literature/cvpr_rewrite_20260926.md`.
- Sheng authentication succeeded and the September 26 frozen diagnostics
  completed: 14,976 new image/method rows plus archived decision-risk analysis.
  At a calibration-selected 1-dB margin, FlowHARQ conditional false ACK is
  3.86% on DIV2K but 30.32% on Kodak. Calibration risk does not transfer.
- On paired Kodak fading, published SwinJSCC with our two-round Chase wrapper
  achieves 26.34 dB versus FlowHARQ's 22.05 dB, with better MSE/MS-SSIM/LPIPS
  and lower charged uses (8228 vs 15171). DIV2K is not an all-metric dominance
  result: the author wrapper has worse mean MSE despite higher mean PSNR.
  See `experiments/sheng_completion_20260926/RESULTS.md`.
- These checks complete the bounded diagnostic, not a competitive-system
  claim. Moving to an author-pretrained codec requires new training and
  calibration; it is not a cosmetic checkpoint replacement.

## Historical research narrative (read with the qualifications above)

## Research Question

Can receiver-side reliability-anchored flow matching act as a zero-airtime HARQ round, reducing physical retransmissions at matched image quality?

## Current Understanding

The final conference-scale prototype supports the core proposition.  A 0.479 M-parameter receiver module preserves reliable tokens exactly, uses instantaneous receiver CSI, and performs a four-step virtual retransmission plus decoding in 5.00 ms on an RTX PRO 6000.  Robust partial flow repair plus a target-aware quality head reduces physical NACKs at matched quality under a leakage-free 3x3 seed protocol and on frozen Kodak24 validation.

The effect is deliberately stated as modest: roughly 1--2 percentage points fewer physical retransmissions, not a large reconstruction gain.  The original squared-loss v2 system was inconclusive; the supported result depends on robust normalized velocity supervision, reconstruction alignment, and a service-boundary decision stage.

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
- H4 rules out a parameter-matched one-step residual explanation: its best conservative threshold gives -0.0147 dB PSNR (95% CI [-0.0232,-0.0055]) and +0.00093 LPIPS, versus +0.0714 dB for four-step RAFM.  The paired 0.0861 dB gap passes the preregistered 0.02 dB mechanism gate.
- Final RTX PRO 6000 latency is 3.11 ms for direct decoding, 5.00 ms for virtual repair plus decoding, and 3.06 ms for local physical-round combining plus decoding.  The latter excludes airtime, scheduling, and propagation.

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
10. H1 exhibits a real interaction at fixed K=4, tau=0.8: reconstruction alignment alone reaches -0.0172 dB and normalized Huber alone -0.0129 dB, while their combination reaches +0.0714 dB. The improvement therefore comes from jointly suppressing outlier gradients and aligning the residual predictor with decoded image quality.
11. H1 FM-only PSNR improves for every training seed (+0.0739, +0.0358, +0.0574 dB). Final NACK changes are also directionally consistent (-2.19, -1.27, -0.71 pp), so H2 targets the magnitude and stability of the decision boundary rather than the repair network.
12. H2's mechanism metrics are mixed: calibration balanced accuracy changes by +0.0220, -0.0236, and -0.0015 across seeds, while ECE improves for only one seed. The paper may claim that target-boundary fine-tuning stabilizes held-out retransmission savings, but not that it universally improves calibration accuracy.
13. The flow-specificity control is decisive on calibration data: a one-step model trained and evaluated at $t=0$ degrades PSNR at both conservative mask thresholds, while four-step RAFM improves it.  This supports trajectory conditioning rather than parameter count or residual supervision as the differentiator.

## Lessons and Constraints

- Never tune and report on the same 100 DIV2K images; retain deterministic calibration/test separation.
- Report confidence intervals over training seeds, not over correlated images or channel draws.
- The matched one-shot control has now been beaten, but the claim remains limited to the tested architecture, calibration split, and latency regime.
- Preserve reliable-token identity as a hard architectural invariant.
- Treat a confidence interval crossing zero as inconclusive, even if the mean has the desired sign.
- Avoid further inference-only threshold sweeps until actual repair quality improves.

## Open Questions

- Does a differentiable retransmission Lagrangian improve the true rate-quality frontier?
- Does the effect become stronger under mobility or bursty/selective corruption?
- Does the result survive a jointly trained codec, measured vehicular channels, and additional datasets?

## Optimization Trajectory

The exploratory pilot reached a nominal 1 pp saving but did not survive leakage-free replication.  H1 repaired the latent mechanism, H2 converted that gain into a statistically supported held-out NACK reduction, and external Kodak24 validation replicated it.  H4 then separated four-step RAFM from a parameter-matched one-step denoiser.  The next advance should address selective physical retransmission or measured channels, not further tune the frozen test protocol.
