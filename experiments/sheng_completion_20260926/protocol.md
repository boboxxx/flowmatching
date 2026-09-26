# Sheng completion protocol — 2026-09-26

Status: **not executed**. Sheng requires a human Tailscale SSH identity check.
The current paper remains a VTC manuscript; CVPR is a reference for argument and
experimental structure, not a change of venue or permission to claim SOTA.

## Question and scope

Does frozen receiver repair save charged channel uses without increasing false
acceptance at a comparable operating point? Existing mean PSNR/LPIPS tolerances
do not answer this. Existing DIV2K/Kodak outcomes are already known, so this is a
**post-hoc diagnostic**, not a newly untouched confirmatory evaluation.

Do not train additional seeds. Preserve H2 and the failed H3/Evidence-HARQ gates.
Do not tune on Kodak, select a favorable SNR, or replace published models with
same-backbone adapters bearing published names.

## Stage A: frozen decision diagnostic (no GPU required)

Use the first existing receiver (training seed 2030), its archived calibration
CSV and decision JSON, and all three original channel CSVs on each dataset.
Recompute scores from the raw prediction plus the recorded calibration bias.
Pair direct, repaired, and common-MRC results by image/SNR/speed/channel seed.
Verify unique rows, disjoint calibration/test image names, fixed K=4/mask=0.8,
finite metrics, and exactly matching physical observations.

Evaluate every predeclared additional safety margin {0, .25, .5, 1, 2, 4} dB
plus always-NACK. On calibration only, choose the smallest margin for each
policy with empirical conditional false ACK <=5% and ACK coverage >=5%.
If none passes, select always-NACK and record `infeasible_nontrivial_policy`.
Never reinterpret undefined conditional risk (zero ACKs) as measured zero risk.
Report the complete test curves, not only the selected point. Selection is
empirical: 20 calibration images with repeated channel conditions do not justify
independent-binomial confidence bounds or a distribution-free risk guarantee.

Report PSNR, MSE (computed per row from PSNR where not archived), LPIPS,
single-scale SSIM explicitly labeled, NACK, ACK coverage, joint false ACK,
conditional false ACK, final outage, and charged uses at eta=0.5. Do not invent
MS-SSIM from SSIM: the existing independent binary16 audit supplies MS-SSIM.

## Stage B: published-baseline comparison (requires verified Sheng access)

First inspect actual GPUs, jobs, author checkpoint hashes, and environment.
The existing SwinJSCC, NTSCC and CDDM author checkpoints are published-model
references; DeepJSCC-f is an author-code retraining with richer feedback.
Their current AWGN numbers are **not** matched Rayleigh HARQ baselines.

Before starting inference, freeze a separate implementation-level protocol for
one matched-channel bridge: published SwinJSCC C32 with one-shot and identical-
payload two-round Chase/MRC, explicit scale signaling, same crop, Rayleigh law,
SNR grid and four metrics. Call the HARQ wrapper an adaptation, not the original
paper's published HARQ method. Checkpoint training-channel mismatch must remain
visible. If attaching the FlowHARQ receiver to that codec requires training,
that is a separate experiment, not a checkpoint-compatible assumption.

A faithful external HARQ comparison must retain DeepJSCC-f's actual feedback
and rate costs or use an independently verified published binary-feedback
implementation. Until this is implemented and run, no matched external-HARQ
superiority claim is allowed. CVPR image restoration models are related work,
not wireless HARQ baselines.

## Completion gate

The project is not scientifically complete until Stage B is resolved and the
decision-risk result is known. A negative result is acceptable; hiding it is
not. Every completed run needs raw rows, hashes, exact invocation, source
revision, environment, image manifest, checkpoint identity, and completion
status. Manuscript claims must follow measured results. Human author details,
scientific approval and conference submission remain author actions.
