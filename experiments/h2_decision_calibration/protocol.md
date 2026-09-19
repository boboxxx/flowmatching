# H2 protocol — target-aware HARQ decision

Status: active after H1 repair success. Classification: confirmatory mechanism follow-up.

## Hypothesis

Global log-MSE regression wastes capacity away from the ACK boundary. Adding a target-aware binary/margin loss for whether post-flow PSNR meets the service target will reduce boundary error and convert repair gains into fewer NACKs.

## Primary metric

Calibration balanced accuracy and expected calibration error at the frozen service threshold, followed by held-out constrained NACK saving. The quality tolerances remain PSNR >= -0.02 dB and LPIPS <= +0.003 relative to adaptive HARQ.

## Guardrail

The test split cannot select the loss weight or scalar calibration. If H1 produces no repair gain, H2 is not sufficient on its own and will not be used to manufacture a result through miscalibration.

## Frozen intervention

- Initialize from each of the three completed H1 checkpoints and freeze encoder, decoder, reliability head, and RAFM.
- Fine-tune only the quality head for 10 epochs at learning rate $10^{-4}$.
- Use equal weights on log-MSE regression and binary NACK cross-entropy at the fixed 24 dB service boundary, with a 1 dB logit temperature.
- Keep K=4, tau=0.8, the data split, channel model, and all reconstruction modules unchanged.
- Fit only one scalar direct and post-flow bias per training seed on the 20-image calibration split.
- Evaluate the frozen decision rule on the same 80-image benchmark with fresh channel seeds 8001--8003; no H2 choice is made from these outputs.

## Acceptance and failure

Calibration balanced accuracy must improve for the post-flow decision without materially worsening ECE. The paper-level held-out criterion remains a positive retransmission saving whose 95% Student-t interval over training seeds excludes zero, with PSNR delta >= -0.02 dB and LPIPS delta <= +0.003. If the interval still crosses zero, proceed to H3 cost-aware policy training rather than tuning H2 on test results.
