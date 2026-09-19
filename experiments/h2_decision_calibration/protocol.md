# H2 protocol — target-aware HARQ decision

Status: preregistered, pending H1. Classification: confirmatory.

## Hypothesis

Global log-MSE regression wastes capacity away from the ACK boundary. Adding a target-aware binary/margin loss for whether post-flow PSNR meets the service target will reduce boundary error and convert repair gains into fewer NACKs.

## Primary metric

Calibration balanced accuracy and expected calibration error at the frozen service threshold, followed by held-out constrained NACK saving. The quality tolerances remain PSNR >= -0.02 dB and LPIPS <= +0.003 relative to adaptive HARQ.

## Guardrail

The test split cannot select the loss weight or scalar calibration. If H1 produces no repair gain, H2 is not sufficient on its own and will not be used to manufacture a result through miscalibration.

