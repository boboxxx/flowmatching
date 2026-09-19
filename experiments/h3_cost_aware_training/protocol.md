# H3 protocol — differentiable rate-quality training

Status: preregistered, pending H1/H2. Classification: confirmatory.

## Hypothesis

Training a soft ACK/NACK policy with a retransmission penalty and a differentiable mixture of repaired versus MRC-decoded outputs will move the rate-quality frontier outward.

## Objective

Add `lambda_retx * mean(p_nack)` to a final reconstruction objective, with `p_nack` produced by a temperature-controlled quality margin. Simulate the correlated second round during training and use the same CSI-weighted MRC as evaluation.

## Acceptance

Across three training seeds, physical retransmission saving must exceed 2 percentage points with a 95% CI excluding zero, while PSNR and LPIPS confidence intervals remain within the predeclared matched-quality tolerances.

## Simplicity control

Compare against a target-aware adaptive HARQ policy without flow. Any saving due only to better calibration is not evidence for virtual retransmission.
