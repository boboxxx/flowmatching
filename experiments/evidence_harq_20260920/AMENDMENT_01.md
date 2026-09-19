# Training feasibility amendment, before posterior training or Kodak testing

Initial candidate A stopped at step 900 (six tuning validations without improvement).
Its four-group reconstruction did not improve the base. Exact oracle savings were zero.
This is a failed *training/codec feasibility candidate*, not evidence that active evidence
acquisition is impossible. Both the original protocol and its negative results are retained.

A fixed four-image training-batch diagnostic, with FP32 and LR 1e-3, reduced MSE
from 0.00165975 to 0.00117368 after 200 updates (base MSE 0.00165628).
Enhancement encoder gradients were nonzero. This excludes a completely disconnected
encoder, and motivates investigating optimization before abandoning the codec.
These are training-set numbers, not test gains, and fixed noise may be overfit.

Candidate B is preregistered here: same architecture, training seed, training images,
and cost model; FP32, peak LR 1e-3, up to 8000 steps, minimum 3000, validate every
200 updates, patience 10. Warmup remains 50 steps. No overfit diagnostic weights are
used. Same author checkpoint, fresh enhancement initialization. Original gate thresholds
and quality tolerances remain unchanged. No posterior training before codec feasibility.

The 40-image oracle set has now been inspected and is explicitly a DEVELOPMENT gate.
Neither a repeated pass nor a repeated failure is confirmatory generalization evidence.
The risk-calibration split (40 distinct images) and Kodak24 remain untouched. Final
cross-dataset testing is performed only after freezing the complete selected policy.
If B also fails, stop this construction and report the negative evidence; do not keep
increasing budgets or adjusting gate tolerances until a pass appears.
