# H1 protocol — robust, reconstruction-aligned flow repair

Status: locked before execution. Classification: confirmatory.

## Hypothesis

Squared absolute velocity error is dominated by rare ZF deep-fade residuals. A robust, per-token scale-normalized velocity loss plus a stronger image reconstruction weight will increase true post-flow reconstruction quality.

## Changes

Compare the v2 objective against three isolated variants using training seed 2030 and the frozen backbone:

1. `rec1`: increase `lambda_reconstruction` from 0.2 to 1.0.
2. `huber`: replace squared velocity error with smooth-L1 after normalizing each token residual by its detached RMS scale.
3. `huber_rec1`: combine both changes.

Everything else remains fixed. Calibration uses the existing 20-image split; the 80-image test split is touched only after choosing among variants.

The conservative inference grid is fixed before results to $K\in\{1,2,4\}$ and $\tau\in\{0.8,0.9\}$. This grid excludes the previously observed aggressive over-repair region and lets the hypothesis test the training objective rather than one arbitrary inference setting.

After calibration selection, the single-seed held-out replication uses three new channel seeds (7001--7003). It is a gate before training seeds 2031 and 2032: the multi-seed expansion proceeds only if the held-out FM-only PSNR delta remains positive and FlowHARQ preserves the matched-quality constraints.

## Prediction and acceptance

- Calibration FM-only PSNR must improve by at least 0.05 dB over direct at a feasible conservative mask.
- LPIPS must not worsen by more than 0.003.
- The winning variant must improve the constrained NACK saving over v2 before expanding to three seeds.

## Failure interpretation

If no variant reaches the FM-only quality criterion, absolute residual fitting is not the main bottleneck; pivot to spatial conditioning or corruption-localized training rather than more loss-weight sweeps.
