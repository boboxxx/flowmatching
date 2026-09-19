# H4 protocol — parameter-matched one-shot residual control

Status: locked before execution. Classification: mechanism ablation.

## Question

Does uniform-time, multi-step RAFM provide value beyond a generic receiver residual denoiser?

## Control

Use the exact H1 architecture, parameter count, frozen SwinJSCC backbone, normalized-Huber target, reconstruction weight 1.0, mask supervision, data, optimizer, 40 epochs, and training seed 2030. Change only the path-time distribution from uniform $t\sim U[0,1]$ to fixed $t=0$, making the network a direct received-to-clean residual predictor. At inference the control uses one Euler/residual step. Calibration compares tau in {0.8,0.9}; the 80-image split remains untouched until the control is selected.

## Interpretation

RAFM is supported as more than a denoiser only if the frozen H1 K=4 calibration PSNR gain exceeds the selected one-shot control by at least 0.02 dB without an LPIPS disadvantage greater than 0.001. Otherwise the paper must soften the flow-specific claim and present the receiver repair principle as the primary contribution.
