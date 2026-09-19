# H2 external validation — Kodak24

Status: locked before execution. Classification: external confirmatory validation.

Use all 24 images of the Kodak Lossless True Color Image Suite at the existing deterministic 256x256 center-crop evaluation transform. No Kodak image is used for training, model selection, mask/step selection, service-target selection, or scalar calibration.

Evaluate the three frozen H2 training seeds with fresh channel seeds 9001--9003, the same SNR/speed grid, K=4, tau=0.8, and the DIV2K-calibrated 24 dB decision rule. Report paired Student-t intervals over training-seed means. Success requires the retransmission-saving direction to replicate while PSNR remains >= -0.02 dB and LPIPS <= +0.003 relative to adaptive HARQ. The dataset snapshot is Git commit `dd2e1105fe8d9afe1120cd7e4c0e5e41f509a81b` with 24 valid 768x512 RGB PNG files.
