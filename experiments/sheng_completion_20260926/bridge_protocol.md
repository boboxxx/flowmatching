# Published-checkpoint channel bridge — fixed before inference

Date: 2026-09-26. No training or calibration. Diagnostic, not a new confirmatory
test set: DIV2K validation images 21–100 and Kodak24 have already been examined.

Compare published SwinJSCC MSE C32 author weights with frozen FlowHARQ receiver
2030. Use identical image hashes/crops, one fixed channel stream (base integer
9262026), SNR {0,3,6,9,12,15}, speeds {30,60,90,120}, correlated flat Rayleigh
fading, 5.9 GHz, 1 ms round gap and perfect CSI. Each case has the same two complex
gains and 4096-element noise vectors across codecs. No added training seeds.

SwinJSCC retains author real/imaginary half-vector packing and one global
packet scale; FlowHARQ retains adjacent real/imaginary pairing and per-token
scales. Both send 4096 complex symbols per round with unit mean complex-symbol
power. Restore binary16 scales at the receiver, transmitted once at 0.5
information bits/use; count 2 uses per feedback message. Do not hide the
different scale cost or training recipe. Use one-shot and always-two-round
Chase/MRC for both codecs; report adaptive/no-repair and FlowHARQ policies only
for the existing receiver. SwinJSCC Chase is our wrapper, **not a published
HARQ architecture**. The author checkpoint was not retrained for this fading
law, so report transfer as such; no SOTA or fair-training claim.

Measure per-image PSNR, MSE, five-scale MS-SSIM, AlexNet LPIPS, target outage,
actual NACK and charged uses. Clamp decoded RGB to [0,1], then compute metrics.
Record 6 methods x 24 conditions x (80+24) images = 14,976 rows. One draw/case
is a bounded diagnostic, not a robustness claim. Save deterministic examples
(first image at 0 and 15 dB, speed 60), not best-looking selections, and an
image/reconstruction hash list. Environment/source/checkpoint hashes and timing
must accompany outputs. A one-image smoke run cannot be labeled completed.

Sanity checks: checkpoint strict trainable-weight load, 8192 real latent
dimensions, normalized power, finite metrics, exact shared gains/noise,
MSE/PSNR consistency, complete grid, and non-overwrite outputs.

Interpretation fixed in advance: if a published one-shot codec exceeds FlowHARQ
on all four quality metrics while using fewer charged symbols, the receiver
ablation remains valid but **does not establish a competitive end-to-end
system**. Report that result in the paper; do not tune it away.
