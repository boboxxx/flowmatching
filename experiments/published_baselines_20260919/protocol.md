# Published-paper baseline replacement: frozen protocol v1

Date: 2026-09-19. This file separates completed author-checkpoint evaluation from
new training and from the project's older internal ablations. It is not a claim
that the proposed method beats published systems.

## 1. Selection grounded in published papers and author implementations

| Baseline | Publication | Role | Implementation evidence |
|---|---|---|---|
| SwinJSCC, base, without SA/RA | Yang et al., IEEE TCCN 11(1), 90–104, 2025; online 2024; [DOI](https://doi.org/10.1109/TCCN.2024.3424842) | Strong fixed-rate Transformer JSCC; separate MSE and MS-SSIM trained models | [Author code](https://github.com/semcomm/SwinJSCC), author public weights; commit `a6d0e6da53548976acbe9317839a077ef31f190f` |
| NTSCC w/o z | Dai et al., IEEE JSAC 40(8), 2300–2316, 2022; [DOI](https://doi.org/10.1109/JSAC.2022.3180802) | Content-adaptive rate–distortion baseline | [Author code](https://github.com/wsxtyrdd/NTSCC_JSAC22), public MSE weights, commit `dd3c745374023af6d01f465c379122be785d936f` |
| DeepJSCC-f | Kurka and Gündüz, IEEE JSAIT 1(1), 178–193, 2020; [DOI](https://doi.org/10.1109/JSAIT.2020.2987203) | Published feedback/progressive transmission reference | [Author code](https://github.com/kurka/deepJSCC-feedback), commit `61a853baf019d589a7be09dc443c72985b891a74`; train original architecture with documented data/runtime adaptation |

Read sources: [SwinJSCC manuscript](https://arxiv.org/html/2308.09361v2),
[NTSCC manuscript](https://arxiv.org/abs/2112.10961),
[DeepJSCC-f manuscript](https://arxiv.org/abs/1911.11174), especially their system
models, losses, channel-use accounting, experimental setups and feedback assumptions.
Publication metadata were cross-checked against publisher/DOI and author sources.

SwinJSCC evaluates separate distortion-trained models; its native high-resolution
benchmark motivates full-resolution Kodak evaluation. NTSCC evaluates quality versus
actual channel bandwidth, including a model for rate signalling. DeepJSCC-f trains
progressive layers sequentially, freezes earlier layers, and uses channel-output
feedback and reconstruction fusion. Those are essential mechanisms, not names
that can be attached to a generic residual adapter.

Other candidates: [DJSCC-H, IEICE 2026](https://doi.org/10.1587/transfun.2025EAL2052)
and [semantic-base fine-grained HARQ, IEEE TWC 2025](https://doi.org/10.1109/TWC.2025.3532501)
are relevant published HARQ work. They are not executable baselines in this release:
an official reproducible implementation/checkpoint has not been established.
Do not relabel our threshold policy or residual adapter as either method.
Preprints without confirmed publication are excluded from the requested main baseline set.

## 2. What is NOT an external baseline

`direct`, `full_harq`, `fm_only`, `adaptive_harq`, `flowharq`, and oracle variants
share this project's network and are internal controls/ablations.
The pre-existing `baselines/deepjscc_f` adapter is an implementation-inspired control,
not a reproduction of the published DeepJSCC-f architecture. Earlier results remain
available for audit but must not be used to claim superiority over DeepJSCC-f.

## 3. Two tracks that must not be silently mixed

### A. Author-checkpoint verification (running/completed artifacts)

- AWGN at 10 dB, matching released fixed-SNR checkpoints.
- Kodak24, all 24 images: full resolution primary; 256×256 center crop secondary.
- No resize, no test-time parameter search, no choosing favorable images.
- SwinJSCC C=32 and C=64, both MSE and MS-SSIM objectives: four public checkpoints.
- NTSCC public quality 1 and 2, MSE, `use_side_info=False`, author default eta=0.2.
  This is the released **NTSCC w/o z** variant, not the full hyperprior-refined model.
- Ten independent channel draws per image, a fixed noise-stream seed 7070.
  This averages channel randomness, not ten training seeds; no across-training-seed
  standard deviation is invented. All checkpoints and images have SHA-256 manifests.
- Supplementary SNR mismatch test: the same frozen 10-dB Swin MSE C32 and NTSCC
  quality-1 weights at SNR 0, 5, and 15 dB, 256 center crops, ten draws. Combine
  with the 10-dB measurement only as an **off-design** curve; do not imply training
  a specialized network at each SNR or adaptive-SNR SwinJSCC.
- Native author model/channel code is used. Only derived resolution buffers can be
  regenerated during loading; any missing/unexpected learned weight is an error.
- NTSCC's byte-mask selection is changed to bool for PyTorch 2.5 compatibility.
  Architecture, weights, rate selection, channel and decoder are unchanged.
- The NTSCC authors explicitly warn that their reorganized release differs slightly
  from the published curves. Results here are measurements of that release.
- SwinJSCC base weights differ in capacity from this project's small Swin backbone.
  They establish a strong external reference, not a capacity-controlled ablation.
- Pretraining differs: NTSCC author weights use 500k OpenImages images. SwinJSCC's
  release configuration includes validation directories in training options, so
  DIV2K validation is not treated as a proven unseen set for those public weights.

### B. Fair project comparison (NOT completed by track A)

The old H2/H3 experiments use time-correlated block Rayleigh fading, receiver CSI,
per-token scale restoration, 256 crops, and adaptive retransmission. AWGN author
results cannot be merged with them into a superiority table.

Before the VTC main comparison, all methods must use one declared source split,
channel law, complex power convention, forward-channel budget, receiver CSI model,
and feedback/side-information accounting. Re-training/adaptation of published
architectures must be named as such, with author-checkpoint verification retained.
All candidate rate/threshold settings must be selected on calibration data only.
No extrapolated RD values or results copied from paper figures count as our experiments.

## 4. Metric definitions: compute from the same reconstruction

For RGB x and clipped reconstruction x_hat in [0,1], let n=3HW:

    MSE_i = ||x_i - x_hat_i||² / n
    PSNR_i = -10 log10(max(MSE_i, 1e-12))
    MS-SSIM-dB_i = -10 log10(max(1 - MS-SSIM_i, 1e-12))

Compute all metrics per image and channel draw, average draws within each image,
then average images equally. With ten draws per image the flat mean is equivalent.
`mean(PSNR_i)` is NOT `-10 log10(mean(MSE_i))`. Both PSNR and MSE are reported;
they measure related pixel fidelity and are not two independent scientific findings.

| Metric | Better | Exact shared convention |
|---|---|---|
| PSNR | higher | RGB float [0,1], no Y-only conversion, no extra border removal |
| MSE | lower | RGB mean squared error [0,1]; 8-bit-unit MSE = this value ×65025 |
| LPIPS | lower | `lpips==0.1.4`, AlexNet, learned version 0.1, RGB scaled to [-1,1], eval mode |
| MS-SSIM | higher | `pytorch-msssim==1.0.0`, five scales, 11×11 Gaussian, sigma 1.5, valid windows, weights [0.0448,0.2856,0.3001,0.2363,0.1333] |

Raw MS-SSIM is the table metric. Its dB transform may be used on RD figures, clearly
labelled; report mean per-image dB, not dB of the dataset-mean score. Identity dB is
capped at 120 for finite output. Inputs must exceed 160 pixels on both axes.

**Author-code discrepancies retained transparently:** SwinJSCC/NTSCC releases use
a four-level MS-SSIM expression that also repeats the final-scale factor in their
product; this is not the standard five-scale implementation. Save it separately as
`native_ms_ssim_4level`, never rename it `ms_ssim`. NTSCC native test PSNR rounds
images to 8-bit; save `psnr_8bit` as a cross-check. Shared main metrics use float RGB.
The MS-SSIM-trained checkpoint label denotes its actual author training objective,
not a claim it was trained with our standardized five-scale metric.

Existing single-scale SSIM is retained only for backwards compatibility; it does
not satisfy an MS-SSIM requirement. No LPIPS model is described as perceptually
optimized unless its training objective actually included perceptual loss.

## 5. Rate and HARQ system accounting

The forward channel is y=h s+w with complex unit average transmit power and
complex noise variance 10^(-SNR/10). In track A h=1. Count one complex symbol as
two real latent values:

    CBR_payload = number of transmitted complex symbols / (3HW)
    mean CBR_HARQ = E[sum_{round=1}^{T} k_round] / (3HW)

SwinJSCC high-resolution downsampling is 16 per axis:
CBR=C/(2×3×16²), giving 1/48 and 1/24 for C=32 and C=64.
DeepJSCC-f downsamples by 4: CBR per layer=c/(2×3×4²)=c/96.
For c=2, one/two rounds use 1/48 and 1/24; the c=4 one-layer control uses 1/24.

For NTSCC, count the selected symbols actually passed to its channel, not the
maximum latent width. Its 16-choice rate map costs 4 bits per 16×16 source patch.
Report payload CBR separately from the author's capacity-achieving signalling model:

    CBR_rate_map_capacity = 4 / [3×16²×log2(1+10^(SNR/10))]

At 10 dB this adds approximately 0.00150555. This is an idealized lower-bound
channel-use model, NOT an implemented reliable LDPC link. The w/o-z release sends
no hyperprior z to the receiver; do not charge z twice or claim refinement with it.

SwinJSCC and NTSCC author channels restore a global latent scale at the receiver;
its transmission cost is not represented in the public model. The existing project
restores per-token scales, a stronger unpriced side-information assumption. These
are disclosed, not silently treated as equal overhead. A deployable comparison must
remove decoder-inaccessible scales or explicitly encode and count them.

DeepJSCC-f assumes ideal channel-output feedback (many real values), not just an
ACK bit. Label its feedback assumption and show forward/feedback cost separately.
For adaptive methods report quality versus expected total CBR, average rounds,
P(PSNR < target), and latency only when measured consistently. An oracle stop rule
using the true source image is an upper bound, not a receiver-deployable policy.

## 6. DeepJSCC-f retraining

The unmodified GPL author implementation is loaded from a separate upstream checkout.
It retains SignalConv2D/GDN CNNs, previous-reconstruction encoder conditioning,
accumulated channel-output decoding, and image fusion. Earlier layers freeze before
training the next layer. The original small Swin adapter is not involved.

- One training seed: 2027. DIV2K train 800 images, random 128 crops and horizontal flip.
- Validation: first 20 DIV2K validation images, center 128 crop, five channel draws.
  Kodak and the other 80 DIV2K validation images do not select checkpoints.
- AWGN 10 dB, MSE loss, Adam 1e-4, batch 8, 100 steps/epoch.
- Maximum 100 epochs/stage; early stop after 3 validation checks with less than
  0.01 dB improvement. Select the best validation checkpoint, not the last epoch.
- Train c=2/L=2 progressively; c=4/L=1 is a same-total-forward-budget control.
- Adaptations: DIV2K instead of the original high-resolution ImageNet training;
  TensorFlow/TFC 2.14.1 instead of TF 1.15.2/TFC 1.3. Record this in every artifact.
- Stopping at the epoch budget is not evidence of convergence. Curves must be
  reviewed and test evaluation completed before any final comparison is published.
- Last-epoch optimizer/model checkpoints are distinct from selected-best weights.

### Completed v2 result

V2 completed the predeclared early-stop protocol and independent Kodak evaluation.
Stage 0/1 of c=2/L=2 stopped after 32/58 epochs with best validation PSNR
24.4282/27.8490 dB; c=4/L=1 stopped after 16 epochs with best validation PSNR
25.9647 dB. The selected checkpoints were never selected on Kodak. The associated
raw CSVs, per-stage histories, test summaries, status JSON, and 48 reconstruction
array checksums are versioned under `artifacts/published_deepjsccf_v2_20260919/`.

### Serialization correction: first DeepJSCC-f attempt is quarantined

The first runtime-adaptation attempt used Keras `.weights.h5`, which does not save
all TensorFlow Compression `tf.Module` kernel/GDN parameters. Reloaded Kodak scores
near 13 dB were consequently invalid, not evidence that DeepJSCC-f is weak.
The first run under `deepjsccf_author` is excluded entirely, including its nominal
best-validation selection, because that selection also used the incomplete format.

The fixed `deepjsccf_author_v2` run starts from scratch with the same seed/protocol
and legacy `.h5` serialization. A two-layer regression compares all 113 weight
tensors after save/reload: legacy format max error=0; incomplete-format negative
control max error=0.6013545394. This gate runs before training. The exporter rejects
old/unverified format metadata. V2 training is followed automatically by Kodak
256-crop export and standardized four-metric scoring, without test-set selection.

## 7. Reproduction and remaining gates

Server: `sheng@100.94.183.27`, project `/home/sheng/flowmatching_vtc2027`.
GPU: RTX 4090. Data/runtime/output paths can be overridden through script options or
documented environment variables; no dependency is installed into the existing cv env.

Entrypoints:

- `scripts/download_published_weights.py`: public author IDs, hashes and objective labels.
- `scripts/run_author_checkpoint_benchmark.sh`: author weights, all four metrics.
- `scripts/run_author_deepjsccf_training.sh`: two-stage and equal-budget control training.
- `scripts/export_author_deepjsccf.py` then `scripts/score_author_deepjsccf.py`:
  save float reconstructions, then use the same four-metric implementation in a
  separate process (TF/cuDNN8 must not be mixed with PyTorch/cuDNN9 in one process).
- `scripts/run_published_metric_audit.sh`: old H2/H3 four-metric audit, NOT external baselines.

Submission gates still open: matched-channel/source training, all side-information
and feedback costs, additional RD operating points, a matched-SNR sweep, and a genuine digital source-code + channel-code
reference if claiming superiority over separation-based communications. The current
author verification is complete as an AWGN reference, not a completed VTC
performance claim.
