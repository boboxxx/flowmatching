# Evidence-adaptive JSCC: preregistered, gated research protocol

Status: implementation and feasibility phase. No new-method performance claim.
Date: 2026-09-20. Training seed: 2027. Independent channel draws are NOT training seeds.
The supplied design is a research hypothesis, not experimental evidence.

## Research question and scope

Can a receiver use conditional generative uncertainty to request useful new evidence,
reducing total charged channel uses at a matched distortion/perceptual fidelity and outage?
Primary domain is wireless RGB image transmission; no task-semantic claim is made without
a separately specified downstream semantic experiment. LPIPS is a paired perceptual
distortion measure, not a distributional perception guarantee. FID on 24 images is invalid.

## Terminals, information, and cost

Source x has N=3HW real values. The pretrained, frozen SwinJSCC encoder produces
z0=E0(x) in R^(32x16x16). A new transmitter-only network produces e=E1(x,z0)
with the same dimensions. Four fixed spatial quadrants form enhancement packets.
The transmitter does NOT see the channel output. Its only receiver information is a
charged binary ACK/request flag and a four-bit group mask. E1 is not named DeepJSCC-l
or SAFG-HARQ: it is our codec, and selection controls on it are internal ablations.

For each real packet u of length 2n, a=sqrt(2 mean(u²)); the transmitted complex
symbols are pairs of u/a. Thus sum |s_i|²/n=1. Independent circular complex
Gaussian noise has variance 10^(-SNR/10); each real component has half that variance.
The receiver reconstructs real features from y using ONLY Q16(a), a 16-bit IEEE
binary16 scalar transmitted as metadata. It never receives a clean per-token RMS.
No unquantized scale is passed across the receiver API. E0/D0 remain frozen.

The first implementation is AWGN, 10 dB training; 0,5,15 dB robustness is evaluated
without retraining. Fading with actual noisy pilots/CSI estimation is a later mandatory
extension before a vehicular-channel claim. AWGN does not require a fading pilot.

Costs for a selected set S are:

    n_payload = 4096 + 1024 |S|
    n_scale = ceil(16(1+|S|)/eta_side)
    n_feedback = ceil((5 + 1[|S|>0])/eta_fb)
    CBR_total = (n_payload+n_scale+n_feedback)/(3*256*256).

Primary eta_side=eta_fb=0.5 information bits per complex channel use, explicitly
assuming error-free coded metadata/control. This is a cost model, NOT a simulated
physical code. Report forward payload and each control component separately;
vary eta in {0.25,0.5,1} for overhead sensitivity. Shared image size, grouping and
codebook/model configuration are fixed a priori. No capacity conversion is used to
hide feedback or scales. Base-only and full-enhancement controls use the same accounting.
The final ACK after a requested transmission is charged. A two-round latency limit holds.

## Progressive codec and the feasibility gate

Stage A trains E1 and latent fusion F(y0,yS,mask,SNR) using random nonempty subsets
(25% additional all-groups examples). Decoder D0 is the unchanged author network.
The empty subset returns D0(y0) exactly. Missing packets are masked BEFORE fusion.
MSE is the preregistered training loss; PSNR/MSE/LPIPS/MS-SSIM are all evaluated.
This initial latent-fusion construction cannot be assumed to exceed the representational
ceiling of D0; the gate measures that limitation rather than hiding it with another loss.

DIV2K training 800 images: training only. Official validation images sorted by name:
first 20 tuning/early stopping; next 40 oracle feasibility gate; last 40 held-out
risk-calibration pilot. Kodak24: final external-domain test, never tuning/calibration.
Each split is hashed and checked for content overlap. The published base checkpoint
was trained by its authors on DIV2K; inherited pretraining is disclosed.
The same gate/calibration sets may NOT repeatedly tune a revised codec: a failure
requires a newly declared study, with previous gates retained as development data.

Stage A uses one seed, up to 2000 steps, batch 8, AdamW, 2e-4 peak LR, warmup 50,
cosine decay, six validations patience after at least 600 steps. Early stopping uses
ONLY average MSE over fixed prefix sizes 1..4 on the tuning split. Validation channel
draws are fixed, training RNG preserved; checkpoint stores optimizer and RNG states.
Worker-prefetch state is not restored, so resuming is not claimed bitwise identical.

Before posterior training, evaluate all 16 subsets with identical first-round observations
and common per-group future noises. This produces an exact **clairvoyant finite-action
upper bound**, which knows the source AND future enhancement noises; it is not a
deployable policy, nor an expected-value-of-information oracle. Its failure to reach
10% saving at matched service excludes this candidate codec. Passing only permits
the next experiment and does not establish that causal policies can reach the bound.

Preregistered joint service: PSNR >=24dB AND LPIPS <=0.30. Also report threshold
sensitivity P in {22,24,26}, L in {0.25,0.30,0.35} without choosing the winner on test.
The reference oracle chooses between base and all four groups with ground-truth
service evaluation. Selective oracle chooses the cheapest passing subset and falls
back to all groups when none pass. Include failed images in total costs and outage.
Primary feasibility requires >=10% savings vs that *adaptive* base/full reference,
no higher outage, >=20% passing reference images, no >0.1dB mean PSNR degradation,
no >0.005 mean LPIPS degradation, and no >0.005 MS-SSIM degradation. Report all
numbers, including comparison to always-full; do not infer matched quality just from
equal mean PSNR. If codec refinement is not consistently useful, record a failed gate.

## Conditional posterior and acquisition (only after the gate passes)

Train q_theta(e | c) with c=(y0,SNR,Q16(a0)). For a source/channel draw, fix c,
sample independent epsilon~N(0,I), t~U[0,1], and u_t=(1-t)epsilon+t*e_standardized.
Regress v_theta(u_t,t,c) to e_standardized-epsilon. The SAME c is retained along
the whole flow path and at inference; training endpoint e is never exposed to the
receiver. This defines conditional flow matching, not a proof of posterior calibration.
Samples differ by independent initial noise with fixed evidence. No ground-truth
mask is used at inference. Standardization constants are fitted to training latents only.

Compute empirical block trace covariance with the unbiased M-1 denominator; initial
M=8, sensitivity M=4/16 and integration steps K=4/8/16. Numerical refinement K is
not additional evidence and must not be asserted to reduce epistemic uncertainty.
Check coverage, variance collapse, rank correlation and AUROC against held-out errors
and marginal utility; positive correlation alone does not certify a posterior.
No active policy advances if uncertainty is uninformative (preregistered AUROC <0.65
for above-training-median block error or nonpositive utility rank correlation).

Bayes risk R(c)=min_a E[d(x,a)|c]. An acquisition score is
V_j(c)=[R(c)-E_{Yj|c}R(c,Yj)]/C_j. Variance is a proxy, NOT equal to V_j for a
nonlinear decoder/perceptual metric. A learned value head must be trained on disjoint
source examples with prospective channel draws and evaluated against simple predictors.
Receiver actions depend only on c and received packet history, never x or future noise.
Compare fixed/random groups, channel-SNR proxy, deterministic residual/utility predictor,
posterior variance, learned utility, and the labelled clairvoyant bound. The common
codec controls isolate policy gain; external published systems establish competitiveness.

## ACK reliability and honest statistical unit

F=1[PSNR<24 or LPIPS>0.30]. Report final outage P(F), false-ACK joint probability
P(F and ACK), conditional false-ACK P(F|ACK), and ACK coverage separately.
A calibration guarantee is marginal over a defined distribution, not P(F|y)<=alpha
for every y. Calibration/threshold search must be independent of model/policy fitting.
Freeze a finite threshold grid before calibration; simultaneous one-sided binomial
bounds with Bonferroni correction may certify conditional ACK risk for iid images,
using ONE preregistered noise draw per image. Repeated noises/crops of the same image
must not inflate calibration n. On insufficient data, abstain and report no certificate.
Forty images cannot certify 5% risk at 95% confidence even with zero failures at one
threshold. An adequately sized independent calibration corpus is therefore required
for a paper-level 5% conditional-ACK guarantee. No guarantee transfers automatically
from DIV2K to Kodak or from AWGN to fading.

## Metrics, baselines, and reporting

All images: RGB float [0,1], center256 primary; per-image MSE and PSNR then average,
AlexNet LPIPS v0.1, standard five-scale MS-SSIM. Report CBR frontiers, failure vs CBR,
latency (GPU synchronized, warmup), sample count, flow evaluations M*K, and peak memory.
No training-seed sweep or mean±std tables by default. Raw per-image/channel CSVs and
image identifiers enable paired auditing. Monte Carlo draws are listed explicitly.
See baseline_registry.json for publication and reproducibility eligibility. Existing
author-protocol results with free scales/ideal channel-output feedback are descriptive
references only until the differing costs/protocols are reconciled. Invalid DeepJSCC-f
V1 remains invalid. No manuscript superiority claim before all mandatory comparisons.
