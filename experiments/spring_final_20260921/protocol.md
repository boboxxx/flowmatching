# Frozen FlowHARQ manuscript completion on Artemis

Declared 2026-09-21, before running the new finite-precision audit.

Scope: the Spring manuscript studies receiver computation before a full-payload
HARQ decision. Existing H2 and H4 weights, calibrations and test protocols remain
fixed. No new model training or seed search is part of this completion. The original
three-training-seed study and its already completed ten-seed extension are reported
as distinct evidence. H3 and the failed evidence-codec study are exploratory work.

## Total-link accounting

For 256 x 256 RGB inputs, 256 tokens of 32 real values yield B=4096 complex payload
symbols per round. The original simulator normalizes each token with a source-dependent
scale. Transport each of the 256 scales as IEEE binary16, once per source image; an
identical Chase retransmission reuses these scales. Let metadata efficiency be eta_s
and feedback efficiency eta_f, in information bits per complex channel use. For NACK a:

    C(a) = B(1+a) + ceil(4096/eta_s) + (1+a)ceil(1/eta_f).

One initial ACK/NACK and, if round two occurs, one final ACK are charged. The model
assumes reliable coded metadata/control, with no specific FEC, pilots, headers,
scheduling or idle-resource implementation. Thus these are charged symbol-equivalent
costs, not a measured complete protocol stack. Primary eta_s=eta_f=0.5; sensitivity
uses 0.25, 0.5, 1, 2, and 4. Also report payload-only and repeated-scale costs.

Each physically separate one-bit control message is rounded up separately in
the implemented accounting. This clarifies the initial shorthand ceiling over
both bits; it does not change the primary eta=0.5 calculation.

First compute this accounting sensitivity using all existing H2 raw decisions.
Label it an analytical re-accounting of the original ideal-scale decisions. It does
not by itself establish quality under quantized metadata.

## New paired finite-precision check

Use only frozen training seed 2030, original channel seed 8001 for DIV2K and 9001
for Kodak, batch size 4, original image order, six SNRs and four speeds. Reuse
the exact two original channel observations across full-precision and binary16
restoration. Compute quantized observations by algebraically replacing the original
receiver scale with its binary16 value. Do not refit either quality bias or mask.
Evaluate direct, full HARQ, FM-only, adaptive HARQ and FlowHARQ on all 80 DIV2K test
images and 24 Kodak images. RGB PSNR/MSE, LPIPS-AlexNet v0.1, standard five-scale
MS-SSIM, NACK, outage and joint/conditional false ACK are recorded per case.

Report the finite-precision result regardless of direction. It is a frozen-checkpoint
robustness audit, not a new confirmatory seed study. Per-condition SNR/speed inspection
does not authorize operating-point selection. Compare its full-precision branch to
the archived same-seed CSV; a mismatch must be explained before claiming replay.

## Mechanism and computation cost

Decompose existing paired decisions into both ACK, saved second round, newly requested
second round and both NACK. Report their frequencies and delivered-quality contributions.
Identity of the full-MRC branch should make the both-NACK quality difference zero.
This distinguishes repaired ACK outputs from changes in stopping behavior.

The existing 3.11/5.00 ms measurements time decoding versus repair-plus-decoding,
not a complete policy including quality heads. A separate frozen-policy benchmark
times the two full receiver policies, without encoding/radio airtime, on the GPU
allocated by Artemis. Any break-even radio-round time must use that measured policy
overhead and the correct retransmission probability difference. No end-to-end latency
benefit is asserted without a specified radio schedule.

## Paper evidence policy

H4 is a parameter-matched recipe control on calibration data, with a different training
time distribution and selected mask threshold; it does not prove FM is uniquely
necessary or isolate ODE integration depth. SSIM in the old study is not MS-SSIM.
Published author baselines (SwinJSCC, NTSCC, CDDM, DeepJSCC-f) retain their documented
AWGN/pretraining/feedback assumptions; no cross-protocol superiority is inferred.
The original H2 tolerances remain PSNR delta >= -0.02 dB and LPIPS <= +0.003.
The paper must explicitly report positive LPIPS deltas as degradation within tolerance.

## Execution amendment (before audit results; 2026-09-21)

The GPU allocation remained pending (job 11403772), with the scheduler moving its
estimated start to the following afternoon. The same frozen-model audit may run
on allocated Artemis CPU nodes with eight threads and two independent dataset
jobs. CPU random-number streams differ from CUDA: ideal/binary16 observations
remain exactly paired within this new audit, but comparison against archived GPU
rows is a diagnostic, not a bit-exact replication requirement. CPU timing will
not be presented as GPU latency. No model, threshold, image, SNR, or speed is
selected using this audit. GPU and CPU result directories remain distinct.
