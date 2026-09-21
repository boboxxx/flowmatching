# Manuscript claim ledger — 2026-09-21

## Narrative

Question: can a fixed DeepJSCC receiver spend computation before requesting an
identical second payload? Method: reliability-anchored four-step latent repair,
then a separately calibrated quality decision; ordinary physical MRC on NACK.
Evaluation: paired stopping decisions, delivered distortion, and charged resources.
This is not posterior uncertainty certification, a new progressive codec, or
selective-token HARQ.

## Supported and bounded claims

- H2's original three-training-run study and already locked ten-run extension
  are separate. Average channel draws within each training run before inference.
- Original quality tolerances are PSNR delta >= -0.02 dB and LPIPS <= +0.003.
  Positive LPIPS deltas are degradation, not improved perceptual quality.
- Both-NACK outputs are identical. The decision-cell decomposition in evidence.json
  sums exactly to the aggregate quality difference, without discarded samples.
- 1.319/1.453 percentage points of NACK reduction are not percent total-link savings.
  The primary once-per-image binary16 metadata cost gives 0.353/0.391% savings.
  Assumed reliable signaling excludes actual coding, pilots, headers and scheduling.
- Four-metric finite-precision evaluation is a new CPU robustness audit using the
  frozen seed-2030 model, not an exact replay of archived CUDA random numbers.
  Original SSIM is never renamed MS-SSIM. Failed stopping decisions must be reported.
- H4 fails its tested calibration mean-PSNR gate. It differs in training-time
  distribution and chosen threshold. Its archived bootstrap samples image-condition
  rows, not independent image clusters; that interval is not used in the manuscript.
  H4 does not isolate integration depth or prove FM uniquely necessary.
- Published SwinJSCC/NTSCC/CDDM author weights are measured under their documented
  AWGN setting. DeepJSCC-f uses author code with DIV2K/TF2 adaptation and ideal
  channel-output feedback; it is not equivalent to binary ACK/NACK. No cross-protocol
  state-of-the-art superiority is inferred.
- GPU timings of decode versus repair+decode omit the quality head. The new CPU
  policy timing is one input on shared hardware and is not a GPU or end-to-end
  latency claim. The paper uses only a conditional, symbolic break-even relation.

## Not supported

Universal perceptual gain; reliable satisfaction of 24 dB for each accepted image;
performance under imperfect CSI or erroneous control; a vehicular deployment or
measured air-interface saving; first flow matching for semantic communication;
benefits of selective retransmission; a successful H3 cross-SNR test.

The original ten-run archive gives conditional false-ACK probabilities
13.51% → 14.03% on DIV2K and 50.88% → 50.57% on Kodak24 (adaptive → FlowHARQ).
FlowHARQ's final outage below 24 dB is 70.89% and 82.92%, respectively. These
are pooled image-condition probabilities, not training-run confidence intervals.
The paper discloses these substantial service-target violations explicitly.

## Completed precision audit

Artemis CPU jobs 11403790_0 (32m03s) and 11403794_1 (9m58s) completed all
24,960 metric rows, with a frozen seed-2030 receiver and no additional training.
Binary16 changes zero adaptive/FlowHARQ stopping decisions in both datasets.
The largest individual PSNR perturbation is 0.02504158 dB. Four-metric results
confirm that modest PSNR/MSE improvements coexist with worse MS-SSIM and LPIPS.
These new CPU draws do not replace the original CUDA multi-run evidence.

## Literature checks

Crossref records verify the nine published DOI references. DataCite records verify
the four arXiv versions; raw metadata and hashes are in citation_records.json.
Recent overlap includes FlowSem (arXiv:2608.21651), channel-realization flow matching
(arXiv:2607.24876; abstract page reports GLOBECOM 2026 acceptance), and generative
semantic HARQ for text (arXiv:2603.15068). SAFG-HARQ is the published semantic-base
paper DOI 10.1109/TWC.2025.3532501, not a separately invented baseline.

## Editorial decisions

H3 failed its preregistered cross-SNR rule (3/6, required 4/6) despite favorable
aggregate means. The evidence-adaptive codec failed its feasibility gate. Both
remain available in the repository and neither is promoted into this manuscript's
validated method. No new method training was started to improve the story.
