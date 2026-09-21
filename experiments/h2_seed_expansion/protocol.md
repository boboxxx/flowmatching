# H2 seed expansion protocol — ten independent training seeds

Status: locked before execution; completed without protocol changes.
Classification: confirmatory robustness expansion.

## Purpose

The original paper result uses three independent training seeds and has a
two-sided 95% interval that only narrowly excludes zero.  This expansion adds
training seeds 2033--2039, producing ten total independent seeds (2030--2039).
It tests stability; it does not reopen model or threshold selection.

## Frozen configuration

- H1: frozen SwinJSCC backbone, normalized-Huber velocity loss,
  reconstruction weight 1.0, 40 epochs, K=4, mask threshold 0.80.
- H2: freeze encoder, decoder, reliability head, and RAFM; train only the
  quality head for 10 epochs with equal regression and 24 dB boundary losses.
- Calibration: use the same first 20 sorted DIV2K validation images and fit
  only direct/post-flow scalar biases independently for each training seed.
- Test: use the remaining 80 images and channel seeds 8001--8003.
- External validation: evaluate the same frozen checkpoints and DIV2K-fitted
  scalar biases on all Kodak24 images with channel seeds 9001--9003.
- Statistics: average channel seeds within each training seed, then calculate a
paired two-sided 95% Student-t interval over the ten training-seed means.

## Acceptance

The expanded claim is supported only if FlowHARQ minus adaptive HARQ retains a
negative retransmission interval while PSNR is at least -0.02 dB and LPIPS is
at most +0.003.  The original three-seed result remains reported separately;
the expansion may not be used to tune the frozen decision rule.

## Outcome

The gate passed on both DIV2K and Kodak24.  Full intervals and provenance are
reported in `RESULTS.md` and the two `analysis_10seeds` result directories.
