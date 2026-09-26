# Delivery status — 2026-09-26

**Frozen-model diagnostics and manuscript revision complete. The current
system is not established as a competitive, reliable HARQ method.**

## Completed in this revision

- Read the main text of FlowIE (CVPR 2024) and ResFlow (CVPR 2025), retrieved
  official CVF BibTeX, cross-checked arXiv metadata, and recorded reusable notes.
- Rewrote the abstract, introduction, contribution framing, experiment structure,
  and conclusion in `paper/main.tex`. Kept IEEEtran/VTC format.
- Added an exact decision-partition identity and explicit coverage, joint and
  conditional false ACK, and final-outage definitions. No new empirical results
  or reliability theorem were added.
- Built the five-page revised PDF with TeX Live, with no undefined citations
  or overfull boxes. Rendered and checked all five pages after adding Sheng results.
- Prepared `scripts/audit_ack_risk.py`. Five new unit tests and four existing
  accounting tests pass locally. Full diagnostic execution completed on Sheng.
- Corrected the evidence boundary at the top of `findings.md`, preserving the
  historical entries and unrelated dirty work. Added all new result boundaries.

## Sheng access

SSH alias `sheng` (139.184.101.105) and alternate configured address
100.100.44.15 timed out. The user-provided 100.94.183.27 reaches Tailscale SSH,
which initially required a human identity check and subsequently succeeded.
No authentication bypass occurred. The runs used an isolated directory:
`/home/sheng/flowmatching_revision_20260926`, on the RTX 4090.

## Completed executions

The risk diagnostic used all six archived test CSVs for receiver 2030, three
channel streams/dataset, and calibration-only margin selection. All source
hashes match. Its exact invocation is recorded in the output manifest.

The published SwinJSCC bridge completed Kodak (3456 rows / 40.36 s) and DIV2K
(11520 rows / 137.98 s), excluding initialization. These are run wall times,
not per-policy latency measurements. One-image smoke outputs are excluded.
An initial import failed because the frozen metrics module predates newer
metric functions; lazy metric imports fixed this before inference. No model
parameters were changed. No training or inference jobs remain running.

Protocol commits `3f96c43` and `47cf94f` precede execution. Raw CSVs, selected
reconstructions, inventories, logs, environment locks and source/checkpoint
hashes are in `artifacts/sheng_completion_20260926/`. The diagnostic input
archive supplies every used CSV and calibration JSON. Model binaries remain
at recorded server paths; manifests are not public checkpoints.

The bridge used `artifacts/spring_final_20260921/runtime/source/` plus the two
bridge scripts. It must not be run against unrelated dirty root modules and
called identical. Validation checks input hashes, 14,976 rows, shared channel
gains, common fallback equality, cost and PSNR/MSE identities, and reconstruction
hashes. Results are post-hoc diagnostics, not an untouched confirmation.

## Scientific decision

On Kodak, the author two-round Chase wrapper improves all four mean quality
metrics at lower charged use. On DIV2K, mean MSE is worse despite higher mean
PSNR. The wrapper is our adaptation, not a faithful published HARQ method.
The calibrated margin still leaves 30.32% conditional false ACK on Kodak.
These results prohibit a reliable-delivery or system-SOTA claim.

The next substantive choice is whether to migrate repair to the published
pretrained codec and retrain/calibrate under a new protocol, or retain this
system as a bounded empirical study. This choice was presented to the user;
no migration or new training has started.

The new paper has 15 citations and remains a draft. It is not submission-ready
evidence of reliable HARQ or SOTA. Anonymous author fields and human submission
checks remain unresolved. No submission has been performed.
