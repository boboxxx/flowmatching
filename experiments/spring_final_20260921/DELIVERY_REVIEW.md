# Delivery review — 2026-09-22 (protocol fixed 2026-09-21)

The frozen-model audit completed on Artemis, using allocated CPU resources while
GPUs were unavailable. DIV2K: 11403790_0 / job 11403791, 32m03s. Kodak: 11403794_1,
9m58s. The failed initial Kodak path attempt and canceled pending GPU request are
preserved in execution logs. No new model training or seed selection occurred.

The complete statistical builder and artifact validator ran successfully both locally
and on Artemis. All 60 original H2 CSVs (524,160 rows), ten calibration decisions,
H1/H4 evidence, and new audit outputs (24,960 rows) are accounted for. The model-free
accounting tests pass (4 tests). Ten checkpoint and ten calibration hashes match
the prior inventory. These checks establish internal consistency, not paper acceptance.

The manuscript uses IEEEtran conference format, five US-letter pages, two figures,
three tables, and thirteen provider-verified citations. TeX Live compiled it without
undefined citations, overfull boxes, or BibTeX warnings. All five pages were rendered
and visually inspected at 120 dpi; the long chart axis label was shortened to avoid
clipping. No font or margin reduction was used to force the page count.

- Manuscript SHA-256: `f897212ae783f44a91c37d2973f8963447adf2877b41e0c0c1b3debfd023fb34`
- Delivered PDF SHA-256: `ac5fbe11ce6dd6d9bc4bd9f5a5a4efd80aaa2ec2a7f89d89a24253fb7efce16e`
- PDF: `output/pdf/FlowHARQ_VTC2027.pdf`

Principal evidence boundary: about 0.35–0.39% charged-use savings under reliable
scale/control signaling, not guaranteed 24 dB delivery, perceptual improvement,
or superiority over published methods on a matched channel. Conditional false ACK
and final outage are disclosed. Published author-model baselines retain separate
AWGN/rate/feedback assumptions. Checkpoint binaries remain on Artemis; public
manifests do not imply public model downloads.

Author identities, affiliations, funding/conflicts, human scientific approval,
TrackChair policy and PDF eXpress validation remain author actions before submission.
Substantive AI assistance is acknowledged. No submission was performed.
