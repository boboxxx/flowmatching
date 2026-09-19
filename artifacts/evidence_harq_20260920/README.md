# Evidence-HARQ feasibility archive

The scientific result is a **failed codec feasibility gate**, not a successful
posterior-guided HARQ system. See the [research decision](../../experiments/evidence_harq_20260920/DECISION.md).

- `candidate_a/`, `candidate_b_fixed/`: completed training histories, selected
  enhancement checkpoints, split hashes, per-image/channel/subset metrics, decisions,
  per-run environment and server inventories. The original B-fixed source snapshot
  preserves launch-time code; the current evaluator additionally supports relocated paths.
- `candidate_b_failed/`: numerical failure retained, not a completed model result.
- `paid_baselines/`: SwinJSCC C32/C64 and NTSCC quality1/quality2, each 240 records.
- `cddm_kodak24.csv` and `.json`: CDDM and its two same-paper JSCC controls,
  720 records; this is one additional published model family, not three.
- `cddm_weights/`: public download sources, sizes and SHA-256, not redistributed
  author checkpoint binaries. Other author weight sources are included in baseline metadata.
- `runtime/`: post-run dependency, CUDA/driver, source-identity and test snapshots.
- `logs/`: complete recorded candidate training and paid-baseline progress logs.
- `archive_audit.json`: 6,800 scored reconstruction records verified; large author
  weights and redundant/failed-run checkpoints remain at their documented source.

CSV image hashes, channel seeds, subset identifiers, model/checkpoint manifests and
metric fields constitute the reconstruction **evaluation manifest**. Individual
reconstruction PNG/NPY tensors were not exported in this study; they are not claimed
to be included. Re-evaluation uses original datasets, which are not redistributed.
All four metrics are evaluated per image/draw and then averaged, with no seed-STD claim.
The 40 development images for codec gates are distinct from Kodak24 baseline images;
do not compare their aggregate scores as though they were the same test set.

`artifact_inventory.json` inside each candidate describes the original server run,
including server-only checkpoint paths. The top-level release inventory describes
files actually present in this archive. No posterior-policy checkpoint or calibrated
ACK certificate exists. The version snapshots are not a hermetic binary environment.
