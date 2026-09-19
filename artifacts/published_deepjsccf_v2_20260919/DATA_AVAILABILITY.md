# DeepJSCC-f v2 data availability

This directory contains the complete public result record for the corrected
DeepJSCC-f v2 run:

- `c2_l2/` and `c4_l1/`: raw per-image/per-draw CSVs, aggregate summaries,
  training histories, selected-run status and reconstruction-export manifests.
- `RECONSTRUCTIONS.sha256`: SHA-256 for all 48 source reconstruction arrays:
  24 Kodak images × two models. The c2_l2 arrays contain both round-one and
  round-two reconstructions; the c4_l1 arrays contain the one-shot control.
- `deepjsccf_training_v2.log`: training, early-stop, export and scoring log.

The 48 float `.npz` reconstruction arrays total approximately 1.3 GB and the
selected TensorFlow checkpoints include 100-MB files. Neither is stored as a
regular Git object. This is intentional: GitHub rejects files larger than 100 MB
and performs poorly with large binary research artifacts. The tracked manifest,
CSV, run configuration, hashes, exact environment lock and exporter scripts
are sufficient to regenerate them and verify each regenerated array against
`RECONSTRUCTIONS.sha256`.

The files under the archived run relative path
`results/published_baselines_20260919/deepjsccf_author_v2/` are the authoritative
binary source at the time of this release. The earlier `deepjsccf_author` run is
explicitly invalid and is not represented in this directory.
