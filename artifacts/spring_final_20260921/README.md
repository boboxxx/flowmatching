# Frozen FlowHARQ Spring evidence package

This package supports the 2026-09-21 H2-focused manuscript revision.

- `original_h2_h4_evidence.tar.gz`: 60 original H2 per-image CSVs (ten trained
  receivers, three channel draws, two datasets), all ten calibration decisions,
  H4 calibration outputs, the selected H1 calibration evidence, and the original GPU microbenchmark.
- `original_evidence_manifest.json`: archive checksum and each member's checksum,
  byte count and CSV row count. Archive paths retain their original `results/` prefix.
- `kodak/` and `div2k/`: completed Artemis CPU audit outputs (5,760 and 19,200 rows).
  Both `status.json` records say `completed`; `validation.json` records passed checks.
- `runtime/`: freshly verified hashes for all ten frozen H2 model binaries and
  calibration JSONs, plus the actual Artemis project source dependency snapshot. This
  separates executed code from unrelated working-tree edits.
- `execution/`: scheduler records and run logs, including the failed Kodak path
  attempt and its corrected retry. No failed output is included as a completed test.

The new audit shares every observation between ideal and binary16 restoration,
computes RGB MSE/PSNR, AlexNet LPIPS and five-scale MS-SSIM, and records both final
outage and false first-round acceptance. The CPU stream differs from the old CUDA
stream even when the seed integer is the same. Compare quantization modes within
this audit; use the archived original CSVs for the original multi-run claim.

Checkpoint binaries are retained at the Artemis paths in `runtime/inventory.json`;
they are not publicly hosted by this small evidence archive. Public artifact hashes
and local availability must not be described as public download availability.
Source images are not redistributed. Input image hashes and crop policy identify
the test data. No new reconstructed PNG collection is saved by this metric-only
audit; each CSV row identifies its reconstruction by image/seed/SNR/speed/method/
scale mode and the frozen inputs. Existing author-baseline reconstruction inventories
remain in their original published-baseline packages.

See `experiments/spring_final_20260921/REPRODUCE.md` for dependencies and commands.
Third-party SwinJSCC files are recorded by hash and obtained from the pinned
submodule (`a6d0e6da53548976acbe9317839a077ef31f190f`), not redistributed in this
artifact directory. The five recorded upstream files match that local checkout.
Raw CSVs retain the original CRLF records and source snapshots retain their original
formatting. Do not normalize their line endings: the published checksums cover bytes.
