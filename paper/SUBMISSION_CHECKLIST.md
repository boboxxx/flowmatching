# VTC2027-Spring submission checklist — 2026-09-21

Official sources:

- Conference and extended deadline: https://events.vtsociety.org/vtc2027-spring/
- Paper format/length: https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/
- IEEE AI disclosure: https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/

Target: Hamburg, 20–23 June 2027; regular-paper deadline 30 September 2026.
Nominal length is five pages; up to seven incurs overlength charges. Do not infer
the identity/review format from a generic IEEE template; check TrackChair.

## Evidence and manuscript

- [x] Freeze H2 as the primary method; keep original three-run and ten-run evidence separate.
- [x] Preserve H3's cross-SNR failure and the failed codec gate in the archive.
- [x] Charge source-dependent scales and ACK/NACK under an explicit signaling model.
- [x] Distinguish NACK percentage points, request reduction, payload saving and charged-use saving.
- [x] Distinguish original SSIM from newly evaluated five-scale MS-SSIM.
- [x] Use published author-model references with explicit protocol/feedback differences.
- [x] Remove claims that H4 proves FM uniquely necessary or isolates integration depth.
- [x] Check original H2 quality tolerance (-0.02 dB / +0.003 LPIPS), loss temperatures and learning rates.
- [x] Archive all 60 original H2 raw CSVs, calibration decisions and checksums.
- [x] Verify ten checkpoint hashes against the original inventory and preserve runtime source.
- [x] Cross-check cited publication metadata and acknowledge recent overlapping FM work.
- [x] Complete both new Artemis audit datasets and validate every output hash.
- [x] Insert the final four-metric/risk audit without suppressing negative results.
- [x] Compile the final five-page draft and inspect every rendered page.
- [x] Rebuild all final evidence on Artemis and validate the complete release package.

The final PDF and evidence package are distributed with the repository release
commit, not by claiming that the manuscript has already been submitted.

## Author actions before actual submission

- [ ] Verify every scientific claim, numerical result, and cited source; no automatic process can certify authorship responsibility.
- [ ] Replace the draft author block using the conference's actual review identity policy.
- [ ] Supply affiliations, emails, funding and conflicts where required.
- [ ] Check the target track and submission metadata in TrackChair.
- [ ] Run the conference's PDF validation/PDF eXpress workflow when its code is available.
- [ ] Confirm that the substantive AI-assistance acknowledgment meets current IEEE and conference requirements.
- [ ] Resolve public checkpoint hosting if publicly downloadable models are promised; current model binaries remain on Artemis, with public manifests only.
- [ ] Make the submission decision. This draft does not demonstrate matched-channel superiority over published HARQ schemes, low false-ACK risk, or real-link latency benefit.
