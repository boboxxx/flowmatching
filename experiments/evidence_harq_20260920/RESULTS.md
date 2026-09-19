# Evidence-HARQ feasibility results — 2026-09-20

This is a gated development study, not a completed posterior-HARQ paper. All negative
and numerically failed candidates are retained. Published baselines and own controls
are distinguished. One training seed (2027); channel draws are not training seeds.

## Codec feasibility

| Candidate | Execution | Oracle saving vs adaptive base/full | Gate |
|---|---|---:|---|
| candidate_a | completed | 0.000% | stop_candidate |
| candidate_b_failed | failed | not available | not evaluated |
| candidate_b_fixed | completed | 0.000% | stop_candidate |

### candidate_a

40 development images × 4 noise draws × 16 subsets; 2,560 reconstruction records.
Oracle knows the source and future noises and is not deployable. All failed images
remain in cost and outage averages. Quality matching additionally limits per-trial
PSNR loss to 0.1dB and LPIPS/MS-SSIM deterioration to 0.005.

| Control | Total CBR | PSNR | MSE | MS-SSIM | LPIPS | Outage |
|---|---:|---:|---:|---:|---:|---:|
| selective_clairvoyant | 0.028167 | 28.545 | 0.002365 | 0.96022 | 0.20861 | 33.12% |
| adaptive_base_full_clairvoyant | 0.028167 | 28.545 | 0.002365 | 0.96022 | 0.20861 | 33.12% |
| always_full | 0.042542 | 28.538 | 0.002367 | 0.96019 | 0.20842 | 33.12% |
| base_only | 0.021047 | 28.554 | 0.002362 | 0.96027 | 0.20863 | 33.12% |

Failed checks: saving_at_least_10_percent, full_refines_base.

### candidate_b_fixed

40 development images × 4 noise draws × 16 subsets; 2,560 reconstruction records.
Oracle knows the source and future noises and is not deployable. All failed images
remain in cost and outage averages. Quality matching additionally limits per-trial
PSNR loss to 0.1dB and LPIPS/MS-SSIM deterioration to 0.005.

| Control | Total CBR | PSNR | MSE | MS-SSIM | LPIPS | Outage |
|---|---:|---:|---:|---:|---:|---:|
| selective_clairvoyant | 0.028167 | 28.554 | 0.002377 | 0.96002 | 0.20838 | 33.12% |
| adaptive_base_full_clairvoyant | 0.028167 | 28.554 | 0.002377 | 0.96002 | 0.20838 | 33.12% |
| always_full | 0.042542 | 28.507 | 0.002394 | 0.95963 | 0.20793 | 33.12% |
| base_only | 0.021047 | 28.554 | 0.002362 | 0.96027 | 0.20863 | 33.12% |

Failed checks: saving_at_least_10_percent, full_refines_base.

## Published author baselines with paid metadata

Kodak24 center256, AWGN10dB; 10 paired channel draws per image. All weights are
released author checkpoints; scale uses IEEE binary16 and metadata/control is charged
at 0.5 information bits per complex channel use. NTSCC additionally pays for its rate
map. These differ from the previously archived author-native free-scale assumptions.

Pairing means common image/draw identifiers and integer noise seeds; different
architectures and packet dimensions do not receive identical-shaped noise tensors.

| Method | Payload CBR | Total CBR | PSNR | MSE | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| ntscc_quality1 | 0.025202 | 0.035792 | 28.900 | 0.001630 | 0.94310 | 0.20627 |
| ntscc_quality2 | 0.040896 | 0.051486 | 30.977 | 0.000961 | 0.96026 | 0.12789 |
| swin_mse_c32 | 0.020833 | 0.021006 | 28.394 | 0.001960 | 0.95184 | 0.21400 |
| swin_mse_c64 | 0.041667 | 0.041840 | 30.817 | 0.001140 | 0.97323 | 0.11035 |
| CDDM_author_C36_paid_scale | 0.023438 | 0.023610 | 27.613 | 0.002232 | 0.94402 | 0.21694 |
| CDDM_author_JSCC_snr10_C36_paid_scale | 0.023438 | 0.023610 | 27.330 | 0.002382 | 0.94318 | 0.22926 |
| CDDM_author_JSCC_snr13_C36_paid_scale | 0.023438 | 0.023610 | 27.380 | 0.002317 | 0.94184 | 0.20852 |

CDDM's two JSCC controls belong to the same published architecture study and do
not represent two additional published methods. Different payload rates must be
shown as frontier points rather than interpreted as equal-rate rankings. Ideal-output-
feedback DeepJSCC-f remains in its original separate result package.

## Scope of conclusions

Failure of a learned progressive codec prevents interpreting its oracle as the ceiling
of all active-HARQ systems. Stop this candidate's posterior/policy stage if its gate
fails. A new codec construction requires its own declared study. Reusing the observed
40 images for development cannot establish confirmatory performance. The current
40-image risk-calibration pilot cannot certify 5% conditional false-ACK risk at 95%
confidence, even at zero observed failures. No such guarantee is claimed.

SAFG-HARQ publication is verified, but its full journal implementation remains
unreproduced; do not substitute a home-made selection rule under that name.

See `DECISION.md` for the stopped-candidate decision and prerequisites of a new study.

See `protocol.md`, `SYSTEM_MODEL.md`, `AMENDMENT_01.md`, `NUMERICAL_FIX_B.md`,
`LITERATURE_NOTES.md`, and `REPRODUCE.md` for provenance and assumptions.
