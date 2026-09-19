# VTC2027-Spring submission checklist

Checked on 2026-09-19 against the official conference pages.

- Venue: IEEE VTC2027-Spring, Hamburg, 20--23 June 2027.
- Regular-paper deadline: **30 September 2026** (extended deadline shown by the conference).
- Nominal length: **5 pages**. Up to 7 pages are allowed with overlength charges; the call asks authors not to submit more than 8 pages for review.
- Candidate tracks: Artificial Intelligence and Machine Learning for Communications; Signal Processing for Wireless Communications; Vehicular Communication.
- Official call: <https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/>
- Conference home: <https://events.vtsociety.org/vtc2027-spring/>

## Before submission

- [ ] **2026-09-19 baseline revision gate:** replace the mechanism-adapter external
  comparison with verified published-paper baselines. SwinJSCC/NTSCC author-weight
  AWGN results and the new DeepJSCC-f author-code training are documented in
  `experiments/published_baselines_20260919/protocol.md`; do not mix their channel,
  feedback, pretraining, or rate assumptions with the old Rayleigh table.
- [ ] Replace/qualify any perceptual-quality improvement claim: the four-metric
  H3 Kodak audit shows FM-only improves PSNR but worsens MS-SSIM and LPIPS at all
  six SNR points. Audit files are in `artifacts/published_metric_audit_20260919`.
- [ ] Replace anonymous author block with the required review identity format after checking TrackChair settings.
- [x] Insert final H4 one-shot-control result and freeze all numbers.
- [x] Re-run `python -m pytest -q` (26 tests on the locked Sheng environment).
- [x] Regenerate paper numbers from the locked H2/H3 and external-baseline artifacts.
- [x] Compile and visually inspect every PDF page.
- [x] Confirm that the primary plot remains legible without uncertainty bands.
- [x] Audit eight accessible VTC full-text versions (2024/2025), distinguish
  author versions from final papers, and record section counts and logic.
- [x] Align the manuscript with code: oracle versus predicted masks, explicit
  integration update, normalization-scale side information, and ablation settings.
- [ ] Account for token-scale signaling overhead or justify a practical receiver
  implementation before making complete-link rate or deployment claims.
- [ ] Run IEEE PDF eXpress / conference PDF validation when the code is available.
- [ ] Re-run the dated novelty search immediately before submission.
- [ ] Disclose permitted AI-assisted language editing as required by the conference policy; the authors remain responsible for all scientific claims and text.
- [ ] Add author names, affiliations, funding acknowledgement, and conflict declarations only when the review policy allows them.
- [ ] Archive exact code commit, checkpoints, calibration JSON, channel seeds, and generated tables.

## Current package state

- Main source: `paper/main.tex`
- Compiled PDF: `paper/build/main.pdf`
- Main evidence: one locked H3 training run, one fixed paired channel realization,
  six SNRs x four speeds, and 80 untouched DIV2K images; exact means only.
- External evidence: frozen Kodak24 evaluation and an independently implemented
  DeepJSCC-f mechanism adapter, both explicitly qualified in the paper.
- H4 mechanism gate: completed and included in the generated paper artifacts.
- H3 status: aggregate rate/quality checks pass, but the primary DIV2K
  cross-SNR consistency rule fails (3/6 rather than 4/6 positive SNR bins).
- The paper reports that failure directly and does not use the older seed-sweep
  confidence-interval claim.
