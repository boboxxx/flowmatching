# VTC2027-Spring submission checklist

Checked on 2026-09-19 against the official conference pages.

- Venue: IEEE VTC2027-Spring, Hamburg, 20--23 June 2027.
- Regular-paper deadline: **30 September 2026** (extended deadline shown by the conference).
- Nominal length: **5 pages**. Up to 7 pages are allowed with overlength charges; the call asks authors not to submit more than 8 pages for review.
- Candidate tracks: Artificial Intelligence and Machine Learning for Communications; Signal Processing for Wireless Communications; Vehicular Communication.
- Official call: <https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/>
- Conference home: <https://events.vtsociety.org/vtc2027-spring/>

## Before submission

- [ ] Replace anonymous author block with the required review identity format after checking TrackChair settings.
- [ ] Insert final H4 one-shot-control result and freeze all numbers.
- [ ] Re-run `python -m pytest -q` and the paper-artifact generator.
- [ ] Compile and visually inspect every PDF page.
- [ ] Confirm that all plots remain legible in grayscale and at 100% zoom.
- [ ] Run IEEE PDF eXpress / conference PDF validation when the code is available.
- [ ] Re-run the dated novelty search immediately before submission.
- [ ] Disclose permitted AI-assisted language editing as required by the conference policy; the authors remain responsible for all scientific claims and text.
- [ ] Add author names, affiliations, funding acknowledgement, and conflict declarations only when the review policy allows them.
- [ ] Archive exact code commit, checkpoints, calibration JSON, channel seeds, and generated tables.

## Current package state

- Main source: `paper/main.tex`
- Compiled PDF: `paper/build/main.pdf`
- Main evidence: 3 training seeds x 3 fresh channel seeds on 80 held-out DIV2K images.
- External evidence: frozen Kodak24 evaluation with three additional channel seeds.
- Remaining scientific gate: H4 parameter-matched one-shot residual control.
