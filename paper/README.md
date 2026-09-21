# Compute Before Retransmit — IEEE VTC2027-Spring draft

The 2026-09-21 rewrite centers on frozen H2 FlowHARQ and a qualified H4 one-step
control. It does not promote H3 or the failed evidence-adaptive codec to primary
results. The original three-run study and existing ten-run extension are distinct.
No extra training or seed sweep was run for this revision.

The narrative is: receiver repair before a physical request → calibrated stopping
with a common MRC fallback → paired resource/quality evidence → decision-cell
explanation → honest scale/control overhead and finite-precision audit.
The original PSNR tolerance is -0.02 dB and LPIPS tolerance is +0.003. Positive
LPIPS deltas are not called improvements. The 24 dB threshold is an operating
target, not a proven reliability guarantee. Five-scale MS-SSIM is distinct from
the original single-scale SSIM archive.

## Rebuild

```bash
python3 scripts/build_spring_story.py
MPLCONFIGDIR=/tmp/flowharq-mpl python3 scripts/plot_spring_story.py
latexmk -pdf -outdir=build -cd paper/main.tex
```

Main numerical evidence is `experiments/spring_final_20260921/evidence.json`;
generated tables/macros use the `spring_` prefix. Original data can be restored
from `artifacts/spring_final_20260921/original_h2_h4_evidence.tar.gz` in a fresh
checkout. Full instructions and runtime snapshots are linked in that package.
`spring_references.bib` is generated from checked Crossref/DataCite metadata,
archived with hashes; old reference and table files are retained as historical
artifacts and are not inputs to this manuscript.

Author-model SwinJSCC, NTSCC and CDDM benchmarks use AWGN and explicitly reported
rates. DeepJSCC-f is author-code retraining with documented dataset/runtime
adaptations and richer channel-output feedback. None is mislabeled as a matched
Rayleigh/binary-feedback test. Recent FlowSem, channel-realization FM and generative
semantic HARQ are acknowledged as overlapping work, not omitted novelty threats.

## Submission status

The paper uses IEEEtran conference format and targets nominal five-page length.
Official VTC2027-Spring guidance checked on 2026-09-21 lists an extended deadline
of 30 September 2026, with the conference in Hamburg on 20–23 June 2027.
See `SUBMISSION_CHECKLIST.md` for authoritative links and remaining human actions.
Author identities, affiliations, funding, conflicts, venue-specific review policy,
and PDF eXpress validation must be confirmed by the submitting authors. No paper
has been submitted automatically and no acceptance is implied.

The existing VTC section-size audit in `VTC_SECTION_AUDIT.md` is historical
structure guidance; it is not a word-count prescription. The new draft is built
around the evidence rather than padding each section to a fixed count.
IEEE-required AI-assistance disclosure covers substantive text and code assistance,
not merely language polishing. The authors still need to verify all claims.
