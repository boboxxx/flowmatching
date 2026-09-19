# Literature and citation audit

Retrieval date: 2026-09-19.  Scope: DeepJSCC image transmission, semantic
HARQ, flow-matching semantic communication, and receiver-side latent repair.
The audit uses primary publisher/preprint records plus an independent metadata
source wherever one exists.  BibTeX for DOI-bearing works was obtained through
the DOI content-negotiation endpoint (`Accept: application/x-bibtex`) and then
checked against the sources below.

## Search protocol and novelty boundary

Queries included `"flow matching" HARQ semantic communication image`,
`"flow matching" "hybrid automatic repeat request"`, `"virtual
retransmission" semantic communication flow matching`, `latent flow matching
wireless image transmission`, and exact-title searches for candidate works.
Sources searched were IEEE Xplore/DOI, arXiv, OpenReview, DBLP, J-STAGE,
Springer, CVF Open Access, and the official DIV2K site.

The search found multiple FM-based semantic receivers (FlowSem, RC-BFM,
channel-aware latent FM, and restoration FM) and multiple semantic/DeepJSCC
HARQ systems (fine-grained image HARQ, DJSCC-H, and generative text HARQ).  It
did **not** find a work whose receiver first performs masked latent FM repair
and then uses post-flow quality to decide whether to request a physical HARQ
round.  This is a dated search result, not a claim that no such work can exist.
The defensible novelty boundary is therefore **FM-before-HARQ decision with an
identity-constrained partial latent transport**, not “the first use of FM in
semantic communication” or “the first semantic HARQ.”

## Per-reference verification

| Key | Primary record | Independent check | Verified fields |
|---|---|---|---|
| `deepjscc` | DOI `10.1109/TCCN.2019.2919300` | arXiv:1809.01733; DBLP `journals/tccn/BourtsoulatzeKG19` | authors, title, journal, 2019, 5(3), 567–579 |
| `swinjscc` | DOI `10.1109/TCCN.2024.3424842` | arXiv:2308.09361; DBLP `journals/tccn/YangWDQNZ25` | authors, title, journal, 2025, 11(1), 90–104 |
| `flowmatching` | OpenReview `PqvMRDCJT9t` | DBLP `conf/iclr/LipmanCBNL23`; arXiv:2210.02747 | authors, title, ICLR 2023 |
| `flowsem` | arXiv:2608.21651 | arXiv metadata mirror search | authors, title, date, preprint status |
| `rcbfm` | arXiv:2607.24876 | J-GLOBAL `202602204379953601` | authors, title, date, preprint status |
| `channel_lfm` | DOI `10.1109/TCCN.2026.3705851` | IEEE TCCN metadata indexed by SciX | authors, title, volume, pages, 2026 |
| `fineharq` | DOI/IEEE Xplore `10.1109/TWC.2025.3532501` | IEEE ComSoc April 2025 contents digest | authors, title, 24(4), 3606–3622 |
| `djscc_h` | DOI/J-STAGE `10.1587/transfun.2025EAL2052` | final J-STAGE PDF | authors, title, E109.A(4), 829–833, 2026 |
| `genharq` | arXiv:2603.15068 | DBLP `journals/corr/abs-2603-15068` | authors, title, 2026, preprint status |
| `mobility_deepjscc` | DOI `10.1186/s13638-026-02632-7` | Springer article page | authors, title, journal, 2026 |
| `ssim` | DOI/IEEE Xplore `10.1109/TIP.2003.819861` | Crossref DOI metadata | authors, title, 13(4), 600–612, 2004 |
| `lpips` | DOI `10.1109/CVPR.2018.00068` | arXiv:1801.03924 | authors, title, CVPR 2018, 586–595 |
| `div2k` | DOI `10.1109/CVPRW.2017.150` | CVF Open Access; official DIV2K site | authors, title, CVPRW 2017; IEEE pagination |

## Claim-level cautions

- FlowSem begins its conditional generation from Gaussian noise and conditions
  on a coarse DeepJSCC reconstruction; do not describe it as token repair.
- RC-BFM begins from a channel-induced semantic state and addresses coupling
  and decoding latency; do not describe it as HARQ.
- The fine-grained semantic HARQ paper retransmits erroneous semantic bases;
  FlowHARQ's conference version uses a full physical second round after the
  virtual repair.  Selective token retransmission is future work.
- The generative semantic HARQ preprint is for text and VAE latents, not image
  DeepJSCC or flow matching.
- VTC novelty language should remain “to the best of our dated search” and
  should avoid an absolute priority claim.

