# Source-grounded baseline selection

Inspected on 2026-09-20. A supplied proposal's citation is a lead, not verification.

## Mandatory executable references

**SwinJSCC**, IEEE TCCN 11(1), 90–104, 2025,
[author code](https://github.com/semcomm/SwinJSCC),
[DOI](https://doi.org/10.1109/TCCN.2024.3424842).
Use released DIV2K MSE C32/C64 checkpoints. Our new base loads exact trainable
weights, rebuilds resolution buffers, and changes only scale transport to quantized,
explicitly charged metadata. A pretrained strong base is needed to avoid attributing
a weak-codec improvement to posterior inference. The legacy experiment's small
Swin backbone is not the same architecture/checkpoint as the official baseline.

**NTSCC**, IEEE JSAC 40(8), 2300–2316, 2022,
[author code and bibliographic record](https://github.com/wsxtyrdd/NTSCC_JSAC22).
Its spatial rate allocation is already content-adaptive; comparison with a constant
rate codec alone would miss an important competing mechanism. The rate map's bits
and scale transport must be counted, not just nonzero forward symbols. Existing
author-protocol runs use a capacity-limit rate-map conversion; preserve that assumption
in their labels rather than merging it with the new fixed-rate control budget.

**DeepJSCC-f**, IEEE JSAIT 1(1), 178–193, 2020,
[author manuscript](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/KurkaGunduz_deepJSCC-f.pdf),
[code](https://github.com/kurka/deepJSCC-feedback).
The paper uses received channel output as feedback to form later transmissions.
Our finite mask-only feedback system does not expose that information to the encoder.
Keep the verified V2 author-architecture reproduction as a published reference, explicitly
label ideal output feedback, and avoid treating its forward CBR as total bidirectional cost.

**CDDM**, IEEE TWC 23(9), 11168–11183, 2024,
[publisher](https://ieeexplore.ieee.org/document/10480348/),
[manuscript](https://arxiv.org/pdf/2309.08895),
[author code](https://github.com/Wireless3C-SJTU/CDDM-channel-denoising-diffusion-model-for-semantic-communication).
Selected as a formally published generative receiver, with public source and weights.
The implemented inference follows the author's SNR-13 encoder, channel at 10 dB,
channel diffusion sampler, and retrained decoder. Author sampler inputs are scaled by
sqrt(1+sigma_real²), with schedule T=1000 and minimum-SNR setting 10 dB. The
evaluated configuration calls the noise predictor 93 times. We retain that schedule;
calling a generic 4-step diffusion implementation 'CDDM' would not reproduce the method.
Two original JSCC controls (trained SNR 10/13) isolate the effect of the denoiser/redecoder.
Common metrics add standard five-scale MS-SSIM and LPIPS to the author's distortion
evaluation. Quantized scalar transport is an explicitly labelled protocol adaptation.
The author sampler is deterministic given its observation; do not claim it provides
calibrated posterior diversity merely because it uses a diffusion model.

## Direct progressive/selective literature

**DeepJSCC-l**, IEEE TWC 20(12), 8081–8095, 2021,
[full author manuscript](https://arxiv.org/pdf/2009.12480),
[institutional publication record](https://www.imperial.ac.uk/information-processing-and-communications-lab/publications/).
Section IV studies successive refinement; Section V multiple descriptions. Independent
layers and arbitrary-subset decoding already exist here. Progressive/packet masking by
itself is therefore not novelty. Our fusion codec differs from its documented encoders,
decoders, and training, and is not presented as a reproduction of DeepJSCC-l.

**SAFG-HARQ**, IEEE TWC 24(4), 3606–3622, 2025,
[publisher record](https://doi.org/10.1109/TWC.2025.3532501).
The publisher abstract confirms semantic-base coding, residual refinement, and selective
retransmission of erroneous semantic bases. This is the closest mandatory related work.
The reported percentage reductions are source-specific, not directly transferable to
our channel/dataset/service. Full journal methods and author code have not been obtained.
The related [2024 arXiv version](https://arxiv.org/pdf/2308.06599) is a six-page
semantic-base paper with different authorship and scope, and cannot be substituted for
the 2025 SAFG-HARQ implementation. Do not invent a random-mask or token-error baseline
and attach this published name to it. No SAFG-HARQ numerical reproduction is claimed.

## Background excluded from the published-baseline count

[DiffCom](https://arxiv.org/html/2406.07390v2) motivates channel-conditioned posterior
inference. Its author repository currently gives an arXiv citation; a formal venue
has not been verified during this audit. It stays background until verified. LTT,
RC-BFM, ADDPS and FlowSem also require individual publication/source verification.

The proposed contribution, if feasibility and comparisons support it, is receiver-only
prediction of *marginal acquisition utility under a charged feedback budget*. Neither
posterior sampling, selective HARQ, progressive coding nor risk calibration alone is new.
