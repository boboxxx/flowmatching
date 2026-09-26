# Sheng revision results

Frozen-checkpoint diagnostics, not new training or untouched confirmation.

## Calibration-selected decision margins

| Dataset | Policy | Margin dB | ACK coverage % | Conditional false ACK % | Outage % |
|---|---|---:|---:|---:|---:|
| calibration | adaptive_harq | 1.0 | 24.583 | 2.542 | 59.792 |
| calibration | flowharq | 1.0 | 26.875 | 4.651 | 60.000 |
| div2k | adaptive_harq | 1.0 | 18.177 | 4.967 | 70.330 |
| div2k | flowharq | 1.0 | 19.323 | 3.863 | 69.965 |
| kodak | adaptive_harq | 1.0 | 18.692 | 31.889 | 82.870 |
| kodak | flowharq | 1.0 | 19.850 | 30.321 | 82.118 |

## Same-channel published-codec transfer

One channel draw per image/condition; six SNRs and four speeds. Author SwinJSCC retains global scale normalization; FlowHARQ retains token scales. The Chase wrapper is not a published HARQ reproduction.

| Dataset | Method | PSNR | MSE | MS-SSIM | LPIPS | Charged uses |
|---|---|---:|---:|---:|---:|---:|
| div2k | SwinJSCC_author_one_shot | 23.0710 | 0.025345 | 0.7730 | 0.3463 | 4130.0 |
| div2k | SwinJSCC_author_Chase_wrapper | 26.1365 | 0.009807 | 0.8852 | 0.2641 | 8228.0 |
| div2k | same_codec_direct | 21.3680 | 0.010282 | 0.8011 | 0.5119 | 12290.0 |
| div2k | same_codec_full_HARQ | 21.9923 | 0.008628 | 0.8403 | 0.4829 | 16388.0 |
| div2k | adaptive_HARQ | 21.9140 | 0.008672 | 0.8371 | 0.4856 | 15284.5 |
| div2k | FlowHARQ | 21.9446 | 0.008653 | 0.8361 | 0.4873 | 15197.0 |
| kodak | SwinJSCC_author_one_shot | 23.0164 | 0.021348 | 0.7506 | 0.3758 | 4130.0 |
| kodak | SwinJSCC_author_Chase_wrapper | 26.3372 | 0.005988 | 0.8851 | 0.2771 | 8228.0 |
| kodak | same_codec_direct | 21.4573 | 0.008556 | 0.7798 | 0.5576 | 12290.0 |
| kodak | same_codec_full_HARQ | 22.0795 | 0.007173 | 0.8249 | 0.5199 | 16388.0 |
| kodak | adaptive_HARQ | 22.0193 | 0.007218 | 0.8216 | 0.5236 | 15271.0 |
| kodak | FlowHARQ | 22.0507 | 0.007193 | 0.8204 | 0.5258 | 15171.4 |

All-four-metric / lower-charged-use dominance by published one-shot Swin on div2k: **False**.
All-four-metric / lower-charged-use dominance by published one-shot Swin on kodak: **False**.

The two-round author wrapper exceeds FlowHARQ on all four mean quality metrics at lower charged use on Kodak, but not on DIV2K: DIV2K mean MSE is worse despite better mean PSNR. This is a transfer diagnostic with unequal training and scale normalization, not a matched-training HARQ superiority claim.
