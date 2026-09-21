# Spring manuscript evidence audit

Frozen H2 results; original and expanded cohorts are distinct. No new training.

| Dataset | NACK saving (pp) | Payload saving (%) | Total saving at eta=0.5 (%) |
|---|---:|---:|---:|
| div2k | 1.319444 | 0.758430 | 0.352913 |
| kodak | 1.452546 | 0.845972 | 0.390886 |

## div2k: paired decision mechanism

| Event | Fraction (%) | PSNR contribution (dB) | LPIPS contribution |
|---|---:|---:|---:|
| both_nack | 72.4965 | +0.000000 | +0.0000000 |
| both_ack | 25.8750 | +0.027186 | +0.0007885 |
| saved_round | 1.4740 | -0.013322 | +0.0006729 |
| extra_round | 0.1545 | +0.000540 | -0.0000310 |

### Original ten-run stopping risk

| Policy | Final outage (%) | Joint false ACK (%) | Conditional false ACK (%) |
|---|---:|---:|---:|
| adaptive_harq | 71.1059 | 3.5156 | 13.5063 |
| flowharq | 70.8872 | 3.8368 | 14.0291 |

Finite-precision audit completed; changed ACK/NACK decisions: {'adaptive_harq': 0, 'flowharq': 0}.
Maximum ideal-branch difference from archived replay: {'psnr': 11.887733459472656, 'lpips': 0.46022963523864746, 'nack': 1.0}.

CPU draws differ from archived CUDA draws; the preceding replay differences are expected and are not a quantization error.

| Binary16 policy | NACK (%) | PSNR | MSE | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| direct | 0.0000 | 21.319377 | 0.01023943 | 0.800025 | 0.512687 |
| full_harq | 100.0000 | 21.984882 | 0.00858695 | 0.840536 | 0.482685 |
| fm_only | 0.0000 | 21.392299 | 0.01016547 | 0.797182 | 0.516388 |
| adaptive_harq | 74.6875 | 21.912125 | 0.00863016 | 0.837882 | 0.485263 |
| flowharq | 73.0208 | 21.948042 | 0.00860811 | 0.837091 | 0.486725 |

## kodak: paired decision mechanism

| Event | Fraction (%) | PSNR contribution (dB) | LPIPS contribution |
|---|---:|---:|---:|
| both_nack | 70.1157 | +0.000000 | +0.0000000 |
| both_ack | 28.1655 | +0.031664 | +0.0007735 |
| saved_round | 1.5856 | -0.006873 | +0.0005638 |
| extra_round | 0.1331 | +0.000475 | -0.0000270 |

### Original ten-run stopping risk

| Policy | Final outage (%) | Joint false ACK (%) | Conditional false ACK (%) |
|---|---:|---:|---:|
| adaptive_harq | 83.3507 | 14.3981 | 50.8793 |
| flowharq | 82.9167 | 15.0463 | 50.5738 |

Finite-precision audit completed; changed ACK/NACK decisions: {'adaptive_harq': 0, 'flowharq': 0}.
Maximum ideal-branch difference from archived replay: {'psnr': 8.831783294677734, 'lpips': 0.4849582314491272, 'nack': 1.0}.

CPU draws differ from archived CUDA draws; the preceding replay differences are expected and are not a quantization error.

| Binary16 policy | NACK (%) | PSNR | MSE | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| direct | 0.0000 | 21.470190 | 0.00855403 | 0.779700 | 0.557457 |
| full_harq | 100.0000 | 22.104738 | 0.00714213 | 0.825870 | 0.518947 |
| fm_only | 0.0000 | 21.579886 | 0.00840861 | 0.776031 | 0.562571 |
| adaptive_harq | 74.3056 | 22.033440 | 0.00719147 | 0.821969 | 0.523831 |
| flowharq | 71.7014 | 22.070383 | 0.00716471 | 0.820905 | 0.525728 |

These costs assume reliable metadata/control and exclude pilots, headers, scheduling and retransmission of erroneous control messages.
The finite-precision check uses one frozen training seed and one existing channel seed per dataset. It is not a replacement for the original confirmatory experiment.
