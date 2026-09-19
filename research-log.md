# Research Log

Chronological, append-only record of research decisions and evidence.

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-09-19 | bootstrap | Implemented SwinJSCC + reliability head + identity-constrained partial FM + ACK/NACK + full second round. RTX PRO 6000 smoke and 40-epoch pilot completed. |
| 2 | 2026-09-19 | inner-loop | Pilot inference ablation found a conservative mask threshold removed most over-repair and gave an exploratory 1 pp NACK reduction at matched mean PSNR. |
| 3 | 2026-09-19 | outer-loop | Rejected pilot as paper evidence because tuning and evaluation shared the same 100 images. Added receiver CSI, time-correlated second round, MRC, LPIPS/SSIM, latency, deterministic calibration/test split, 3 training seeds and 3 channel seeds. |
| 4 | 2026-09-19 | inner-loop | Held-out v2 completed: NACK delta -0.10 pp with 95% CI [-0.41, 0.20] pp; PSNR delta -0.004 dB with CI [-0.023, 0.016]. Result is statistically inconclusive. |
| 5 | 2026-09-19 | outer-loop | Mechanism diagnosis: post-flow calibration MAE (~0.93 dB) dwarfs repair effect; squared velocity loss is exposed to extreme ZF residuals; original retransmission penalty is absent. Direction DEEPEN through H1 robust repair, H2 target-aware decision, H3 cost-aware joint training. |
| 6 | 2026-09-19 | protocol | Locked H1 before execution: same-seed comparison of v2, stronger reconstruction weight, normalized Huber velocity loss, and their combination. Acceptance requires at least +0.05 dB FM-only PSNR with LPIPS delta no worse than +0.003 on calibration. |
| 7 | 2026-09-19 | inner-loop | Implemented H1 objectives and paired bootstrap comparator; 17 local and Artemis tests passed. Submitted RTX array `11397130_[0-3]`. Held-out test remains untouched pending calibration selection. |
| 8 | 2026-09-19 | evidence | Audited final-epoch v2 training logs. FM-loss medians are ~0.27, while seed-wise maxima are 1.7e3, 8.4e3, and 1.6e5. This validates the preregistered deep-fade outlier mechanism motivating normalized Huber training. |
| 9 | 2026-09-19 | evidence | Oracle ACK/NACK audit on 17,280 held-out paired cases: FM crosses the 24 dB boundary upward 37 times and downward 39 times. Oracle retransmission saving is -0.012 pp, so v2's learned +0.104 pp is a decision-calibration artifact. Prioritize repair before H2. |
| 10 | 2026-09-19 | inner-loop | H1 calibration accepted. Best prelocked setting is normalized Huber + reconstruction weight 1.0, K=4, tau=0.8: FM-only PSNR +0.0714 dB, bootstrap CI [+0.0570,+0.0866], LPIPS +0.00203. Four of six conservative settings pass. |
| 11 | 2026-09-19 | gate | Frozen H1 decision calibration saves 1.04 pp NACK with +0.022 dB final PSNR on calibration. Submitted first held-out replication as jobs `11397165_[0-2]` and `11397168`; test data were not used for H1 selection. |
| 12 | 2026-09-19 | held-out gate | Seed 2030 across channel seeds 7001--7003 saves 2.1875 pp NACK relative to adaptive HARQ while changing PSNR by +0.0327 dB and LPIPS by +0.00145. Gate passed; submitted seeds 2031--2032 and the remaining confirmatory 3x3 matrix as jobs `11397172`, `11397174`, `11397175`, and `11397176`. |
| 13 | 2026-09-19 | confirmatory | H1 3x3 matrix completed. Repair is significant (+0.0271 dB final PSNR, CI [+0.0120,+0.0422]), but NACK saving is 1.389 pp with CI crossing zero. All seeds are directionally positive; activate preregistered H2 target-boundary calibration. |
| 14 | 2026-09-19 | inner-loop | Implemented quality-head-only H2 with 24 dB boundary BCE; 19 tests and a two-step RTX smoke passed. Submitted jobs `11397196`, `11397200`, `11397202`, and `11397203` using fresh channel seeds 8001--8003. |
