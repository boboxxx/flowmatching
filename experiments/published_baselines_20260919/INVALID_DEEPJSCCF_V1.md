# INVALID — DO NOT USE FOR PAPER COMPARISONS

All checkpoints and test scores in the server directory
`results/published_baselines_20260919/deepjsccf_author/` (without `_v2`) are excluded.

Keras `.weights.h5` failed to serialize all TensorFlow Compression kernel/GDN
parameters. The reloaded ~13 dB test score is a checkpoint I/O failure, not the
performance of the published method. Best-epoch selection was affected as well.
Keep the files only as diagnostic evidence; do not include them in baseline tables.

The replacement run is `deepjsccf_author_v2/`, from scratch, same protocol/seed,
after passing exact roundtrip verification of all 113 two-layer weight tensors.
No data from the failed test are used for checkpoint/hyperparameter selection.
