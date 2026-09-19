# Reproducible environments

`artemis-cu128.requirements.txt` records the exact Python packages used for the
confirmatory Artemis runs.  The matching interpreter is Python 3.10.0 and the
GPU runtime is CUDA 12.8.  Create the environment from this directory with:

```bash
conda env create -f artemis-cu128.yml
```

The lock deliberately pins PyTorch's CUDA build.  CPU-only development should
use `requirements.txt`; it is not a substitute for the confirmatory lock.

LPIPS was historically installed into a repository-local `vendor/` directory.
New runs install the same `lpips==0.1.4` release into the locked environment so
evaluation does not depend on an untracked vendor tree.

