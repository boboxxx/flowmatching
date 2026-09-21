# Reproduce the frozen Spring audit

## Scope and prerequisites

Original H2 CSVs are sufficient to regenerate the main claim, decision decomposition,
and charged-cost sensitivity. Neural inference additionally requires the model file
listed by SHA-256 in `artifacts/spring_final_20260921/runtime/inventory.json`, the
original datasets, LPIPS weights, and the actual executed source snapshot.
Do not assume the current working tree is byte-identical to the archived runtime.

The base environment is `environment/artemis-cu128.requirements.txt` (Python 3.10,
PyTorch 2.7.1+cu128). Each completed audit contains its actual `pip-freeze.txt`.
The only added metric dependency is `pytorch-msssim==1.0.0`, installed without
dependency upgrades into a project-local directory and added to `PYTHONPATH`.
The CSV metrics use RGB [0,1], per-image MSE and PSNR, AlexNet LPIPS with [-1,1]
inputs, and standard five-scale MS-SSIM (data_range=1, window=11). Report means of
per-image PSNR, not PSNR computed from mean MSE. Source images use center256 crops.

## Tables and figures, no model inference

In a fresh repository clone, extract the archive at repository root:

```bash
tar -xzf artifacts/spring_final_20260921/original_h2_h4_evidence.tar.gz
python3 scripts/build_spring_story.py
MPLCONFIGDIR=/tmp/flowharq-mpl python3 scripts/plot_spring_story.py
latexmk -pdf -outdir=build -cd paper/main.tex
```

The builder checks original CSV counts, paired MRC identity, metric decomposition,
and completed audit hashes. It writes machine-readable `evidence.json`, manuscript
macros, and tables. Numeric plot data are never fitted or invented. Regenerating
figures requires Matplotlib; the statistical builder itself uses the standard library.

## Exact inference inputs

Create an isolated replay directory containing `runtime/source/` at its root,
`scripts/audit_spring_link.py`, and the pinned SwinJSCC submodule at
`upstream/SwinJSCC/` (commit `a6d0e6da53548976acbe9317839a077ef31f190f`).
Its five `net/` source hashes are verified in the runtime inventory; third-party
code is not duplicated in the published artifact. Install dependencies from the
lock and make LPIPS available.
Restore the checkpoint and calibration JSON at paths passed explicitly below.
Use the exact audit script snapshot to reproduce the recorded script hash.

```bash
python scripts/audit_spring_link.py \
  --checkpoint /path/to/seed_2030/checkpoint.pt \
  --calibration /path/to/seed_2030/decision.json \
  --data /path/to/DIV2K_valid_HR --dataset div2k --seed 8001 \
  --device cpu --threads 8 --output /new/output/div2k
```

For Kodak use its 24-image directory, `--dataset kodak --seed 9001`. The DIV2K
directory must contain all 100 validation images; the first 20 are excluded inside
the script. Outputs refuse to overwrite an existing directory. GPU execution has
different random draws; do not label it a CPU bit-exact replay.

## Artemis execution

`slurm/spring_link_audit_cpu.sbatch` parameterizes the project, interpreter, datasets,
and output root with `FLOWHARQ_*` variables. The CPU allocation uses eight threads
per dataset task, no GPU. The original GPU job 11403772 was canceled while pending
because resources were unavailable. DIV2K ran as 11403790_0. The initial Kodak
attempt 11403790_1 failed before evaluation because its default data path was absent;
correcting it to the original experiment's documented directory produced 11403794_1.
No model or threshold changed in that correction.

The raw policy-latency benchmark is batch one, 20 warmups and 100 repetitions, on a
single fixed input with observations precomputed. CPU results may be affected by
shared-node contention and execution order; they are archived but not used to assert
latency superiority. There is no additional training or new seed-selection experiment.
