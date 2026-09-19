"""Post-run source/environment snapshot; never mislabel it as launch provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ntscc-python", required=True)
    args = p.parse_args()
    if (args.output/"runtime.json").exists():
        raise FileExistsError("Use a fresh output directory for a completed runtime snapshot")
    args.output.mkdir(parents=True, exist_ok=True)
    for label, executable in (("cv", sys.executable), ("ntscc", args.ntscc_python)):
        freeze = subprocess.check_output([executable, "-m", "pip", "freeze"], text=True)
        (args.output/f"{label}-pip-freeze.txt").write_text(freeze)
    import torch
    (args.output/"torch-build.txt").write_text(torch.__config__.show())
    (args.output/"gpu.txt").write_text(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True))
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_evidence_harq.py",
                            "tests/test_metrics.py", "tests/test_channel.py"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output/"tests.txt").write_text(tests.stdout)
    tests.check_returncode()
    paths = list(Path("evidence_harq").glob("*.py")) + [Path(name) for name in (
        "scripts/evaluate_paid_jscc.py", "scripts/evaluate_author_cddm.py",
        "scripts/evaluate_published_checkpoint.py", "flowharq/metrics.py",
        "flowharq/perceptual.py", "ugp/model.py", "tests/test_evidence_harq.py")]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                  kind="post_run_environment_and_current_source_snapshot",
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  verification="tests/test_evidence_harq.py tests/test_metrics.py tests/test_channel.py passed",
                  base_commit=revision.stdout.strip() if revision.returncode == 0 else None,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  note="Version snapshot, not a hermetic binary/wheel lock; Sheng project root is an rsync-deployed source tree without Git metadata, so code identity is given by source hashes. Final evaluation path overrides were added after the registered runs.")
    (args.output/"runtime.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
