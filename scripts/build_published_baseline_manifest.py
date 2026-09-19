"""Create a compact, checksummed inventory of published-baseline artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "published_baselines_20260919_manifest.json"
INVENTORIES = (
    ROOT / "artifacts" / "published_author_benchmark_20260919_complete",
    ROOT / "artifacts" / "published_metric_audit_20260919",
    ROOT / "artifacts" / "published_deepjsccf_v2_20260919",
    ROOT / "experiments" / "published_baselines_20260919",
    ROOT / "environment",
)


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reconstruction_list = ROOT / "artifacts" / "published_deepjsccf_v2_20260919" / "RECONSTRUCTIONS.sha256"
    reconstruction_lines = [line for line in reconstruction_list.read_text().splitlines() if line.strip()]
    if len(reconstruction_lines) != 48:
        raise ValueError(f"Expected 48 reconstruction checksums, found {len(reconstruction_lines)}")
    files = []
    for directory in INVENTORIES:
        if not directory.exists():
            raise FileNotFoundError(directory)
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            if path.resolve() == args.output.resolve():
                continue
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    payload = {
        "schema": 1,
        "scope": "Published-paper baseline artifacts and legacy four-metric audit",
        "reconstruction_arrays": {
            "count": 48,
            "not_stored_in_git": True,
            "sha256_list": str(reconstruction_list.relative_to(ROOT)),
            "reason": "~1.3 GB float arrays; GitHub regular Git files are unsuitable",
        },
        "files": files,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"files": len(files), "output": str(args.output)}))


if __name__ == "__main__":
    main()
