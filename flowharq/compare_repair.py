from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare paired direct and FM-only reconstructions on one split."
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-psnr-gain", type=float, default=0.05)
    parser.add_argument("--max-lpips-delta", type=float, default=0.003)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2030)
    return parser.parse_args()


def paired_rows(path: str) -> list[tuple[dict, dict]]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["image"],
                row["snr_db"],
                row["speed_kmh"],
                row["csi_age_ms"],
                row["channel_seed"],
            )
            grouped[key][row["method"]] = row
    pairs = []
    for key, methods in grouped.items():
        if not {"direct", "fm_only"}.issubset(methods):
            raise ValueError(f"missing paired methods for {key} in {path}")
        pairs.append((methods["direct"], methods["fm_only"]))
    if not pairs:
        raise ValueError(f"no samples in {path}")
    return pairs


def interval(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize(path: str, draws: int, rng: np.random.Generator) -> dict:
    pairs = paired_rows(path)
    metrics = {}
    for name in ("psnr", "ssim", "lpips"):
        values = np.asarray(
            [float(fm[name]) - float(direct[name]) for direct, fm in pairs],
            dtype=np.float64,
        )
        metrics[f"{name}_delta"] = float(values.mean())
        metrics[f"{name}_delta_ci95"] = interval(values, draws, rng)
    mask_rates = np.asarray([float(fm["mask_rate"]) for _, fm in pairs])
    return {
        "source_csv": str(Path(path)),
        "samples": len(pairs),
        "training_seed": int(pairs[0][1]["training_seed"]),
        "flow_steps": int(pairs[0][1]["flow_steps"]),
        "mask_threshold": float(pairs[0][1]["mask_threshold"]),
        "mask_rate": float(mask_rates.mean()),
        **metrics,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    candidates = [summarize(path, args.bootstrap, rng) for path in args.csv]
    for candidate in candidates:
        candidate["accepted"] = (
            candidate["psnr_delta"] >= args.min_psnr_gain
            and candidate["lpips_delta"] <= args.max_lpips_delta
        )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = max(
        accepted or candidates,
        key=lambda row: (row["psnr_delta"], -row["lpips_delta"]),
    )
    output = {
        "protocol": {
            "min_psnr_gain_db": args.min_psnr_gain,
            "max_lpips_delta": args.max_lpips_delta,
            "bootstrap_draws": args.bootstrap,
            "accepted_candidates": len(accepted),
        },
        "selected": selected,
        "candidates": candidates,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
