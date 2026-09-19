from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


METRICS = ("psnr", "ssim", "lpips", "nack", "transmission_rounds")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate independent FlowHARQ runs and paired confidence intervals."
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def mean(values):
    materialized = list(values)
    return sum(materialized) / len(materialized)


def interval(values: list[float]) -> dict:
    n = len(values)
    center = mean(values)
    if n < 2:
        return {"n": n, "mean": center, "std": float("nan"), "ci_low": center, "ci_high": center}
    variance = sum((value - center) ** 2 for value in values) / (n - 1)
    standard_deviation = math.sqrt(variance)
    half_width = float(t.ppf(0.975, n - 1)) * standard_deviation / math.sqrt(n)
    return {
        "n": n,
        "mean": center,
        "std": standard_deviation,
        "ci_low": center - half_width,
        "ci_high": center + half_width,
    }


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != "test":
                    raise ValueError(f"non-test row in {path}")
                for key in (
                    "snr_db",
                    "speed_kmh",
                    "psnr",
                    "ssim",
                    "lpips",
                    "nack",
                    "transmission_rounds",
                ):
                    row[key] = float(row[key])
                row["training_seed"] = int(row["training_seed"])
                row["channel_seed"] = int(row["channel_seed"])
                rows.append(row)
    if not rows:
        raise ValueError("no rows")
    return rows


def run_means(rows: list[dict], condition_key: str | None) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row["training_seed"],
            row["method"],
            row[condition_key] if condition_key else "all",
        )
        for metric in METRICS:
            grouped[key][metric].append(row[metric])
    return {
        key: {metric: mean(values) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def aggregate(run_values: dict, condition_name: str) -> list[dict]:
    grouped = defaultdict(list)
    for (_, method, condition), metrics in run_values.items():
        for metric, value in metrics.items():
            grouped[(method, condition, metric)].append(value)
    result = []
    for (method, condition, metric), values in sorted(grouped.items(), key=str):
        result.append(
            {"method": method, condition_name: condition, "metric": metric, **interval(values)}
        )
    return result


def paired_deltas(run_values: dict, condition_name: str) -> list[dict]:
    conditions = sorted({(train, condition) for train, _, condition in run_values}, key=str)
    grouped = defaultdict(list)
    for train, condition in conditions:
        adaptive = run_values[(train, "adaptive_harq", condition)]
        flow = run_values[(train, "flowharq", condition)]
        for metric in METRICS:
            grouped[(condition, metric)].append(flow[metric] - adaptive[metric])
    return [
        {condition_name: condition, "metric": metric, **interval(values)}
        for (condition, metric), values in sorted(grouped.items(), key=str)
    ]


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = load(args.csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_snr_runs = run_means(rows, condition_key="snr_db")
    per_speed_runs = run_means(rows, condition_key="speed_kmh")
    overall_runs = run_means(rows, condition_key=None)
    estimates = aggregate(per_snr_runs, "snr_db") + aggregate(overall_runs, "snr_db")
    deltas = paired_deltas(per_snr_runs, "snr_db") + paired_deltas(overall_runs, "snr_db")
    speed_estimates = aggregate(per_speed_runs, "speed_kmh")
    speed_deltas = paired_deltas(per_speed_runs, "speed_kmh")
    write_csv(output / "estimates.csv", estimates)
    write_csv(output / "paired_deltas.csv", deltas)
    write_csv(output / "speed_estimates.csv", speed_estimates)
    write_csv(output / "speed_paired_deltas.csv", speed_deltas)
    provenance = {
        "input_csvs": [str(Path(path)) for path in args.csv],
        "training_seeds": sorted({row["training_seed"] for row in rows}),
        "channel_seeds": sorted({row["channel_seed"] for row in rows}),
        "independent_run_pairs": len(
            {(row["training_seed"], row["channel_seed"]) for row in rows}
        ),
        "test_images": len({row["image"] for row in rows}),
        "rows": len(rows),
        "confidence_interval": (
            "two-sided 95% Student-t interval over independent training-seed means; "
            "each mean averages all channel seeds"
        ),
        "paired_difference": "FlowHARQ minus adaptive HARQ within each training seed",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
