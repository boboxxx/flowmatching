from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select FlowHARQ inference settings on a calibration split only."
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-psnr", type=float, default=24.0)
    parser.add_argument(
        "--target-psnrs",
        help="comma-separated quality targets to search on calibration data",
    )
    parser.add_argument(
        "--reference-config",
        help="use the target PSNR frozen in another calibration JSON",
    )
    parser.add_argument("--psnr-tolerance", type=float, default=0.02)
    parser.add_argument("--lpips-tolerance", type=float, default=0.003)
    return parser.parse_args()


def load_samples(path: str) -> tuple[list[dict], dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    metadata = None
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "calibration":
                raise ValueError(f"{path} is not a calibration CSV")
            key = (
                row["image"],
                float(row["snr_db"]),
                float(row["speed_kmh"]),
                float(row["csi_age_ms"]),
            )
            grouped[key][row["method"]] = row
            candidate = {
                "flow_steps": int(row["flow_steps"]),
                "mask_threshold": float(row["mask_threshold"]),
                "training_seed": int(row["training_seed"]),
                "channel_seed": int(row["channel_seed"]),
            }
            if metadata is None:
                metadata = candidate
            elif metadata != candidate:
                raise ValueError(f"inconsistent metadata within {path}")
    samples = []
    for key, methods in grouped.items():
        required = {"direct", "full_harq", "fm_only"}
        if not required.issubset(methods):
            raise ValueError(f"incomplete sample {key} in {path}")
        samples.append(methods)
    if not samples or metadata is None:
        raise ValueError(f"no samples in {path}")
    return samples, metadata


def mean(values):
    materialized = list(values)
    return sum(materialized) / len(materialized)


def metric(row: dict, name: str) -> float:
    return float(row[name])


def select(first: dict, second: dict, nack: bool, name: str) -> float:
    return metric(second if nack else first, name)


def evaluate(path: str, target_psnr: float) -> dict:
    samples, metadata = load_samples(path)
    direct_errors = []
    fm_errors = []
    for methods in samples:
        direct_errors.append(
            metric(methods["direct"], "psnr")
            - metric(methods["direct"], "predicted_direct_psnr_raw")
        )
        fm_errors.append(
            metric(methods["fm_only"], "psnr")
            - metric(methods["fm_only"], "predicted_fm_psnr_raw")
        )
    direct_bias = mean(direct_errors)
    fm_bias = mean(fm_errors)
    records = {"adaptive_harq": [], "flowharq": []}
    for methods in samples:
        direct_prediction = (
            metric(methods["direct"], "predicted_direct_psnr_raw") + direct_bias
        )
        fm_prediction = metric(methods["fm_only"], "predicted_fm_psnr_raw") + fm_bias
        decisions = {
            "adaptive_harq": direct_prediction < target_psnr,
            "flowharq": fm_prediction < target_psnr,
        }
        first_round = {
            "adaptive_harq": methods["direct"],
            "flowharq": methods["fm_only"],
        }
        for method, nack in decisions.items():
            records[method].append(
                {
                    "nack": float(nack),
                    "psnr": select(first_round[method], methods["full_harq"], nack, "psnr"),
                    "ssim": select(first_round[method], methods["full_harq"], nack, "ssim"),
                    "lpips": select(first_round[method], methods["full_harq"], nack, "lpips"),
                }
            )
    result = {
        "source_csv": str(Path(path)),
        **metadata,
        "samples": len(samples),
        "target_psnr": target_psnr,
        "direct_quality_bias": direct_bias,
        "fm_quality_bias": fm_bias,
        "direct_calibration_mae_db": mean(abs(error - direct_bias) for error in direct_errors),
        "fm_calibration_mae_db": mean(abs(error - fm_bias) for error in fm_errors),
    }
    for method, rows in records.items():
        result[method] = {
            name: mean(row[name] for row in rows)
            for name in ("nack", "psnr", "ssim", "lpips")
        }
    result["retransmission_reduction"] = (
        result["adaptive_harq"]["nack"] - result["flowharq"]["nack"]
    )
    result["psnr_delta_db"] = (
        result["flowharq"]["psnr"] - result["adaptive_harq"]["psnr"]
    )
    result["lpips_delta"] = (
        result["flowharq"]["lpips"] - result["adaptive_harq"]["lpips"]
    )
    return result


def main():
    args = parse_args()
    if args.reference_config:
        reference = json.loads(Path(args.reference_config).read_text())["selected"]
        target_psnrs = [float(reference["target_psnr"])]
    elif args.target_psnrs:
        target_psnrs = [float(value) for value in args.target_psnrs.split(",")]
    else:
        target_psnrs = [args.target_psnr]
    candidates = [
        evaluate(path, target_psnr)
        for path in args.csv
        for target_psnr in target_psnrs
    ]
    feasible = [
        row
        for row in candidates
        if row["psnr_delta_db"] >= -args.psnr_tolerance
        and row["lpips_delta"] <= args.lpips_tolerance
    ]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["retransmission_reduction"],
            row["psnr_delta_db"],
            -row["lpips_delta"],
            -row["flow_steps"],
        ),
    )
    output = {
        "protocol": {
            "selection_split": "calibration",
            "target_psnrs": target_psnrs,
            "psnr_tolerance_db": args.psnr_tolerance,
            "lpips_tolerance": args.lpips_tolerance,
            "feasible_candidates": len(feasible),
            "total_candidates": len(candidates),
        },
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda row: (row["flow_steps"], row["mask_threshold"], row["target_psnr"]),
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
