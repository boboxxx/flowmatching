from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-min", type=float, default=18.0)
    parser.add_argument("--threshold-max", type=float, default=30.0)
    parser.add_argument("--threshold-step", type=float, default=0.25)
    return parser.parse_args()


def load_samples(path: str):
    grouped = defaultdict(dict)
    predictions = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["image"], float(row["snr_db"]), float(row["speed_kmh"]), float(row["csi_age_ms"]))
            grouped[key][row["method"]] = float(row["psnr"])
            predictions[key] = (
                float(row["predicted_direct_psnr"]),
                float(row["predicted_fm_psnr"]),
            )
    samples = []
    for key, methods in grouped.items():
        required = {"direct", "full_harq", "fm_only"}
        if not required.issubset(methods):
            raise ValueError(f"incomplete method set for {key}")
        samples.append((key, methods, predictions[key]))
    return samples


def main():
    args = parse_args()
    samples = load_samples(args.csv)
    thresholds = []
    value = args.threshold_min
    while value <= args.threshold_max + 1e-9:
        thresholds.append(round(value, 8))
        value += args.threshold_step
    snrs = sorted({key[1] for key, _, _ in samples})
    rows = []
    for snr in snrs:
        selected = [sample for sample in samples if sample[0][1] == snr]
        for threshold in thresholds:
            for method in ("adaptive_harq", "flowharq"):
                output_psnr = []
                nacks = []
                for _, values, prediction in selected:
                    if method == "adaptive_harq":
                        nack = prediction[0] < threshold
                        first_round = values["direct"]
                    else:
                        nack = prediction[1] < threshold
                        first_round = values["fm_only"]
                    output_psnr.append(values["full_harq"] if nack else first_round)
                    nacks.append(float(nack))
                rows.append(
                    {
                        "snr_db": snr,
                        "method": method,
                        "quality_threshold_db": threshold,
                        "psnr": sum(output_psnr) / len(output_psnr),
                        "retransmission_rate": sum(nacks) / len(nacks),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), sharex=False, sharey=False)
    for axis, snr in zip(axes.flat, snrs):
        for method, label, color, marker in (
            ("adaptive_harq", "Adaptive HARQ", "#059669", "D"),
            ("flowharq", "FlowHARQ", "#dc2626", "o"),
        ):
            points = [row for row in rows if row["snr_db"] == snr and row["method"] == method]
            axis.plot(
                [row["retransmission_rate"] for row in points],
                [row["psnr"] for row in points],
                color=color,
                marker=marker,
                markevery=max(1, len(points) // 8),
                markersize=4,
                linewidth=1.5,
                label=label,
            )
        axis.set_title(f"SNR = {snr:g} dB")
        axis.set_xlabel("Retransmission rate")
        axis.set_ylabel("PSNR (dB)")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {len(rows)} frontier rows and paired figures")


if __name__ == "__main__":
    main()

