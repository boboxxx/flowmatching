from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "direct": "DeepJSCC",
    "full_harq": "Full HARQ",
    "fm_only": "DeepJSCC + RAFM",
    "adaptive_harq": "Adaptive HARQ",
    "flowharq": "FlowHARQ",
}
COLORS = {
    "direct": "#6b7280",
    "full_harq": "#2563eb",
    "fm_only": "#7c3aed",
    "adaptive_harq": "#059669",
    "flowharq": "#dc2626",
}
MARKERS = {
    "direct": "o",
    "full_harq": "s",
    "fm_only": "^",
    "adaptive_harq": "D",
    "flowharq": "*",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("snr_db", "speed_kmh", "psnr", "nack", "transmission_rounds"):
            row[key] = float(row[key])
    return rows


def aggregate(rows: list[dict], x_key: str, y_key: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row[x_key])].append(row[y_key])
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
    }


def line_plot(rows, x_key, y_key, xlabel, ylabel, output, methods):
    aggregate_values = aggregate(rows, x_key, y_key)
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    for method in methods:
        points = sorted(
            (x, value)
            for (candidate, x), value in aggregate_values.items()
            if candidate == method
        )
        if not points:
            continue
        x, y = zip(*points)
        axis.plot(
            x,
            y,
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=6,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    line_plot(
        rows,
        "snr_db",
        "psnr",
        "SNR (dB)",
        "PSNR (dB)",
        output / "psnr_vs_snr",
        tuple(LABELS),
    )
    line_plot(
        rows,
        "snr_db",
        "nack",
        "SNR (dB)",
        "Physical retransmission rate",
        output / "retransmission_rate_vs_snr",
        ("full_harq", "adaptive_harq", "flowharq"),
    )
    line_plot(
        rows,
        "speed_kmh",
        "nack",
        "Vehicle speed (km/h)",
        "Physical retransmission rate",
        output / "retransmission_rate_vs_speed",
        ("full_harq", "adaptive_harq", "flowharq"),
    )
    print(f"wrote PDF and PNG figures to {output}")


if __name__ == "__main__":
    main()

