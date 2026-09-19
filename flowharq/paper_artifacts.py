from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "direct": "#666666",
    "full_harq": "#009E73",
    "adaptive_harq": "#0072B2",
    "flowharq": "#D55E00",
}
LABELS = {
    "direct": "DeepJSCC",
    "full_harq": "Full HARQ",
    "adaptive_harq": "Adaptive HARQ",
    "flowharq": "FlowHARQ",
}
MARKERS = {"direct": "o", "full_harq": "s", "adaptive_harq": "D", "flowharq": "*"}
TABLE_METHODS = (
    ("direct", "DeepJSCC"),
    ("full_harq", "Full HARQ"),
    ("fm_only", "FM only"),
    ("adaptive_harq", "Adaptive HARQ"),
    ("flowharq", "FlowHARQ"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--latency-json", required=True)
    parser.add_argument("--paper-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean", "std", "ci_low", "ci_high"):
            row[key] = float(row[key])
        row["n"] = int(row["n"])
    return rows


def row_for(rows: list[dict], metric: str, snr: str, method: str | None = None) -> dict:
    selected = [
        row
        for row in rows
        if row["metric"] == metric
        and row["snr_db"] == snr
        and (method is None or row.get("method") == method)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for metric={metric}, snr={snr}, method={method}")
    return selected[0]


def plot(estimates: list[dict], speed_estimates: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
    panels = (
        (axes[0], "psnr", ("direct", "full_harq", "adaptive_harq", "flowharq"), "PSNR (dB)"),
        (axes[1], "nack", ("full_harq", "adaptive_harq", "flowharq"), "Physical retransmission rate"),
    )
    for axis, metric, methods, ylabel in panels:
        for method in methods:
            rows = [
                row
                for row in estimates
                if row["metric"] == metric
                and row["method"] == method
                and row["snr_db"] != "all"
            ]
            rows.sort(key=lambda row: float(row["snr_db"]))
            x = [float(row["snr_db"]) for row in rows]
            center = [row["mean"] for row in rows]
            low = [row["ci_low"] for row in rows]
            high = [row["ci_high"] for row in rows]
            axis.plot(
                x,
                center,
                label=LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.35,
                markersize=4.2,
            )
            axis.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
        axis.set_xlabel("Nominal SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 3, 6, 9, 12, 15])
        axis.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
        axis.legend(frameon=False, ncol=2, loc="best", handlelength=1.8)
    axes[1].set_ylim(-0.03, 1.05)
    axis = axes[2]
    for method in ("adaptive_harq", "flowharq"):
        rows = [
            row
            for row in speed_estimates
            if row["metric"] == "nack" and row["method"] == method
        ]
        rows.sort(key=lambda row: float(row["speed_kmh"]))
        x = [float(row["speed_kmh"]) for row in rows]
        center = [row["mean"] for row in rows]
        low = [row["ci_low"] for row in rows]
        high = [row["ci_high"] for row in rows]
        axis.plot(
            x,
            center,
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.35,
            markersize=4.2,
        )
        axis.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
    axis.set_xlabel("Vehicle speed (km/h)")
    axis.set_ylabel("Physical retransmission rate")
    axis.set_xticks([30, 60, 90, 120])
    axis.set_ylim(-0.03, 1.05)
    axis.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, loc="best", handlelength=1.8)
    figure.tight_layout(w_pad=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def fmt(value: float, digits: int) -> str:
    rounded = round(value, digits)
    if abs(rounded) < 0.5 * 10 ** (-digits):
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def write_main_table(estimates: list[dict], destination: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Retx. $\downarrow$ \\",
        r"\midrule",
    ]
    for method, label in TABLE_METHODS:
        values = {
            metric: row_for(estimates, metric, "all", method)["mean"]
            for metric in ("psnr", "ssim", "lpips", "nack")
        }
        cells = (
            label,
            fmt(values["psnr"], 2),
            fmt(values["ssim"], 4),
            fmt(values["lpips"], 4),
            fmt(values["nack"], 3),
        )
        if method == "flowharq":
            cells = tuple(r"\textbf{" + cell + "}" for cell in cells)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    destination.write_text("\n".join(lines) + "\n")


def write_calibration_table(calibration: dict, destination: Path) -> None:
    selected = calibration["selected"]
    threshold = float(selected["mask_threshold"])
    rows = [
        row
        for row in calibration["candidates"]
        if abs(float(row["mask_threshold"]) - threshold) < 1e-9
        and abs(float(row["target_psnr"]) - float(selected["target_psnr"])) < 1e-9
    ]
    rows.sort(key=lambda row: int(row["flow_steps"]))
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$K$ & $\tau_r$ & $\Delta$PSNR & $\Delta$LPIPS & $\Delta$Retx. \\",
        r"\midrule",
    ]
    for row in rows:
        cells = (
            str(row["flow_steps"]),
            fmt(row["mask_threshold"], 2),
            fmt(row["psnr_delta_db"], 3),
            fmt(row["lpips_delta"], 4),
            fmt(100.0 * row["retransmission_reduction"], 2),
        )
        rendered = " & ".join(cells) + r" \\"
        if int(row["flow_steps"]) == int(selected["flow_steps"]):
            rendered = r"\bfseries " + rendered
        lines.append(rendered)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    destination.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    analysis = Path(args.analysis_dir)
    paper = Path(args.paper_dir)
    estimates = read_csv(analysis / "estimates.csv")
    speed_estimates = read_csv(analysis / "speed_estimates.csv")
    deltas = read_csv(analysis / "paired_deltas.csv")
    provenance = json.loads((analysis / "provenance.json").read_text())
    calibration = json.loads(Path(args.calibration_json).read_text())
    selected = calibration["selected"]
    latency = json.loads(Path(args.latency_json).read_text())
    plot(estimates, speed_estimates, paper / "figures" / "quality_retransmission")
    write_main_table(estimates, paper / "generated" / "main_table.tex")
    write_calibration_table(calibration, paper / "generated" / "calibration_table.tex")

    retransmission = row_for(deltas, "nack", "all")
    psnr = row_for(deltas, "psnr", "all")
    lpips = row_for(deltas, "lpips", "all")
    macros = {
        "NumTrainSeeds": len(provenance["training_seeds"]),
        "NumChannelSeeds": len(provenance["channel_seeds"]),
        "NumTestImages": provenance["test_images"],
        "SelectedFlowSteps": selected["flow_steps"],
        "SelectedMaskThreshold": fmt(selected["mask_threshold"], 2),
        "MainRetxDelta": fmt(100.0 * retransmission["mean"], 2),
        "MainRetxCILow": fmt(100.0 * retransmission["ci_low"], 2),
        "MainRetxCIHigh": fmt(100.0 * retransmission["ci_high"], 2),
        "MainPSNRDelta": fmt(psnr["mean"], 3),
        "MainPSNRCILow": fmt(psnr["ci_low"], 3),
        "MainPSNRCIHigh": fmt(psnr["ci_high"], 3),
        "MainLPIPSDelta": fmt(lpips["mean"], 4),
        "VirtualLatencyMs": fmt(
            latency["virtual_retransmission_and_decode"]["mean_ms"], 2
        ),
        "PhysicalLatencyMs": fmt(
            latency["physical_harq_combine_and_decode"]["mean_ms"], 2
        ),
    }
    lines = ["% Auto-generated from held-out evaluation artifacts."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    (paper / "generated" / "results_macros.tex").write_text("\n".join(lines) + "\n")
    (paper / "generated" / "results_manifest.json").write_text(
        json.dumps(
            {
                "macros": macros,
                "calibration": str(Path(args.calibration_json)),
                "latency": str(Path(args.latency_json)),
                "analysis": str(analysis),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    main()
