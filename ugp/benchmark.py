import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .channel import ChannelContext, temporal_correlation
from .data import ImageDataset
from .metrics import psnr, spearman_per_sample
from .model import UGPDeepJSCC


MODES = ("equal", "importance_only", "uncertainty_only", "full")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def conditions():
    result = []
    for speed in (0.0, 60.0, 120.0):
        result.append(("mobility", 5.0, speed, 2.0))
    for age in (0.0, 1.0, 2.0, 5.0):
        condition = ("csi_age", 5.0, 120.0, age)
        if condition not in result:
            result.append(condition)
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UGPDeepJSCC(args.upstream).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    loader = DataLoader(ImageDataset(args.data, train=False, limit=args.limit), batch_size=1)
    images = [(image.to(device), name[0]) for image, name in loader]
    rows = []
    with torch.no_grad():
        for condition_index, (sweep, snr_value, speed_value, age_value) in enumerate(conditions()):
            snr = torch.tensor([snr_value], device=device)
            speed = torch.tensor([speed_value], device=device)
            age = torch.tensor([age_value], device=device)
            context = ChannelContext(snr, speed, age, temporal_correlation(speed, age))
            for mode in MODES:
                for image_index, (image, name) in enumerate(images):
                    # Identical image/channel realization across ablation modes.
                    torch.manual_seed(args.seed + condition_index * 1_000_003 + image_index)
                    reconstruction, _, allocation = model(image, context, mode=mode)
                    error = model.local_error(
                        image, reconstruction, allocation["uncertainty"].shape[1]
                    )
                    rows.append(
                        {
                            "sweep": sweep,
                            "mode": mode,
                            "image": name,
                            "snr_db": snr_value,
                            "speed_kmh": speed_value,
                            "csi_age_ms": age_value,
                            "psnr": float(psnr(image, reconstruction).item()),
                            "spearman": float(
                                spearman_per_sample(allocation["uncertainty"], error).item()
                            ),
                            "power_min": float(allocation["power"].min().item()),
                            "power_max": float(allocation["power"].max().item()),
                            "power_mean": float(allocation["power"].mean().item()),
                        }
                    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (
            row["sweep"],
            row["mode"],
            row["snr_db"],
            row["speed_kmh"],
            row["csi_age_ms"],
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        summary.append(
            {
                "sweep": key[0],
                "mode": key[1],
                "snr_db": key[2],
                "speed_kmh": key[3],
                "csi_age_ms": key[4],
                "images": len(values),
                "psnr_mean": sum(x["psnr"] for x in values) / len(values),
                "spearman_mean": sum(x["spearman"] for x in values) / len(values),
                "power_mean": sum(x["power_mean"] for x in values) / len(values),
                "power_min": min(x["power_min"] for x in values),
                "power_max": max(x["power_max"] for x in values),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

