import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .channel import ChannelContext, temporal_correlation
from .data import ImageDataset
from .metrics import psnr, spearman_per_sample
from .model import UGPDeepJSCC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "equal", "importance_only", "uncertainty_only"), default="full")
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=120.0)
    parser.add_argument("--age", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UGPDeepJSCC(args.upstream).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    loader = DataLoader(ImageDataset(args.data, train=False, limit=args.limit), batch_size=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for image, name in loader:
            image = image.to(device)
            snr = torch.full((1,), args.snr, device=device)
            speed = torch.full((1,), args.speed, device=device)
            age = torch.full((1,), args.age, device=device)
            context = ChannelContext(snr, speed, age, temporal_correlation(speed, age))
            reconstruction, _, allocation = model(image, context, mode=args.mode)
            local_error = model.local_error(image, reconstruction, allocation["uncertainty"].shape[1])
            rows.append(
                {
                    "image": name[0],
                    "mode": args.mode,
                    "snr_db": args.snr,
                    "speed_kmh": args.speed,
                    "csi_age_ms": args.age,
                    "psnr": float(psnr(image, reconstruction).item()),
                    "spearman": float(spearman_per_sample(allocation["uncertainty"], local_error).item()),
                    "power_mean": float(allocation["power"].mean().item()),
                }
            )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

