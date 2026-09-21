import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .channel import sample_context
from .data import ImageDataset
from .metrics import psnr, spearman_per_sample
from .model import UGPDeepJSCC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("backbone", "uncertainty", "joint"), default="joint")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def set_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad = True
    if stage == "backbone":
        for parameter in model.protection.parameters():
            parameter.requires_grad = False
    elif stage == "uncertainty":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.protection.uncertainty.parameters():
            parameter.requires_grad = True


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ImageDataset(args.data, train=True, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = UGPDeepJSCC(args.upstream, latent_dim=args.latent_dim).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
    set_stage(model, args.stage)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    log_path = output / "metrics.jsonl"
    for epoch in range(args.epochs):
        model.train()
        for image, _ in loader:
            image = image.to(device, non_blocking=True)
            context = sample_context(image.shape[0], device)
            allocation_mode = "full" if args.stage == "joint" else "equal"
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                reconstruction, _, allocation = model(image, context, mode=allocation_mode)
                losses = model.loss(image, reconstruction, allocation)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            correlation = spearman_per_sample(
                allocation["uncertainty"].detach(), losses["local_error"].detach()
            ).mean()
            record = {
                "epoch": epoch,
                "step": step,
                "loss": float(losses["total"].detach()),
                "l1": float(losses["l1"].detach()),
                "uncertainty_nll": float(losses["uncertainty_nll"].detach()),
                "psnr": float(psnr(image, reconstruction).mean().detach()),
                "spearman": float(correlation),
                "power_mean": float(allocation["power"].mean().detach()),
                "power_min": float(allocation["power"].min().detach()),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
            output / "checkpoint.pt",
        )
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
