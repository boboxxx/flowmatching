from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ugp.channel import sample_context
from ugp.data import ImageDataset

from .metrics import mask_f1, psnr
from .model import FlowHARQJSCC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage", choices=("backbone", "flow", "decision", "joint"), default="flow"
    )
    parser.add_argument("--backbone", help="UGP/FlowHARQ checkpoint used to initialize encoder and decoder")
    parser.add_argument("--resume", help="full FlowHARQ checkpoint")
    parser.add_argument(
        "--initialize",
        help="load full model weights without optimizer/epoch state (for stage changes)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--mask-fraction", type=float, default=0.35)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--flow-hidden-dim", type=int, default=128)
    parser.add_argument("--flow-depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-reliability", type=float, default=0.5)
    parser.add_argument("--lambda-quality", type=float, default=0.2)
    parser.add_argument("--lambda-decision", type=float, default=0.0)
    parser.add_argument("--lambda-reconstruction", type=float, default=0.2)
    parser.add_argument("--decision-target-psnr", type=float, default=24.0)
    parser.add_argument("--decision-temperature-db", type=float, default=1.0)
    parser.add_argument(
        "--fm-loss",
        choices=("mse", "normalized_huber"),
        default="mse",
        help="velocity objective; normalized_huber is robust to ZF deep-fade outliers",
    )
    parser.add_argument(
        "--fm-time",
        choices=("uniform", "zero"),
        default="uniform",
        help="uniform trains flow matching; zero trains a one-shot residual control",
    )
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def set_trainable(model: FlowHARQJSCC, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if stage == "backbone":
        for module in (model.reliability, model.flow, model.quality):
            for parameter in module.parameters():
                parameter.requires_grad = False
    elif stage == "flow":
        for module in (model.encoder, model.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
    elif stage == "decision":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.quality.parameters():
            parameter.requires_grad = True


def save_checkpoint(path: Path, model, optimizer, args, epoch, step) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "step": step,
        },
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if args.resume and args.initialize:
        raise ValueError("--resume and --initialize are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        ImageDataset(args.data, train=True, limit=args.limit),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = FlowHARQJSCC(
        args.upstream,
        latent_dim=args.latent_dim,
        mask_fraction=args.mask_fraction,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_depth=args.flow_depth,
    ).to(device)
    if args.backbone:
        print(json.dumps({"backbone": model.load_backbone(args.backbone)}), flush=True)
    if args.initialize:
        initialization = torch.load(args.initialize, map_location="cpu")
        model.load_state_dict(initialization["model"])
        print(json.dumps({"initialize": args.initialize}), flush=True)
    start_epoch = 0
    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        step = int(payload.get("step", 0))
    set_trainable(model, args.stage)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    if args.resume and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Frozen backbone should not update BatchNorm/dropout state either.
        if args.stage == "flow":
            model.encoder.eval()
            model.decoder.eval()
        elif args.stage == "decision":
            for module in (model.encoder, model.decoder, model.reliability, model.flow):
                module.eval()
        for image, _ in loader:
            image = image.to(device, non_blocking=True)
            context = sample_context(image.shape[0], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                clean = model.encode(image, context)
                observation = model.transmit(clean, context)
                if args.stage == "backbone":
                    reconstruction = model.decode(observation.tokens, context)
                    total = torch.nn.functional.l1_loss(reconstruction, image)
                    losses = {"reconstruction": total}
                    artifacts = {"reconstruction_post": reconstruction}
                else:
                    losses, artifacts = model.flow_losses(
                        image,
                        clean,
                        observation,
                        context,
                        flow_steps=args.flow_steps,
                        fm_loss_type=args.fm_loss,
                        fm_time_mode=args.fm_time,
                        decision_target_psnr=args.decision_target_psnr,
                        decision_temperature_db=args.decision_temperature_db,
                    )
                    if args.stage == "decision":
                        total = (
                            args.lambda_quality * losses["quality"]
                            + args.lambda_decision * losses["decision"]
                        )
                    else:
                        total = (
                            args.lambda_fm * losses["fm"]
                            + args.lambda_reliability * losses["reliability"]
                            + args.lambda_quality * losses["quality"]
                            + args.lambda_decision * losses["decision"]
                            + args.lambda_reconstruction * losses["reconstruction"]
                        )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()
            record = {
                "epoch": epoch,
                "step": step,
                "stage": args.stage,
                "loss": float(total.detach()),
                "psnr_post": float(psnr(image, artifacts["reconstruction_post"]).mean().detach()),
            }
            for name, value in losses.items():
                record[name] = float(value.detach())
            if args.stage != "backbone":
                record.update(
                    {
                        "psnr_pre": float(psnr(image, artifacts["reconstruction_pre"]).mean().detach()),
                        "mask_f1": float(
                            mask_f1(artifacts["probabilities"] >= 0.5, artifacts["oracle_mask"]).mean().detach()
                        ),
                        "mask_rate": float((artifacts["probabilities"] >= 0.5).float().mean().detach()),
                    }
                )
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        save_checkpoint(output / "checkpoint.pt", model, optimizer, args, epoch, step)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
