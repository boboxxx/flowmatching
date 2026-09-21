from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ugp.channel import ChannelContext, temporal_correlation
from ugp.data import ImageDataset

from .metrics import psnr, ssim
from .model import FlowHARQJSCC
from .perceptual import LPIPSMetric


METHODS = (
    "direct",
    "full_harq",
    "fm_only",
    "adaptive_harq",
    "flowharq",
    "oracle_adaptive",
    "oracle_flowharq",
)


def parse_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", type=parse_list, default=parse_list("0,3,6,9,12,15"))
    parser.add_argument("--speeds", type=parse_list, default=parse_list("30,60,90,120"))
    parser.add_argument("--age-ms", type=float, default=2.0)
    parser.add_argument("--harq-interval-ms", type=float, default=1.0)
    parser.add_argument("--target-psnr", type=float, default=24.0)
    parser.add_argument("--direct-quality-bias", type=float, default=0.0)
    parser.add_argument("--fm-quality-bias", type=float, default=0.0)
    parser.add_argument(
        "--calibration-json",
        help="Frozen calibration output; overrides flow steps, mask threshold, biases, and target.",
    )
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=("all", "calibration", "test"), default="test")
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--vendor", default="vendor")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def choose(first: torch.Tensor, second: torch.Tensor, nack: torch.Tensor) -> torch.Tensor:
    return torch.where(nack[:, None, None, None], second, first)


def build_dataset(args) -> ImageDataset:
    dataset = ImageDataset(args.data, train=False, limit=0)
    if not 0 < args.calibration_count < len(dataset.paths):
        raise ValueError("calibration-count must leave non-empty calibration and test splits")
    if args.split == "calibration":
        dataset.paths = dataset.paths[: args.calibration_count]
    elif args.split == "test":
        dataset.paths = dataset.paths[args.calibration_count :]
    if args.limit:
        dataset.paths = dataset.paths[: args.limit]
    return dataset


def main():
    args = parse_args()
    if args.calibration_json:
        calibration = json.loads(Path(args.calibration_json).read_text())
        selected = calibration["selected"]
        args.flow_steps = int(selected["flow_steps"])
        args.mask_threshold = float(selected["mask_threshold"])
        args.direct_quality_bias = float(selected["direct_quality_bias"])
        args.fm_quality_bias = float(selected["fm_quality_bias"])
        args.target_psnr = float(selected["target_psnr"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu")
    saved_args = payload.get("args", {})
    training_seed = int(saved_args.get("seed", -1))
    model = FlowHARQJSCC(
        args.upstream,
        latent_dim=int(saved_args.get("latent_dim", 32)),
        mask_fraction=float(saved_args.get("mask_fraction", 0.35)),
        flow_hidden_dim=int(saved_args.get("flow_hidden_dim", 128)),
        flow_depth=int(saved_args.get("flow_depth", 2)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    integration_time_mode = "zero" if saved_args.get("fm_time") == "zero" else "midpoint"
    perceptual = LPIPSMetric(args.vendor).to(device) if args.lpips else None
    loader = DataLoader(
        build_dataset(args),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.no_grad():
        for snr_db in args.snrs:
            for speed_kmh in args.speeds:
                for image, names in loader:
                    image = image.to(device, non_blocking=True)
                    batch = image.shape[0]
                    snr = torch.full((batch,), snr_db, device=device)
                    speed = torch.full((batch,), speed_kmh, device=device)
                    age = torch.full((batch,), args.age_ms, device=device)
                    context = ChannelContext(snr, speed, age, temporal_correlation(speed, age))
                    clean = model.encode(image, context)
                    first_observation = model.transmit(clean, context)
                    second_observation = model.transmit(
                        clean,
                        context,
                        previous=first_observation,
                        round_gap_ms=args.harq_interval_ms,
                    )
                    first = first_observation.tokens
                    combined = model.combine(first_observation, second_observation)

                    direct = model.decode(first, context)
                    full_harq = model.decode(combined, context)
                    repaired, probabilities, mask = model.repair(
                        first_observation,
                        context,
                        steps=args.flow_steps,
                        threshold=args.mask_threshold,
                        integration_time_mode=integration_time_mode,
                    )
                    fm_only = model.decode(repaired, context)
                    receiver_context = first_observation.receiver_features(context)
                    predicted_direct_raw = model.quality.log_mse_to_psnr(
                        model.quality(first, probabilities, receiver_context)
                    )
                    predicted_fm_raw = model.quality.log_mse_to_psnr(
                        model.quality(repaired, probabilities, receiver_context)
                    )
                    predicted_direct = predicted_direct_raw + args.direct_quality_bias
                    predicted_fm = predicted_fm_raw + args.fm_quality_bias
                    adaptive_nack = predicted_direct < args.target_psnr
                    flow_nack = predicted_fm < args.target_psnr
                    direct_psnr = psnr(image, direct)
                    fm_psnr = psnr(image, fm_only)
                    oracle_adaptive_nack = direct_psnr < args.target_psnr
                    oracle_flow_nack = fm_psnr < args.target_psnr
                    reconstructions = {
                        "direct": direct,
                        "full_harq": full_harq,
                        "fm_only": fm_only,
                        "adaptive_harq": choose(direct, full_harq, adaptive_nack),
                        "flowharq": choose(fm_only, full_harq, flow_nack),
                        "oracle_adaptive": choose(direct, full_harq, oracle_adaptive_nack),
                        "oracle_flowharq": choose(fm_only, full_harq, oracle_flow_nack),
                    }
                    nacks = {
                        "direct": torch.zeros_like(flow_nack),
                        "full_harq": torch.ones_like(flow_nack),
                        "fm_only": torch.zeros_like(flow_nack),
                        "adaptive_harq": adaptive_nack,
                        "flowharq": flow_nack,
                        "oracle_adaptive": oracle_adaptive_nack,
                        "oracle_flowharq": oracle_flow_nack,
                    }
                    if perceptual is not None:
                        stacked_reconstructions = torch.cat(
                            [reconstructions[method] for method in METHODS], dim=0
                        )
                        stacked_references = image.repeat(len(METHODS), 1, 1, 1)
                        perceptual_values = perceptual(
                            stacked_references, stacked_reconstructions
                        ).reshape(len(METHODS), batch)
                    else:
                        perceptual_values = torch.full(
                            (len(METHODS), batch),
                            float("nan"),
                            device=image.device,
                        )
                    for method_index, method in enumerate(METHODS):
                        values = psnr(image, reconstructions[method])
                        ssim_values = ssim(image, reconstructions[method])
                        for index, name in enumerate(names):
                            rows.append(
                                {
                                    "image": name,
                                    "method": method,
                                    "snr_db": snr_db,
                                    "speed_kmh": speed_kmh,
                                    "csi_age_ms": args.age_ms,
                                    "harq_interval_ms": args.harq_interval_ms,
                                    "round_correlation": float(
                                        temporal_correlation(speed, torch.full_like(speed, args.harq_interval_ms))[index]
                                    ),
                                    "first_gain_sq": float(first_observation.gain_sq[index].mean()),
                                    "second_gain_sq": float(second_observation.gain_sq[index].mean()),
                                    "effective_snr_db": float(first_observation.effective_snr_db[index]),
                                    "psnr": float(values[index]),
                                    "ssim": float(ssim_values[index]),
                                    "lpips": float(perceptual_values[method_index, index]),
                                    "nack": int(nacks[method][index]),
                                    "transmission_rounds": 1 + int(nacks[method][index]),
                                    "predicted_direct_psnr": float(predicted_direct[index]),
                                    "predicted_fm_psnr": float(predicted_fm[index]),
                                    "predicted_direct_psnr_raw": float(predicted_direct_raw[index]),
                                    "predicted_fm_psnr_raw": float(predicted_fm_raw[index]),
                                    "mask_rate": float(mask[index].mean()),
                                    "flow_steps": args.flow_steps,
                                    "mask_threshold": args.mask_threshold,
                                    "target_psnr": args.target_psnr,
                                    "direct_quality_bias": args.direct_quality_bias,
                                    "fm_quality_bias": args.fm_quality_bias,
                                    "calibration_json": args.calibration_json or "",
                                    "split": args.split,
                                    "training_seed": training_seed,
                                    "channel_seed": args.seed,
                                }
                            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for snr_db in args.snrs:
        summary[str(snr_db)] = {}
        for method in METHODS:
            selected = [r for r in rows if r["snr_db"] == snr_db and r["method"] == method]
            summary[str(snr_db)][method] = {
                "psnr": sum(r["psnr"] for r in selected) / len(selected),
                "ssim": sum(r["ssim"] for r in selected) / len(selected),
                "lpips": sum(r["lpips"] for r in selected) / len(selected),
                "retransmission_rate": sum(r["nack"] for r in selected) / len(selected),
                "transmission_rounds": sum(r["transmission_rounds"] for r in selected) / len(selected),
            }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "csv": str(output), "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
