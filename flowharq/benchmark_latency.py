from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from ugp.channel import ChannelContext, temporal_correlation
from ugp.data import ImageDataset

from .model import FlowHARQJSCC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="upstream/SwinJSCC")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--speed", type=float, default=90.0)
    parser.add_argument("--age-ms", type=float, default=2.0)
    parser.add_argument("--harq-interval-ms", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-json")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=200)
    return parser.parse_args()


def measure_cuda(function, warmup: int, repetitions: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.stdev(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "repetitions": repetitions,
    }


def main():
    args = parse_args()
    if args.calibration_json:
        selected = json.loads(Path(args.calibration_json).read_text())["selected"]
        args.flow_steps = int(selected["flow_steps"])
        args.mask_threshold = float(selected["mask_threshold"])
    if not torch.cuda.is_available():
        raise RuntimeError("latency benchmark requires CUDA events")
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu")
    saved_args = payload.get("args", {})
    model = FlowHARQJSCC(
        args.upstream,
        latent_dim=int(saved_args.get("latent_dim", 32)),
        mask_fraction=float(saved_args.get("mask_fraction", 0.35)),
        flow_hidden_dim=int(saved_args.get("flow_hidden_dim", 128)),
        flow_depth=int(saved_args.get("flow_depth", 2)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    image, _ = ImageDataset(args.data, train=False, limit=1)[0]
    image = image[None].to(device)
    snr = torch.tensor([args.snr], device=device)
    speed = torch.tensor([args.speed], device=device)
    age = torch.tensor([args.age_ms], device=device)
    context = ChannelContext(snr, speed, age, temporal_correlation(speed, age))
    with torch.no_grad():
        clean = model.encode(image, context)
        first = model.transmit(clean, context)
        second = model.transmit(
            clean, context, previous=first, round_gap_ms=args.harq_interval_ms
        )
        combined = model.combine(first, second)

        def direct_decode():
            return model.decode(first.tokens, context)

        def virtual_retransmission():
            repaired, probabilities, _ = model.repair(
                first,
                context,
                steps=args.flow_steps,
                threshold=args.mask_threshold,
            )
            reconstruction = model.decode(repaired, context)
            predicted = model.quality(
                repaired, probabilities, first.receiver_features(context)
            )
            return reconstruction, predicted

        def physical_harq_receiver():
            return model.decode(model.combine(first, second), context)

        results = {
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": 1,
            "image_size": list(image.shape[-2:]),
            "snr_db": args.snr,
            "speed_kmh": args.speed,
            "flow_steps": args.flow_steps,
            "mask_threshold": args.mask_threshold,
            "direct_decode": measure_cuda(direct_decode, args.warmup, args.repetitions),
            "virtual_retransmission_and_decode": measure_cuda(
                virtual_retransmission, args.warmup, args.repetitions
            ),
            "physical_harq_combine_and_decode": measure_cuda(
                physical_harq_receiver, args.warmup, args.repetitions
            ),
        }
        # Keep the precomputed tensor live so combine timing is not optimized away.
        results["combined_checksum"] = float(combined.mean())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
