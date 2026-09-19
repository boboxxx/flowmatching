"""All-subset evaluation with paired noise; never used as a deployed selector."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import torch

from flowharq.metrics import mse, psnr, ms_ssim
from flowharq.perceptual import LPIPSMetric
from .codec import PublishedSwin, EnhancementCodec
from .data import Images, sha256
from .physical import LinkBudget, packet_awgn, enhancement_awgn, subset_mask
from .train_codec import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", choices=["tuning", "oracle_gate", "risk_calibration", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--snrs", nargs="+", type=float, default=[10.])
    p.add_argument("--seed", type=int, default=8128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-subsets", type=int, default=4)
    p.add_argument("--data-root", type=Path, help="Relocate this split's images; registered basenames and hashes are still enforced")
    p.add_argument("--upstream", type=Path, help="Override historical SwinJSCC source path")
    p.add_argument("--base-checkpoint", type=Path, help="Override historical author checkpoint path")
    args = p.parse_args()
    if args.draws < 1 or args.batch_subsets < 1 or args.limit < 0:
        p.error("draws/batch-subsets must be positive and limit nonnegative")
    torch.set_num_threads(4)
    cfg = json.loads((args.run/"configuration.json").read_text())
    status = json.loads((args.run/"status.json").read_text())
    if status["status"] != "completed" and not args.limit:
        raise ValueError("full evaluation requires a completed non-smoke training run")
    manifest = json.loads((args.run/"split_manifest.json").read_text())[args.split]
    if args.limit:
        manifest = manifest[:args.limit]
    for item in manifest:
        if args.data_root is not None:
            item["path"] = str(args.data_root/Path(item["name"]).name)
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("image changed after split was registered")
    dataset = Images([Path(item["path"]) for item in manifest])
    device = torch.device("cuda")
    base = PublishedSwin(args.upstream or cfg["upstream"], args.base_checkpoint or cfg["base_checkpoint"]).to(device)
    codec = EnhancementCodec(bounded_evidence=cfg.get("bounded_evidence",False)).to(device).eval()
    if sha256(args.run/"best.pt") != status["best_checkpoint_sha256"]:
        raise ValueError("selected enhancement checkpoint hash mismatch")
    checkpoint = torch.load(args.run/"best.pt", map_location="cpu", weights_only=False)
    codec.load_state_dict(checkpoint["model"])
    perceptual = LPIPSMetric().to(device).eval()
    budget = LinkBudget()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with args.output.open("w", newline="") as handle, torch.inference_mode():
        writer = None
        for index, item in enumerate(manifest):
            image, name = dataset[index]
            image = image[None].to(device)
            z = base.encode(image)
            e = codec.encode(image, z)
            for snr in args.snrs:
                for draw in range(args.draws):
                    key = f"{args.seed}:{item['sha256']}:{snr}:{draw}"
                    noise_seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    torch.manual_seed(noise_seed)
                    y, _ = packet_awgn(z, snr)
                    ye, _ = enhancement_awgn(e, snr)
                    for first in range(0, 16, args.batch_subsets):
                        codes = torch.arange(first, min(16, first+args.batch_subsets), device=device)
                        count = len(codes)
                        mask = subset_mask(codes)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        recon = base.decode(codec.fuse(y.expand(count,-1,-1,-1), ye.expand(count,-1,-1,-1), mask, snr))
                        torch.cuda.synchronize()
                        seconds = time.perf_counter()-started
                        reference = image.expand(count,-1,-1,-1)
                        quality = dict(mse=mse(reference,recon), psnr=psnr(reference,recon),
                                       ms_ssim=ms_ssim(reference,recon), lpips=perceptual(reference,recon))
                        for j, code in enumerate(codes.tolist()):
                            k = bin(code).count("1")
                            row = dict(split=args.split,image=name,image_sha256=item["sha256"],snr_db=snr,
                                       draw=draw,noise_seed=noise_seed,subset=code,groups=k,
                                       **budget.count(k), **{key: float(value[j]) for key,value in quality.items()},
                                       batched_decoder_seconds_per_subset=seconds/count)
                            if writer is None:
                                writer = csv.DictWriter(handle,fieldnames=row)
                                writer.writeheader()
                            writer.writerow(row)
                            records += 1
                    handle.flush()
            print(json.dumps(dict(stage="all_subset_evaluation",split=args.split,image=name,
                                  images_done=index+1,images=len(manifest),rows=records)), flush=True)
    save_json(args.output.with_suffix(".json"),dict(status="smoke_only" if args.limit else "completed",
              split=args.split,images=len(manifest),draws=args.draws,snrs=args.snrs,rows=records,
              checkpoint_sha256=sha256(args.run/"best.pt"),csv_sha256=sha256(args.output),
              warning="Contains every counterfactual future observation; oracle is clairvoyant, not deployable.",
              latency_note="Batched enumeration timing only; not a deployment latency benchmark."))


if __name__ == "__main__":
    main()
