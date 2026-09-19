"""Stage A: train independent enhancement packets with a frozen author base codec."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .codec import PublishedSwin, EnhancementCodec
from .data import Images, paths_at, seed_all, seed_worker, sha256
from .physical import packet_awgn, enhancement_awgn, subset_mask


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def validate(base, codec, loader, device):
    codec.eval()
    sums = np.zeros(5)
    count = 0
    # fork_rng preserves the training RNG; fixed validation channel draws.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        torch.manual_seed(11007)
        for images, _ in loader:
            images = images.to(device)
            z = base.encode(images)
            y, _ = packet_awgn(z, 10.)
            e = codec.encode(images, z)
            ye, _ = enhancement_awgn(e, 10.)
            for k, code in enumerate((0, 1, 3, 7, 15)):
                mask = subset_mask(torch.full((images.shape[0],), code, device=device))
                recon = base.decode(codec.fuse(y, ye, mask, 10.))
                sums[k] += (images-recon).square().flatten(1).mean(1).sum().item()
            count += images.shape[0]
    codec.train()
    return {f"mse_k{k}": float(sums[k]/count) for k in range(5)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--valid-data", required=True)
    p.add_argument("--test-data", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--minimum-steps", type=int, default=600)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--bounded-evidence", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    previous_excepthook = sys.excepthook
    def record_failure(kind, value, traceback):
        if args.output.exists():
            status_path=args.output/"status.json"
            failure=json.loads(status_path.read_text()) if status_path.exists() else {}
            failure.update(status="failed",error_type=kind.__name__,error=str(value))
            save_json(status_path,failure)
        previous_excepthook(kind,value,traceback)
    sys.excepthook=record_failure
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths, valid_paths, test_paths = map(paths_at, (args.train_data, args.valid_data, args.test_data))
    if len(valid_paths) != 100 or len(test_paths) != 24:
        raise ValueError("Preregistered data require DIV2K valid=100 and Kodak=24")
    splits = dict(train=train_paths, tuning=valid_paths[:20], oracle_gate=valid_paths[20:60],
                  risk_calibration=valid_paths[60:], test=test_paths)
    if (args.output / "configuration.json").exists() and not args.resume:
        raise FileExistsError("Use --resume or a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_json(args.output / "configuration.json", {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        manifest = {name: [dict(path=str(path), name=path.name, sha256=sha256(path)) for path in paths]
                    for name, paths in splits.items()}
        # Validate actual content, not just filenames. Test images may never enter training.
        previous = set()
        for name, records in manifest.items():
            hashes = {r["sha256"] for r in records}
            if previous & hashes:
                raise ValueError(f"content overlap detected in {name}")
            previous |= hashes
        save_json(args.output / "split_manifest.json", manifest)
        save_json(args.output / "provenance.json", dict(
            torch=torch.__version__, python=platform.python_version(), host=platform.node(),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
            base_checkpoint_sha256=sha256(args.base_checkpoint),
            source_sha256={str(f): sha256(f) for f in sorted(Path("evidence_harq").glob("*.py"))},
            upstream_commit=subprocess.check_output(["git", "-C", args.upstream, "rev-parse", "HEAD"], text=True).strip(),
            status="smoke_only" if args.smoke else "running"))
        (args.output / "pip-freeze.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    base = PublishedSwin(args.upstream, args.base_checkpoint).to(device)
    codec = EnhancementCodec(bounded_evidence=args.bounded_evidence).to(device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=1e-4)
    train_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Images(train_paths, True), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=seed_worker, generator=train_gen)
    tuning = DataLoader(Images(splits["tuning"]), batch_size=4, num_workers=2)
    start_step, best, stale = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.output/"last.pt", map_location="cpu", weights_only=False)
        codec.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best, stale = checkpoint["step"], checkpoint["best"], checkpoint["stale"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        train_gen.set_state(checkpoint["loader_rng"])
        # Worker prefetch state is not serializable; resumed crops/order are not bitwise identical.
    history_path = args.output / "training.csv"
    started = time.time()
    iterator = iter(loader)
    with history_path.open("a" if args.resume else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "grad_norm", "lr", "elapsed_s"])
        if not args.resume:
            writer.writeheader()
        for step in range(start_step+1, args.steps+1):
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _ = next(iterator)
            images = images.to(device, non_blocking=True)
            # Primary AWGN=10dB; 0/5/15dB are held-out channel robustness evaluations.
            snr = 10.
            with torch.no_grad():
                z = base.encode(images)
                y, _ = packet_awgn(z, snr)
            lr = args.lr * min(1., step/50) * (.1 + .9 * .5 * (1 + np.cos(np.pi*step/args.steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            codes = torch.randint(1, 16, (images.shape[0],), device=device)
            codes[torch.rand(images.shape[0], device=device) < .25] = 15
            mask = subset_mask(codes)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and args.precision == "bf16"):
                e = codec.encode(images, z)
                ye, _ = enhancement_awgn(e, snr)
                reconstruction = base.decode(codec.fuse(y, ye, mask, snr))
                loss = (images.float()-reconstruction.float()).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite enhancement training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            writer.writerow(dict(step=step, loss=loss.item(), grad_norm=float(norm), lr=lr, elapsed_s=time.time()-started))
            handle.flush()
            if step == 1 or step % 25 == 0:
                print(json.dumps(dict(stage="codec", step=step, steps=args.steps, loss=loss.item(), elapsed_s=time.time()-started)), flush=True)
            if step % args.validate_every == 0 or step == args.steps:
                metrics = validate(base, codec, tuning, device)
                objective = np.mean([metrics[f"mse_k{k}"] for k in range(1, 5)])
                improved = objective < best - 1e-6
                stale = 0 if improved else stale+1
                best = min(best, objective)
                state = dict(model=codec.state_dict(), optimizer=optimizer.state_dict(), step=step,
                             best=float(best), stale=stale, metrics=metrics, args=vars(args),
                             python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                             loader_rng=train_gen.get_state())
                for filename in (["last.pt", "best.pt"] if improved else ["last.pt"]):
                    torch.save(state, args.output/(filename+".tmp"))
                    (args.output/(filename+".tmp")).replace(args.output/filename)
                status = dict(stage="codec", status="running", step=step, steps=args.steps,
                              best_validation_mse=best, **metrics, elapsed_s=time.time()-started)
                save_json(args.output/"status.json", status)
                print(json.dumps(status), flush=True)
                if step >= args.minimum_steps and stale >= args.patience:
                    break
    status.update(status="smoke_only" if args.smoke else "completed", stopping_step=step,
                  best_checkpoint_sha256=sha256(args.output/"best.pt"))
    save_json(args.output/"status.json", status)


if __name__ == "__main__":
    main()
