"""Evaluate author networks; shared RGB metrics plus separately named native metrics.

One process per upstream avoids their colliding `net`, `loss`, and `utils` imports.
Weights must come from the recorded author download manifest, not arbitrary pickle files.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def load_author_state(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Author-generated resolution buffers are rebuilt, never skip trainable weights.
    skip = lambda key: "attn_mask" in key or "rate_adaption.mask" in key
    state = {k: v for k, v in state.items() if not skip(k)}
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not skip(k) for k in result.missing_keys):
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    return list(result.missing_keys)


def build(args, manifest):
    sys.path.insert(0, str(args.upstream.resolve()))
    if args.family == "swin":
        from net.network import SwinJSCC
        from ugp.model import small_swin_kwargs
        enc, dec = small_swin_kwargs(manifest["channels"])
        enc["depths"], dec["depths"] = [2, 2, 6, 2], [2, 6, 2, 2]
        model_args = SimpleNamespace(model="SwinJSCC_w/o_SAandRA", C=str(manifest["channels"]),
                                     multiple_snr="10", channel_type="awgn",
                                     distortion_metric=manifest["objective"], trainset="DIV2K")
        config = SimpleNamespace(encoder_kwargs=enc, decoder_kwargs=dec, logger=None,
                                 pass_channel=True, downsample=4, norm=False,
                                 device="cuda", CUDA=True)
        model = SwinJSCC(model_args, config)
    else:
        from config import config
        from net.NTSCC_Hyperior import NTSCC_Hyperprior
        config.device = "cuda"
        config.use_side_info = False
        config.eta = args.eta
        model = NTSCC_Hyperprior(config)
    missing = load_author_state(model, args.checkpoint)
    # Native Swin/NTSCC code has a nonstandard 4-level MS-SSIM expression.
    # Keep it only for author-code cross-checking; never substitute for common 5-scale MS-SSIM.
    from loss.distortion import MS_SSIM
    native_metric = MS_SSIM(data_range=1., levels=4, channel=3).cuda().eval()
    return model.cuda().eval(), native_metric, missing


def main():
    # Model construction is also reused with the historical frozen receiver,
    # whose metrics module predates these additional evaluation functions.
    from flowharq.metrics import mse, psnr, ms_ssim, ms_ssim_db
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["swin", "ntscc"], required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop", type=int, default=0, help="0: native full resolution, otherwise center crop")
    p.add_argument("--snrs", nargs="+", type=float, default=[10.])
    p.add_argument("--draws", type=int, default=10, help="Channel draws, not independent training seeds")
    p.add_argument("--seed", type=int, default=7070)
    p.add_argument("--limit", type=int, default=0, help="Smoke test only; zero uses all images")
    p.add_argument("--eta", type=float, default=.2)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    manifest = json.loads(args.checkpoint.with_suffix(".json").read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Downloaded checkpoint hash mismatch")
    model, native_metric, missing = build(args, manifest)
    import lpips
    perceptual = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
    paths = sorted(x for x in args.data.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No test images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace prior results: {args.output}")
    records = []
    with args.output.open("w") as handle, torch.inference_mode():
        writer = None
        for snr in args.snrs:
            for image_id, path in enumerate(paths):
                arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.
                ref = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
                if args.crop:
                    h, w = ref.shape[-2:]
                    if min(h, w) < args.crop:
                        raise ValueError("Crop would require resizing/padding")
                    top, left = (h-args.crop)//2, (w-args.crop)//2
                    ref = ref[:, :, top:top+args.crop, left:left+args.crop]
                if any(x % 128 for x in ref.shape[-2:]):
                    raise ValueError("Author models require dimensions divisible by 128")
                for draw in range(args.draws):
                    # Paired per image/SNR/draw across models, independent of model initialization.
                    noise_seed = args.seed + image_id*10000 + round(snr*100)*100000 + draw
                    torch.manual_seed(noise_seed)
                    if args.family == "swin":
                        reconstruction, cbr, _, _, _ = model(ref, given_SNR=snr)
                        rate_bits, rate_cbr = 0., 0.
                    else:
                        model.channel.chan_param = snr
                        result = model(ref)
                        cbr, reconstruction = result[4], result[6]
                        rate_bits = (ref.shape[-2]//16)*(ref.shape[-1]//16)*4
                        # Same capacity-achieving side-channel assumption as author test().
                        rate_cbr = rate_bits / math.log2(1+10**(snr/10)) / ref.numel()
                    reconstruction = reconstruction.clamp(0, 1)
                    score = ms_ssim(ref, reconstruction)
                    row = dict(model=manifest["name"], objective=manifest["objective"],
                               image=path.name, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               height=ref.shape[-2], width=ref.shape[-1], snr_db=snr, draw=draw,
                               noise_seed=noise_seed, cbr_payload=float(cbr), rate_map_bits=rate_bits,
                               cbr_rate_map_capacity=rate_cbr, cbr_with_rate_map_capacity=float(cbr)+rate_cbr,
                               mse=mse(ref, reconstruction).item(), psnr=psnr(ref, reconstruction).item(),
                               ms_ssim=score.item(), ms_ssim_db=ms_ssim_db(score).item(),
                               lpips=perceptual(ref*2-1, reconstruction*2-1).item(),
                               native_ms_ssim_4level=(1-native_metric(ref, reconstruction)).mean().item(),
                               psnr_8bit=psnr((ref*255).round()/255, (reconstruction*255).round()/255).item())
                    if not all(math.isfinite(row[k]) for k in ["mse", "psnr", "ms_ssim", "lpips"]):
                        raise FloatingPointError("Non-finite quality metric")
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                    records.append(row)
                handle.flush()
                print(json.dumps(dict(model=manifest["name"], snr=snr, image=path.name,
                                      completed_rows=len(records))), flush=True)
    keys = ["cbr_payload", "cbr_rate_map_capacity", "cbr_with_rate_map_capacity", "mse", "psnr",
            "ms_ssim", "ms_ssim_db", "lpips", "native_ms_ssim_4level", "psnr_8bit"]
    summary = []
    for snr in args.snrs:
        group = [r for r in records if r["snr_db"] == snr]
        summary.append(dict(snr_db=snr, images=len(paths), draws=args.draws,
                            **{k: float(np.mean([r[k] for r in group])) for k in keys}))
    meta = dict(checkpoint=manifest, configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                upstream_commit=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
                regenerated_buffers=missing, status="smoke_only" if args.limit else "completed",
                normalization_side_information="Author oracle global scale retained; cost not modelled",
                metrics="RGB float [0,1], per-image/draw then arithmetic average; standard five-scale MS-SSIM; AlexNet LPIPS v0.1",
                summary=summary)
    args.output.with_suffix(".summary.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
