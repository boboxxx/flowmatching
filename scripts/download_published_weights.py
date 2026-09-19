"""Download only explicitly selected, publicly released author checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import gdown

WEIGHTS = {
    "swin_mse_c32": ("1h2Dk9jckeRg6Rah32S_kLq89CtPdGdwj", "SwinJSCC", "MSE", 32),
    "swin_mse_c64": ("1PMYjEjU-N3q9Q6xcCAu2jnGbLvEI3hiv", "SwinJSCC", "MSE", 64),
    "swin_msssim_c32": ("1uEO5jpxFI_krhnDt1GGLA7TzUQ0WZtps", "SwinJSCC", "MS-SSIM", 32),
    "swin_msssim_c64": ("1TDBBkukV1MoHzWqRNqFxjsHyxi3il8bO", "SwinJSCC", "MS-SSIM", 64),
    "ntscc_quality1": ("12VwWfbpVtp6GRk6Xo8u-vLy7xtb9c6lB", "NTSCC", "MSE", None),
    "ntscc_quality2": ("1xzCrD8JSPQ-4V6Sv4pTa_Shd2mCRIvv3", "NTSCC", "MSE", None),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", nargs="+", choices=WEIGHTS, default=list(WEIGHTS))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        ident, paper, objective, channels = WEIGHTS[name]
        path = args.output / (name + ".pth")
        if not path.exists():
            temporary = path.with_suffix(".download")
            result = gdown.download(id=ident, output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError(f"Author download failed: {name}")
            temporary.replace(path)
        manifest = dict(name=name, paper=paper, objective=objective, channels=channels,
                        snr_db=10, channel="AWGN", source=f"https://drive.google.com/file/d/{ident}/view",
                        bytes=path.stat().st_size, sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                        if hasattr(hashlib, "file_digest") else hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)

if __name__ == "__main__":
    main()
