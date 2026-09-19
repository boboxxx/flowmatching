"""Released SwinJSCC/NTSCC weights with quantized, charged side information."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evidence_harq.data import Images, paths_at, sha256
from evidence_harq.train_codec import save_json
from flowharq.metrics import mse, psnr, ms_ssim
from flowharq.perceptual import LPIPSMetric
from evaluate_published_checkpoint import build


def install_quantized_scale(model,family):
    original=model.channel.forward
    counted={"scales":0}
    def forward(channel,values,*args,**kwargs):
        if family=="swin":
            supplied=kwargs.get("avg_pwr",args[1] if len(args)>1 else False)
            power=supplied if supplied is not False and supplied is not None else values.square().mean()
        else:
            supplied=kwargs.get("avg_pwr",args[0] if args else None)
            power=values.square().mean() if supplied is None else supplied
        raw_scale=torch.sqrt(torch.as_tensor(power,device=values.device)*2)
        quantized=raw_scale.half().float()
        if not torch.isfinite(quantized).all() or (quantized<=0).any():
            raise FloatingPointError("unrepresentable author packet scale")
        result=original(values,*args,**kwargs)
        counted["scales"]+=quantized.numel()
        # Algebraically identical to multiplying the normalized channel output
        # by Q16(scale) instead of the author's unquantized scale. Only Q16 reaches D.
        if isinstance(result,tuple):
            return (result[0]*(quantized/raw_scale),*result[1:])
        return result*(quantized/raw_scale)
    model.channel.forward=types.MethodType(forward,model.channel)
    return counted


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=["swin","ntscc"],required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--eta",type=float,default=.2)
    p.add_argument("--draws",type=int,default=10)
    p.add_argument("--seed",type=int,default=8128)
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    manifest=json.loads(args.checkpoint.with_suffix(".json").read_text())
    if sha256(args.checkpoint)!=manifest["sha256"]:
        raise ValueError("author checkpoint changed")
    model,_,missing=build(args,manifest)
    counted=install_quantized_scale(model,args.family)
    metric=LPIPSMetric().cuda().eval()
    paths=paths_at(args.data)
    if args.limit:
        paths=paths[:args.limit]
    dataset=Images(paths)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with args.output.open("w",newline="") as handle,torch.inference_mode():
        writer=None
        for i,path in enumerate(paths):
            image,name=dataset[i]
            image=image[None].cuda()
            image_hash=sha256(path)
            for draw in range(args.draws):
                noise_seed=int(hashlib.sha256(f"{args.seed}:{image_hash}:10.0:{draw}".encode()).hexdigest()[:8],16)
                torch.manual_seed(noise_seed)
                counted["scales"]=0
                if args.family=="swin":
                    recon,cbr,_,_,_=model(image,given_SNR=10.)
                    rate_bits=0
                else:
                    model.channel.chan_param=10.
                    result=model(image)
                    recon,cbr=result[6],result[4]
                    rate_bits=(image.shape[-2]//16)*(image.shape[-1]//16)*4
                recon=recon.clamp(0,1)
                payload=float(cbr)*image.numel()
                scale_uses=math.ceil(counted["scales"]*16/.5)
                rate_uses=math.ceil(rate_bits/.5)
                row=dict(model=manifest["name"]+"_paid_metadata",image=name,image_sha256=image_hash,
                         snr_db=10.,draw=draw,noise_seed=noise_seed,payload_uses=payload,
                         scale_uses=scale_uses,rate_map_bits=rate_bits,rate_map_uses=rate_uses,feedback_uses=2,
                         cbr_payload=float(cbr),cbr_total=(payload+scale_uses+rate_uses+2)/image.numel(),
                         mse=mse(image,recon).item(),psnr=psnr(image,recon).item(),
                         ms_ssim=ms_ssim(image,recon).item(),lpips=metric(image,recon).item())
                if not all(math.isfinite(row[k]) for k in ("psnr","mse","lpips","ms_ssim","cbr_total")):
                    raise FloatingPointError("non-finite baseline output")
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                records.append(row)
            handle.flush()
            print(json.dumps(dict(model=manifest["name"],images_done=i+1,images=len(paths))),flush=True)
    import subprocess
    save_json(args.output.with_suffix(".json"),dict(status="smoke_only" if args.limit else "completed",
              checkpoint=manifest,upstream_commit=subprocess.check_output(
                  ["git","-C",str(args.upstream),"rev-parse","HEAD"],text=True).strip(),
              images=len(paths),draws=args.draws,seed=args.seed,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              source_sha256=sha256(__file__),csv_sha256=sha256(args.output),
              summary={k:float(np.mean([row[k] for row in records])) for k in
                       ("psnr","mse","lpips","ms_ssim","cbr_payload","cbr_total")},
              protocol="RGB center256; IEEE binary16 scale; reliable metadata/control at 0.5 info bits/complex use; fixed 4-bit NTSCC token-rate index; one final ACK.",
              native_protocol_deviations="Explicit scalar quantization and metadata/control cost; author neural weights and physical AWGN unchanged."))


if __name__=="__main__":
    main()
