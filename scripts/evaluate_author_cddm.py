"""Author CDDM weights and sampler, common metrics, explicitly charged scale.

No import of the author's experiment drivers: they parse CLI and write MongoDB
at import time. Only original model/channel/sampler classes are instantiated.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evidence_harq.data import Images, paths_at, sha256
from evidence_harq.train_codec import save_json
from flowharq.metrics import mse, psnr, ms_ssim
from flowharq.perceptual import LPIPSMetric


def checked_load(module,path):
    meta=json.loads(path.with_suffix(".json").read_text())
    if sha256(path)!=meta["sha256"]:
        raise ValueError(f"author weight hash mismatch: {path}")
    state=torch.load(path,map_location="cpu",weights_only=True)
    module.load_state_dict(state,strict=True)
    module.eval().requires_grad_(False)
    return meta


def author_module(name,path):
    # Import the original file without executing Diffusion/__init__.py, which
    # imports training drivers and their unrelated database dependencies.
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--draws",type=int,default=10)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--seed",type=int,default=8128)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(args.upstream.resolve()/"CDDM"))
    from Autoencoder.net.network import JSCC_encoder, JSCC_decoder
    from Autoencoder.net.channel import Channel
    UNet=author_module("cddm_original_model",args.upstream/"CDDM/Diffusion/Model.py").UNet
    ChannelDiffusionSampler=author_module("cddm_original_sampler",args.upstream/"CDDM/Diffusion/Diffusion.py").ChannelDiffusionSampler
    shared=dict(img_size=(256,256),window_size=8,mlp_ratio=4.,qkv_bias=True,qk_scale=None,
                norm_layer=torch.nn.LayerNorm,patch_norm=True)
    config=SimpleNamespace(channel_type="awgn",device="cuda",CUDA=True,
            encoder_kwargs=dict(**shared,patch_size=2,in_chans=3,embed_dims=[128,192,256,320],
                                depths=[2,2,6,2],num_heads=[4,6,8,10]),
            decoder_kwargs=dict(**shared,embed_dims=[320,256,192,128],depths=[2,6,2,2],num_heads=[10,8,6,4]))
    encoder13,decoder13,redecoder10=JSCC_encoder(config,36).cuda(),JSCC_decoder(config,36).cuda(),JSCC_decoder(config,36).cuda()
    encoder10,decoder10=JSCC_encoder(config,36).cuda(),JSCC_decoder(config,36).cuda()
    diffusion=UNet(T=1000,ch=576,ch_mult=[1,2,2],attn=[1],num_res_blocks=2,dropout=.1,input_channel=36).cuda()
    checkpoints={}
    for module,name in [(encoder13,"encoder_snr13"),(decoder13,"decoder_snr13"),(redecoder10,"redecoder_snr10"),
                        (encoder10,"encoder_snr10"),(decoder10,"decoder_snr10"),(diffusion,"CDDM_snr13")]:
        checkpoints[name]=checked_load(module,args.weights/(name+"_channel_awgn_C36.pt"))
    sampler=ChannelDiffusionSampler(diffusion,noise_schedule=1,t_max=10,beta_1=1e-4,beta_T=.02,T=1000).cuda().eval()
    channel=Channel(config)
    perceptual=LPIPSMetric().cuda().eval()
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
            z13,_=encoder13(image)
            z10,_=encoder10(image)
            for draw in range(args.draws):
                noise_seed=int(hashlib.sha256(f"{args.seed}:{image_hash}:10.0:{draw}".encode()).hexdigest()[:8],16)
                torch.manual_seed(noise_seed)
                y13,pwr13,h=channel(z13,10.)
                torch.manual_seed(noise_seed)
                y10,pwr10,_=channel(z10,10.)
                scale13=pwr13.sqrt().half().float()
                scale10=pwr10.sqrt().half().float()
                torch.cuda.synchronize()
                started=time.perf_counter()
                # Exactly the author's inference scaling and SNR mapping; do not replace with a generic DDPM.
                sample=sampler(y13/math.sqrt(1+.5*10**(-10/10)),10,13,h,"awgn")
                torch.cuda.synchronize()
                denoise_seconds=time.perf_counter()-started
                predictions={
                    "CDDM_author_C36_paid_scale":redecoder10(sample*scale13),
                    "CDDM_author_JSCC_snr13_C36_paid_scale":decoder13(torch.cat((y13.real,y13.imag),dim=2)*scale13),
                    "CDDM_author_JSCC_snr10_C36_paid_scale":decoder10(torch.cat((y10.real,y10.imag),dim=2)*scale10),
                }
                for model,recon in predictions.items():
                    recon=recon.clamp(0,1)
                    uses=z13.numel()/2
                    row=dict(model=model,image=name,image_sha256=image_hash,draw=draw,noise_seed=noise_seed,
                             snr_db=10.,payload_uses=uses,scale_uses=32,feedback_uses=2,
                             cbr_payload=uses/image.numel(),cbr_total=(uses+34)/image.numel(),
                             mse=mse(image,recon).item(),psnr=psnr(image,recon).item(),
                             lpips=perceptual(image,recon).item(),ms_ssim=ms_ssim(image,recon).item(),
                             diffusion_nfe=int(sampler.match_snr_t(10).item())+1 if model=="CDDM_author_C36_paid_scale" else 0,
                             diffusion_seconds=denoise_seconds if model=="CDDM_author_C36_paid_scale" else 0.)
                    if not all(math.isfinite(row[k]) for k in ("psnr","mse","lpips","ms_ssim")):
                        raise FloatingPointError("non-finite CDDM metric")
                    if writer is None:
                        writer=csv.DictWriter(handle,fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                    records.append(row)
                handle.flush()
            print(json.dumps(dict(stage="published_cddm",images_done=i+1,images=len(paths),rows=len(records))),flush=True)
    summaries=[]
    for model in sorted({row["model"] for row in records}):
        subset=[row for row in records if row["model"]==model]
        summaries.append(dict(model=model,images=len(paths),draws=args.draws,
                             **{key:float(np.mean([r[key] for r in subset]))
                                for key in ("psnr","mse","lpips","ms_ssim","cbr_payload","cbr_total","diffusion_nfe")}))
    save_json(args.output.with_suffix(".json"),dict(status="smoke_only" if args.limit else "completed",
              paper_doi="10.1109/TWC.2024.3379244",upstream_commit=subprocess.check_output(
                  ["git","-C",str(args.upstream),"rev-parse","HEAD"],text=True).strip(),
              checkpoints=checkpoints,summary=summaries,csv_sha256=sha256(args.output),
              adaptation="Author architecture/checkpoints/channel/sampler; quantized global scale, common center256 RGB metrics, charged scalar and one final ACK.",
              side_channel="Ideal reliable control/metadata, 0.5 information bits per complex use; no physical control code simulated.",
              timing_note="Raw sampling time only, not certified full latency; concurrent GPU jobs may affect timing."))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":
    main()
