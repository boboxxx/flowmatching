"""Paired published-Swin transfer diagnostic; see the locked bridge protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from pytorch_msssim import ms_ssim
from flowharq.model import FlowHARQJSCC
from flowharq.perceptual import LPIPSMetric
from ugp.channel import ChannelContext, ChannelObservation, temporal_correlation
from evaluate_published_checkpoint import build


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def image_tensor(path):
    with Image.open(path) as src:
        src = src.convert('RGB')
        w, h = src.size
        if min(w, h) < 256:
            raise ValueError('No resizing/padding permitted')
        array = np.array(src.crop(((w-256)//2, (h-256)//2, (w-256)//2+256, (h-256)//2+256)))
    return torch.from_numpy(array.copy()).permute(2,0,1)[None].float().cuda()/255


def observations(tokens, h1, h2, noise1, noise2, snr, author=False):
    assert tokens.numel() == 8192
    if author:
        flat = tokens.flatten()
        z = torch.complex(flat[:4096], flat[4096:])
        scale = (tokens.square().mean()*2).sqrt()
    else:
        pair = tokens.reshape(1,256,16,2)
        z = torch.complex(pair[...,0], pair[...,1])
        scale = z.abs().square().mean(-1,keepdim=True).add(1e-8).sqrt()
    tx = z/scale
    assert abs(float(tx.abs().square().mean())-1) < 1e-4
    q = scale.half().float()
    if not torch.isfinite(q).all() or (q <= 0).any():
        raise ValueError('Unrepresentable binary16 scale')
    decoded = []
    for gain, noise in ((h1,noise1),(h2,noise2)):
        gain_sq = gain.abs().square()
        received = gain*tx + noise.reshape_as(tx)*10**(-snr/20)
        restored = received*gain.conj()/(gain_sq+1e-8)*q
        if author:
            state = torch.cat((restored.real.flatten(),restored.imag.flatten())).reshape_as(tokens)
        else:
            state = torch.stack((restored.real,restored.imag),-1).reshape_as(tokens)
        obs = ChannelObservation(state, gain.reshape(1,1,1), gain_sq.reshape(1,1,1),
                                 (snr+10*gain_sq.clamp_min(1e-8).log10()).reshape(1),
                                 ((.5*10**(-snr/10))/(gain_sq+1e-8)).reshape(1,1,1))
        decoded.append(obs)
    return decoded, q.numel()*16


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dataset',choices=['div2k','kodak'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(9262026)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    upstream=args.base/'upstream/SwinJSCC'
    checkpoint=args.base/'artifacts/published_weights/swin_mse_c32.pth'
    manifest=json.loads(checkpoint.with_suffix('.json').read_text())
    assert sha(checkpoint)==manifest['sha256'] and manifest['channels']==32
    author,_,missing=build(SimpleNamespace(family='swin',upstream=upstream,checkpoint=checkpoint),manifest)
    own_path=args.base/'results/h2/seed_2030/checkpoint.pt'
    decision_path=args.base/'results/h2/seed_2030/decision.json'
    payload=torch.load(own_path,map_location='cpu',weights_only=False)
    own=FlowHARQJSCC(str(upstream)).cuda().eval()
    own.load_state_dict(payload['model'],strict=True)
    selected=json.loads(decision_path.read_text())['selected']
    assert selected['flow_steps']==4 and selected['mask_threshold']==.8 and selected['target_psnr']==24
    perceptual=LPIPSMetric().cuda().eval()
    paths=sorted(x for x in args.data.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg'})
    if args.dataset=='div2k':
        assert len(paths)==100
        paths=paths[20:]
    else:
        assert len(paths)==24
    if args.limit:
        paths=paths[:args.limit]
    args.output.mkdir(parents=True)
    started=time.time()
    rows=0
    reconstructions=[]
    save(args.output/'inputs.json',dict(images=[dict(name=x.name,sha256=sha(x)) for x in paths],
         author_checkpoint=manifest,own_checkpoint_sha256=sha(own_path),calibration_sha256=sha(decision_path)))
    sources=list((ROOT/'flowharq').glob('*.py'))+list((ROOT/'ugp').glob('*.py'))+[Path(__file__),Path(__file__).with_name('evaluate_published_checkpoint.py')]
    save(args.output/'runtime.json',dict(host=platform.node(),python=sys.version,torch=torch.__version__,
         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),invocation=sys.argv,
         source_hashes={str(x.relative_to(ROOT)):sha(x) for x in sources},
         upstream_sources={str(x.relative_to(upstream)):sha(x) for x in (upstream/'net').glob('*.py')},
         regenerated_buffers=missing,author_parameters=sum(x.numel() for x in author.parameters()),
         flowharq_parameters=sum(x.numel() for x in own.parameters())))
    (args.output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    with (args.output/'raw.csv').open('w',newline='') as stream, torch.inference_mode():
        writer=None
        for image_index,path in enumerate(paths):
            reference=image_tensor(path)
            author.encoder.update_resolution(256,256)
            author.decoder.update_resolution(16,16)
            z_author=author.encoder(reference,10.,32,author.model)
            for snr in (0.,3.,6.,9.,12.,15.):
                for speed in (30.,60.,90.,120.):
                    key=f'9262026:{sha(path)}:{snr}:{speed}'
                    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
                    generator=torch.Generator(device='cpu').manual_seed(seed)
                    def cn(n):
                        parts=torch.randn(2,n,generator=generator)
                        return torch.complex(parts[0],parts[1]).cuda()/2**.5
                    h1,e=cn(2).unbind()
                    v=torch.tensor([speed],device='cuda')
                    rho=temporal_correlation(v,torch.ones_like(v))[0]
                    h2=rho*h1+(1-rho.square()).sqrt()*e
                    n1,n2=cn(4096),cn(4096)
                    context=ChannelContext(torch.tensor([snr],device='cuda'),v,torch.full_like(v,2),
                                           temporal_correlation(v,torch.full_like(v,2)))
                    z=own.encode(reference,context)
                    (oa,ob),own_bits=observations(z,h1,h2,n1,n2,snr)
                    (sa,sb),author_bits=observations(z_author,h1,h2,n1,n2,snr,author=True)
                    direct=own.decode(oa.tokens,context)
                    full=own.decode(own.combine(oa,ob),context)
                    repaired,probabilities,_=own.repair(oa,context,steps=4,threshold=.8)
                    fm=own.decode(repaired,context)
                    feat=oa.receiver_features(context)
                    pa=own.quality.log_mse_to_psnr(own.quality(oa.tokens,probabilities,feat))+selected['direct_quality_bias']
                    pf=own.quality.log_mse_to_psnr(own.quality(repaired,probabilities,feat))+selected['fm_quality_bias']
                    na,nf=int(pa.item()<24),int(pf.item()<24)
                    outputs=[('SwinJSCC_author_one_shot',author.decoder(sa.tokens,snr,author.model),0,author_bits),
                             ('SwinJSCC_author_Chase_wrapper',author.decoder(own.combine(sa,sb),snr,author.model),1,author_bits),
                             ('same_codec_direct',direct,0,own_bits),('same_codec_full_HARQ',full,1,own_bits),
                             ('adaptive_HARQ',full if na else direct,na,own_bits),
                             ('FlowHARQ',full if nf else fm,nf,own_bits)]
                    for method,estimate,nack,bits in outputs:
                        estimate=estimate.clamp(0,1)
                        mse=(reference-estimate).square().mean().item()
                        row=dict(dataset=args.dataset,image=path.name,snr_db=snr,speed_kmh=speed,channel_seed=seed,
                                 h1_real=h1.real.item(),h1_imag=h1.imag.item(),h2_real=h2.real.item(),h2_imag=h2.imag.item(),
                                 method=method,nack=nack,scale_bits=bits,payload_uses=4096*(1+nack),
                                 total_uses=4096*(1+nack)+2*bits+2*(1+nack),mse=mse,
                                 psnr=-10*np.log10(mse),ms_ssim=ms_ssim(reference,estimate,data_range=1.).item(),
                                 lpips=perceptual(reference,estimate).item(),outage=int(mse>10**(-2.4)))
                        if not all(np.isfinite(row[k]) for k in ('mse','psnr','ms_ssim','lpips')):
                            raise ValueError('Non-finite reconstruction metric')
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);rows+=1
                        if image_index==0 and speed==60 and snr in (0,15):
                            filename=f'{int(snr)}dB_{method}.png'
                            array=(estimate[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                            Image.fromarray(array).save(args.output/filename)
                            reconstructions.append(dict(file=filename,image=path.name,snr=snr,method=method,sha256=sha(args.output/filename)))
                    stream.flush()
            print(json.dumps(dict(dataset=args.dataset,images_done=image_index+1,images=len(paths),rows=rows)),flush=True)
    assert rows==len(paths)*24*6
    save(args.output/'reconstructions.json',reconstructions)
    save(args.output/'status.json',dict(status='smoke_only' if args.limit else 'completed',rows=rows,
         elapsed_seconds=time.time()-started,csv_sha256=sha(args.output/'raw.csv'),
         caveat='Published codec with our flat-Rayleigh Chase wrapper; not a published HARQ reproduction. Frozen unequal training recipes.'))


if __name__=='__main__':
    main()
