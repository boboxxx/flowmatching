"""Frozen-model finite-precision audit; no training or calibration changes."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from ugp.channel import ChannelContext, temporal_correlation
from ugp.data import ImageDataset
from flowharq.model import FlowHARQJSCC
from flowharq.metrics import psnr
from flowharq.perceptual import LPIPSMetric


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def quantized_observation(clean, observation):
    paired = clean.reshape(*clean.shape[:2], clean.shape[-1]//2, 2)
    symbols = torch.complex(paired[..., 0], paired[..., 1])
    scale = symbols.abs().square().mean(-1, keepdim=True).add(1e-8).sqrt()
    quantized = scale.half().float()
    if not torch.isfinite(quantized).all() or (quantized <= 0).any():
        raise FloatingPointError('binary16 scale not representable')
    return replace(observation, tokens=observation.tokens*(quantized/scale))


def predictions(model, first, second, context, selected):
    combined = model.combine(first, second)
    direct = model.decode(first.tokens, context)
    full = model.decode(combined, context)
    repaired, probabilities, _ = model.repair(first, context, steps=4, threshold=.8)
    fm = model.decode(repaired, context)
    features = first.receiver_features(context)
    predicted_direct = model.quality.log_mse_to_psnr(model.quality(first.tokens, probabilities, features)) + selected['direct_quality_bias']
    predicted_fm = model.quality.log_mse_to_psnr(model.quality(repaired, probabilities, features)) + selected['fm_quality_bias']
    a, f = predicted_direct < 24, predicted_fm < 24
    zero = torch.zeros_like(a)
    return {'direct': (direct, zero), 'full_harq': (full, ~zero), 'fm_only': (fm, zero),
            'adaptive_harq': (torch.where(a[:, None, None, None], full, direct), a),
            'flowharq': (torch.where(f[:, None, None, None], full, fm), f)}


def policy_latency(model, first, second, context, selected):
    def policy(use_flow):
        if use_flow:
            state, probabilities, _ = model.repair(first, context, steps=4, threshold=.8)
            bias = selected['fm_quality_bias']
        else:
            _, probabilities = model.predict_unreliability(first, context)
            state, bias = first.tokens, selected['direct_quality_bias']
        prediction = model.quality.log_mse_to_psnr(model.quality(state, probabilities, first.receiver_features(context)))+bias
        # This batch-one benchmark includes branch selection and exactly one decode.
        if (prediction < 24).item():
            state = model.combine(first, second)
        return model.decode(state, context)
    result = {}
    for use_flow in (False, True):
        for _ in range(20):
            policy(use_flow)
        samples = []
        for _ in range(100):
            if first.tokens.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            policy(use_flow)
            if first.tokens.is_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        result['flowharq' if use_flow else 'adaptive_harq'] = dict(mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)), repetitions=len(samples))
    result['note'] = 'Batch one; both observations precomputed. Includes reliability, quality head, branch selection, optional repair/MRC and one decode. Excludes radio, encoding, equalization, feedback, queues; one fixed input only.'
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--data', required=True)
    p.add_argument('--dataset', choices=['div2k', 'kodak'], required=True)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--upstream', default='upstream/SwinJSCC')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['args']
    model = FlowHARQJSCC(args.upstream, latent_dim=config.get('latent_dim',32),
                        flow_hidden_dim=config.get('flow_hidden_dim',128), flow_depth=config.get('flow_depth',2)).to(args.device).eval()
    model.load_state_dict(payload['model'], strict=True)
    selected = json.loads(args.calibration.read_text())['selected']
    assert selected['flow_steps']==4 and selected['mask_threshold']==.8 and selected['target_psnr']==24
    perceptual = LPIPSMetric().to(args.device).eval()
    data = ImageDataset(args.data, train=False)
    if args.dataset=='div2k':
        assert len(data.paths)==100
        data.paths = data.paths[20:]
    else:
        assert len(data.paths)==24
    if args.limit:
        data.paths = data.paths[:args.limit]
    loader = DataLoader(data, batch_size=4, num_workers=4, pin_memory=args.device=='cuda')
    write_json(args.output/'inputs.json', dict(images=[dict(name=path.name,sha256=digest(path)) for path in data.paths], checkpoint_sha256=digest(args.checkpoint), calibration_sha256=digest(args.calibration)))
    (args.output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    paths=[Path('ugp/channel.py'),Path('ugp/model.py'),Path('ugp/data.py'),Path('flowharq/model.py'),Path('flowharq/modules.py'),Path('flowharq/metrics.py'),Path('flowharq/perceptual.py'),Path(__file__)]
    snapshot=args.output/'source_snapshot'
    snapshot.mkdir()
    import shutil
    for path in paths:
        shutil.copy2(path,snapshot/str(path.relative_to(ROOT) if path.is_absolute() else path).replace('/','__'))
    write_json(args.output/'provenance.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               source_sha256={str(path):digest(path) for path in paths},torch=torch.__version__,cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
               device=args.device, msssim_version=__import__('importlib.metadata',fromlist=['version']).version('pytorch-msssim'),
               slurm_job_id=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename))
    rows=0
    with (args.output/'raw.csv').open('w',newline='') as stream, torch.inference_mode():
        writer=None
        last=None
        for snr_db in (0.,3.,6.,9.,12.,15.):
            for speed_kmh in (30.,60.,90.,120.):
                for image,names in loader:
                    image=image.to(args.device)
                    b=len(image)
                    snr=torch.full((b,),snr_db,device=args.device)
                    speed=torch.full((b,),speed_kmh,device=args.device)
                    age=torch.full((b,),2.,device=args.device)
                    context=ChannelContext(snr,speed,age,temporal_correlation(speed,age))
                    clean=model.encode(image,context)
                    first=model.transmit(clean,context)
                    second=model.transmit(clean,context,previous=first,round_gap_ms=1.)
                    last=(image,first,second,context)
                    for scale_mode in ('ideal','binary16'):
                        a,bobs=(first,second) if scale_mode=='ideal' else (quantized_observation(clean,first),quantized_observation(clean,second))
                        results=predictions(model,a,bobs,context,selected)
                        for method,(recon,nack) in results.items():
                            metrics=dict(psnr=psnr(image,recon),mse=(image-recon).square().flatten(1).mean(1),
                                         lpips=perceptual(image,recon),ms_ssim=ms_ssim(image,recon,data_range=1.,size_average=False))
                            for i,name in enumerate(names):
                                n=int(nack[i])
                                row=dict(dataset=args.dataset,image=name,training_seed=2030,channel_seed=args.seed,
                                         snr_db=snr_db,speed_kmh=speed_kmh,scale_mode=scale_mode,method=method,nack=n,
                                         payload_uses=4096*(1+n),scale_bits=4096,feedback_bits=1+n,
                                         total_uses=4096*(1+n)+8192+2*(1+n),
                                         **{k:float(v[i]) for k,v in metrics.items()},
                                         outage=int(metrics['psnr'][i]<24),false_ack=int(not n and metrics['psnr'][i]<24))
                                if not all(np.isfinite(row[k]) for k in metrics):
                                    raise FloatingPointError(row)
                                if writer is None:
                                    writer=csv.DictWriter(stream,fieldnames=row)
                                    writer.writeheader()
                                writer.writerow(row)
                                rows+=1
                stream.flush()
                print(json.dumps(dict(dataset=args.dataset,snr=snr_db,speed=speed_kmh,rows=rows)),flush=True)
        # Slice an already observed condition for a fixed-input complete-policy timing check.
        _,a,b,c=last
        one=lambda o: replace(o,**{key:value[:1] for key,value in vars(o).items()})
        c=ChannelContext(*(value[:1] for value in vars(c).values()))
        latency=policy_latency(model,one(a),one(b),c,selected)
        write_json(args.output/'policy_latency.json',latency)
    write_json(args.output/'status.json',dict(status='smoke_only' if args.limit else 'completed',rows=rows,csv_sha256=digest(args.output/'raw.csv')))


if __name__=='__main__':
    main()
