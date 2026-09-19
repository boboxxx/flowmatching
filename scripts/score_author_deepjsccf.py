"""Use exactly the same four metric implementations as the author-weight benchmark."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flowharq.metrics import mse, psnr, ms_ssim, ms_ssim_db


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    meta=json.loads((args.input/"manifest.json").read_text())
    torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net="alex",version="0.1").cuda().eval()
    rows=[]
    if args.output.exists():
        raise FileExistsError("Do not overwrite earlier results")
    with torch.inference_mode():
        for path in sorted(args.input.glob("*.npz")):
            payload=np.load(path)
            ref=torch.from_numpy(payload["reference"]).permute(2,0,1)[None].cuda()
            for draw, rounds in enumerate(payload["reconstruction"]):
                for stage, image in enumerate(rounds):
                    rec=torch.from_numpy(image).permute(2,0,1)[None].cuda()
                    score=ms_ssim(ref,rec)
                    row=dict(model="DeepJSCC-f_author_DIV2K", image=path.stem+".png",
                             image_sha256=meta["images"][path.stem+".png"], draw=draw,
                             noise_seed=int(payload["seeds"][draw]), rounds=stage+1,
                             snr_db=meta["configuration"]["snr"], height=ref.shape[-2], width=ref.shape[-1],
                             cbr_payload=(stage+1)*meta["cbr_per_layer"], mse=mse(ref,rec).item(),
                             psnr=psnr(ref,rec).item(), ms_ssim=score.item(), ms_ssim_db=ms_ssim_db(score).item(),
                             lpips=metric(ref*2-1,rec*2-1).item())
                    rows.append(row)
    if len(rows)!=24*meta["configuration"]["draws"]*meta["training_status"]["configuration"]["layers"]:
        raise ValueError("Incomplete export")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for rounds in sorted({x["rounds"] for x in rows}):
        group=[x for x in rows if x["rounds"]==rounds]
        summary.append(dict(rounds=rounds,images=24,draws=meta["configuration"]["draws"],
                            **{k:float(np.mean([x[k] for x in group])) for k in
                               ["cbr_payload","mse","psnr","ms_ssim","ms_ssim_db","lpips"]}))
    args.output.with_suffix(".summary.json").write_text(json.dumps(dict(provenance=meta,summary=summary),indent=2)+"\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
