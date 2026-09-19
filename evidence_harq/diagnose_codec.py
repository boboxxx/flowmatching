"""Single-batch optimization diagnostic; never a held-out performance result."""
import argparse
import json
from pathlib import Path
import torch
from .codec import PublishedSwin, EnhancementCodec
from .data import Images, paths_at, seed_all
from .physical import packet_awgn, enhancement_awgn, subset_mask
from .train_codec import save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=200)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg=json.loads((args.run/"configuration.json").read_text())
    seed_all(9127)
    torch.set_num_threads(4)
    base=PublishedSwin(cfg["upstream"],cfg["base_checkpoint"]).cuda()
    codec=EnhancementCodec(bounded_evidence=cfg.get("bounded_evidence",False)).cuda()
    codec.load_state_dict(torch.load(args.run/"best.pt",map_location="cpu",weights_only=False)["model"])
    dataset=Images(paths_at(cfg["train_data"])[:4])
    image=torch.stack([dataset[i][0] for i in range(4)]).cuda()
    with torch.no_grad():
        z=base.encode(image)
        noise=torch.randn_like(z)
        y,_=packet_awgn(z,10.,noise)
        base_loss=(base.decode(y)-image).square().mean().item()
    mask=subset_mask(torch.full((4,),15,device="cuda"))
    optimizer=torch.optim.Adam(codec.parameters(),lr=1e-3)
    records=[]
    for step in range(args.steps+1):
        e=codec.encode(image,z)
        ye,_=enhancement_awgn(e,10.,noise)
        recon=base.decode(codec.fuse(y,ye,mask,10.))
        loss=(image-recon).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step%25==0:
            row=dict(step=step,mse=loss.item(),base_mse=base_loss,
                     evidence_rms=e.square().mean().sqrt().item(),
                     encoder_grad=codec.image_encoder[0].weight.grad.norm().item(),
                     fusion_grad=codec.fusion[-1].weight.grad.norm().item())
            records.append(row)
            print(json.dumps(row),flush=True)
        if step<args.steps:
            torch.nn.utils.clip_grad_norm_(codec.parameters(),1.)
            optimizer.step()
    save_json(args.output,dict(status="diagnostic_only",fixed_training_images=[p.name for p in dataset.paths],
                              records=records,warning="Training-set overfit; not generalization evidence"))


if __name__=="__main__":
    main()
