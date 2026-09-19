"""Export float reconstructions; score in a separate process with shared PyTorch metrics.

Separate processes prevent TF/cuDNN8 and PyTorch/cuDNN9 runtime interference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--snr", type=float, default=10)
    p.add_argument("--seed", type=int, default=7070)
    args = p.parse_args()
    status = json.loads((args.run/"status.json").read_text())
    if status["status"] != "training_finished_requires_evaluation_and_convergence_review":
        raise ValueError("Training incomplete or smoke-only; do not export as a baseline result")
    if status.get("checkpoint_format") != "legacy_hdf5_verified":
        raise ValueError("Unverified Keras checkpoint format; do not evaluate incomplete TFC weights")
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    sys.path.insert(0, str(args.upstream.resolve()))
    from jscc import DeepJSCCF
    cfg = status["configuration"]
    image = tf.keras.Input(shape=(None, None, 3))
    previous = None
    outputs = []
    checkpoint_hashes = {}
    for stage in range(cfg["layers"]):
        layer = DeepJSCCF(args.snr, cfg["channels"], "awgn", None, stage>0, stage, name=f"layer{stage}")
        if stage == 0:
            current = layer(image)
        else:
            decoded, feedback_image, channel_out, feedback_channel, gain = previous
            current = layer((image, feedback_image, feedback_channel, decoded, channel_out, gain))
        model = tf.keras.Model(image, current[0])
        weight_path = args.run/f"stage{stage}_selected.h5"
        model.load_weights(str(weight_path))
        checkpoint_hashes[weight_path.name] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        model.trainable = False
        outputs.append(current[0])
        previous = current
    model = tf.keras.Model(image, outputs)
    paths = sorted(args.data.glob("*.png"))
    if len(paths) != 24:
        raise ValueError("Expected all 24 Kodak images")
    for index, path in enumerate(paths):
        ref = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)/255
        if args.crop:
            h,w = ref.shape[:2]
            top,left = (h-args.crop)//2,(w-args.crop)//2
            ref = ref[top:top+args.crop, left:left+args.crop]
        reconstructions = []
        seeds = []
        for draw in range(args.draws):
            seed = args.seed + index*10000 + round(args.snr*100)*100000 + draw
            tf.random.set_seed(seed)
            result = model(tf.convert_to_tensor(ref[None]), training=False)
            if not isinstance(result, (list,tuple)):
                result = [result]
            reconstructions.append(np.stack([np.clip(x.numpy()[0],0,1) for x in result]))
            seeds.append(seed)
        np.savez_compressed(args.output/f"{path.stem}.npz", reference=ref,
                            reconstruction=np.stack(reconstructions), seeds=np.asarray(seeds))
        print(json.dumps(dict(image=path.name, draws=args.draws, layers=cfg["layers"])), flush=True)
    meta = dict(training_status=status, checkpoint_sha256=checkpoint_hashes,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                images={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
                cbr_per_layer=cfg["channels"]/96, feedback="ideal channel-output feedback",
                status="export_complete_not_convergence_certified")
    (args.output/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")

if __name__ == "__main__":
    main()
