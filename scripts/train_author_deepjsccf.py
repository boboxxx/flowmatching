"""DIV2K data adapter for Kurka & Gunduz's unmodified DeepJSCC-f networks.

This is a documented dataset/runtime adaptation, not an exact ImageNet replication.
Stage-wise freezing, Adam 1e-4, MSE and ideal channel-output feedback follow the author code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import tensorflow as tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--snr", type=float, default=10.)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh directory; never overwrite checkpoints")
    args.output.mkdir(parents=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("This training run requires GPU")
    sys.path.insert(0, str(args.upstream.resolve()))
    from jscc import DeepJSCCF, psnr_metric
    training = sorted(args.train.glob("*.png"))
    validation = sorted(args.valid.glob("*.png"))[:20]
    if len(training) != 800 or len(validation) != 20:
        raise ValueError("Expected 800 DIV2K training images and first 20 validation images")
    # Content hashes also detect accidental overlap even if filenames differ.
    train_hash = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in training}
    val_hash = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in validation}
    if set(train_hash.values()) & set(val_hash.values()):
        raise ValueError("Training/validation image overlap")
    meta = dict(configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                upstream_commit=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
                tensorflow=tf.__version__, train_hashes=train_hash, validation_hashes=val_hash,
                adaptation="DIV2K instead of ImageNet; TF 2.14.1/TFC 2.14.1 instead of TF 1.15.2/TFC 1.3",
                crop=128, learning_rate=1e-4, loss="MSE", feedback="ideal channel-output feedback",
                cbr_per_layer=args.channels/96, stages=[], checkpoint_format="legacy_hdf5_verified")
    status_path = args.output / "status.json"
    meta["status"] = "running"
    status_path.write_text(json.dumps(meta, indent=2)+"\n")

    def read(path, augment):
        x = tf.image.convert_image_dtype(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)
        if augment:
            x = tf.image.random_crop(x, [128, 128, 3])
            x = tf.image.random_flip_left_right(x)
        else:
            shape = tf.shape(x)
            x = tf.image.crop_to_bounding_box(x, (shape[0]-128)//2, (shape[1]-128)//2, 128, 128)
        return x, x

    train = tf.data.Dataset.from_tensor_slices([str(x) for x in training]).shuffle(800, seed=args.seed)
    train = train.map(lambda x: read(x, True), num_parallel_calls=4).batch(args.batch_size).repeat().prefetch(2)
    valid = tf.data.Dataset.from_tensor_slices([str(x) for x in validation])
    valid = valid.map(lambda x: read(x, False), num_parallel_calls=2).cache().repeat(5).batch(args.batch_size).prefetch(2)
    image = tf.keras.Input(shape=(None, None, 3))
    previous = None
    start = time.time()
    for stage in range(args.layers):
        layer = DeepJSCCF(args.snr, args.channels, "awgn", None, stage>0, stage, name=f"layer{stage}")
        if stage == 0:
            current = layer(image)
        else:
            decoded, feedback_image, channel_out, feedback_channel, gain = previous
            current = layer((image, feedback_image, feedback_channel, decoded, channel_out, gain))
        model = tf.keras.Model(image, current[0])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=[psnr_metric])
        early = tf.keras.callbacks.EarlyStopping(monitor="val_psnr_metric", mode="max", min_delta=.01,
                                                 patience=3, restore_best_weights=True, verbose=1)
        # Keras' new .weights.h5 format omits TFC tf.Module kernel/GDN parameters.
        # Legacy .h5 includes them; check_deepjsccf_serialization.py is a mandatory gate.
        best = str(args.output / f"stage{stage}_best.h5")
        callbacks = [early, tf.keras.callbacks.CSVLogger(str(args.output/f"stage{stage}_history.csv")),
                     tf.keras.callbacks.ModelCheckpoint(best, monitor="val_psnr_metric", mode="max",
                                                        save_best_only=True, save_weights_only=True),
                     tf.keras.callbacks.TerminateOnNaN()]
        history = model.fit(train, epochs=1 if args.smoke else args.epochs,
                            steps_per_epoch=2 if args.smoke else args.steps_per_epoch,
                            validation_data=valid, validation_steps=1 if args.smoke else None,
                            callbacks=callbacks, verbose=2)
        if not all(tf.math.is_finite(x) for x in history.history["loss"]):
            raise FloatingPointError("Non-finite training loss")
        # Select best validation epoch even when stopping at the budget, then freeze.
        tf.train.Checkpoint(model=model, optimizer=model.optimizer).write(str(args.output/f"stage{stage}_last_resume"))
        model.load_weights(best)
        model.save_weights(str(args.output/f"stage{stage}_selected.h5"))
        meta["stages"].append(dict(stage=stage, epochs=len(history.epoch),
                                    best_validation_psnr=max(history.history["val_psnr_metric"]),
                                    early_stopped=early.stopped_epoch>0, elapsed_seconds=time.time()-start,
                                    cbr=(stage+1)*args.channels/96))
        status_path.write_text(json.dumps(meta, indent=2)+"\n")
        model.trainable = False
        previous = current
    meta["status"] = "smoke_only" if args.smoke else "training_finished_requires_evaluation_and_convergence_review"
    status_path.write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps({"status":meta["status"],"stages":meta["stages"]}), flush=True)

if __name__ == "__main__":
    main()
