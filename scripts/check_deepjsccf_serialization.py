"""Regression gate: check every TFC kernel/GDN parameter survives checkpoint I/O."""
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from jscc import DeepJSCCF


def chain():
    image=tf.keras.Input(shape=(None,None,3))
    first=DeepJSCCF(10,2,"awgn",None,False,0,name="layer0")(image)
    base=tf.keras.Model(image,first[0])
    base.trainable=False
    decoded,fb_image,out,fb_out,gain=first
    second=DeepJSCCF(10,2,"awgn",None,True,1,name="layer1")((image,fb_image,fb_out,decoded,out,gain))
    return tf.keras.Model(image,second[0])


def main():
    tf.keras.utils.set_random_seed(101)
    original=chain()
    with tempfile.TemporaryDirectory(prefix="deepjsccf-roundtrip-") as tmp:
        path=Path(tmp)/"complete.h5"
        original.save_weights(str(path))
        tf.keras.utils.set_random_seed(202)
        restored=chain()
        restored.load_weights(str(path))
        before,after=original.get_weights(),restored.get_weights()
        assert len(before)==len(after)
        deltas=[float(np.max(np.abs(a-b))) for a,b in zip(before,after)]
        assert max(deltas)==0.,max(deltas)
        # Negative control documents why the discarded first attempt is invalid.
        bad=Path(tmp)/"incomplete.weights.h5"
        original.save_weights(str(bad))
        tf.keras.utils.set_random_seed(303)
        incomplete=chain()
        incomplete.load_weights(str(bad))
        bad_deltas=[float(np.max(np.abs(a-b))) for a,b in zip(before,incomplete.get_weights())]
        assert max(bad_deltas)>0.,"Re-audit format behavior if the runtime changes"
        print({"status":"PASS","weight_tensors":len(before),"legacy_h5_max_error":max(deltas),
               "new_weights_h5_max_error":max(bad_deltas)},flush=True)

if __name__=="__main__":
    main()
