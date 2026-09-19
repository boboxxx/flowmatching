"""Six specific public files linked by the published CDDM author's README."""
import argparse
import hashlib
import json
from pathlib import Path
import gdown

FILES = {
    "CDDM_snr13_channel_awgn_C36.pt": "10PowelxqyxTJtMY-CbKyMp5ETOT2fGvE",
    "decoder_snr10_channel_awgn_C36.pt": "10bw0hDCFo1wQPVvrIhExluXBQEU2oa48",
    "decoder_snr13_channel_awgn_C36.pt": "10d_fhsFFK7zW_r2yHMRwkMWodbyQFgVp",
    "encoder_snr10_channel_awgn_C36.pt": "10hjiQcdRJ6IGSHUhzKSEvyZ4tazW9yI1",
    "encoder_snr13_channel_awgn_C36.pt": "10iu47OwgLcriQFcxCZm9u5_DFB2DgkEY",
    "redecoder_snr10_channel_awgn_C36.pt": "10jdF_zKnBsAqOed6ufGgXFvI37Z4Qd75",
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,identifier in FILES.items():
        path=args.output/name
        if not path.exists():
            temporary=path.with_suffix(".download")
            if gdown.download(id=identifier,output=str(temporary),quiet=False,resume=True) is None:
                raise RuntimeError(f"failed public author download: {name}")
            temporary.replace(path)
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):
                digest.update(block)
        manifest=dict(name=name,source=f"https://drive.google.com/file/d/{identifier}/view",
                      author_folder="https://drive.google.com/drive/folders/1-oEe1F_-OqkWoeYOtyOGJLiGJskLbcyX",
                      bytes=path.stat().st_size,sha256=digest.hexdigest(),paper_doi="10.1109/TWC.2024.3379244")
        path.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    main()
