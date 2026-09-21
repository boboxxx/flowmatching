"""Read-only inventory of frozen Artemis inputs; output is a new research artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoints=[]
    for seed in range(2030,2040):
        for filename in ('checkpoint.pt','decision.json'):
            path=Path(f'results/h2/seed_{seed}/{filename}')
            checkpoints.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
    sources=[]
    for folder in ('ugp','flowharq','upstream/SwinJSCC/net'):
        for path in sorted(Path(folder).rglob('*.py')):
            target=args.output/'source'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            sources.append(dict(path=str(path),sha256=digest(path)))
    for path in sorted(Path('results/h2').glob('seed_20*/decision.json')):
        target=args.output/'calibration'/path.parent.name/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    record=dict(platform=platform.platform(),python=platform.python_version(),checkpoints=checkpoints,sources=sources,
                note='Checkpoint paths are on Artemis. This manifest does not mean the model binaries are publicly hosted. Source snapshot is separate from the working Git tree.')
    (args.output/'inventory.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(checkpoints=len(checkpoints),sources=len(sources),output=str(args.output))))


if __name__=='__main__':main()
