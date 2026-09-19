"""Create reproducibility inventories and source snapshots as generated artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    sha=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--snapshot-source",action="store_true")
    args=p.parse_args()
    if args.snapshot_source:
        destination=args.run/"source_snapshot"
        if destination.exists():
            raise FileExistsError(destination)
        source=Path("evidence_harq")
        provenance=json.loads((args.run/"provenance.json").read_text())
        for name,expected in provenance["source_sha256"].items():
            if digest(Path(name))!=expected:
                raise ValueError(f"current source differs from launch: {name}")
        destination.mkdir()
        for path in sorted(source.glob("*.py")):
            shutil.copy2(path,destination/path.name)
    inventory=[]
    for path in sorted(args.run.rglob("*")):
        if path.is_file() and path.suffix!=".tmp" and path!=args.run/"artifact_inventory.json":
            inventory.append(dict(path=str(path.relative_to(args.run)),bytes=path.stat().st_size,sha256=digest(path)))
    (args.run/"artifact_inventory.json").write_text(json.dumps(dict(files=inventory),indent=2)+"\n")
    print(json.dumps(dict(run=str(args.run),files=len(inventory))))


if __name__=="__main__":
    main()
