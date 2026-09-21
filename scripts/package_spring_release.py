"""Archive exact original evidence and a manifest, without packaging image datasets."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    output=ROOT/'artifacts/spring_final_20260921'
    output.mkdir(parents=True,exist_ok=True)
    sources=sorted((ROOT/'results/h2/test').glob('train_20*_channel_*.csv'))
    sources+=sorted((ROOT/'results/h2/kodak').glob('train_20*_channel_*.csv'))
    assert len(sources)==60
    sources+=sorted((ROOT/'results/h2').glob('seed_20*/decision.json'))
    assert len(sources)==70
    sources+=sorted((ROOT/'results/h4').rglob('*.csv'))
    sources+=sorted((ROOT/'results/h4').rglob('*.json'))
    sources.append(ROOT/'results/h2/latency/rtx_pro_6000.json')
    sources.append(ROOT/'results/h1/comparison.json')
    sources.append(ROOT/'results/h1/huber_rec1/calibration_k4_t080.csv')
    archive=output/'original_h2_h4_evidence.tar.gz'
    manifest=[]
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for source in sources:
            relative=source.relative_to(ROOT).as_posix()
            tar.add(source,arcname=relative,recursive=False)
            row=dict(path=relative,bytes=source.stat().st_size,sha256=digest(source))
            if source.suffix=='.csv':
                with source.open(newline='') as f:
                    row['rows']=sum(1 for _ in csv.DictReader(f))
            manifest.append(row)
    record=dict(archive=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,files=manifest)
    (output/'original_evidence_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'}))


if __name__=='__main__':main()
