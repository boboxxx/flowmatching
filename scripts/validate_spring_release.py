"""Fail closed unless the completed Spring release is internally consistent."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'artifacts/spring_final_20260921'
    runtime=json.loads((base/'runtime/inventory.json').read_text())
    inventory={r['path']:r['sha256'] for r in runtime['checkpoints']+runtime['sources']}
    assert len(runtime['checkpoints'])==20
    for source in runtime['sources']:
        path=ROOT/source['path'] if source['path'].startswith('upstream/') else base/'runtime/source'/source['path']
        assert sha(path)==source['sha256']
    original=json.loads((base/'original_evidence_manifest.json').read_text())
    assert sha(base/original['archive'])==original['sha256']
    with tarfile.open(base/original['archive'],'r:gz') as t:
        for member in original['files']:
            assert hashlib.sha256(t.extractfile(member['path']).read()).hexdigest()==member['sha256']
    outputs={}
    for name,count in [('div2k',19200),('kodak',5760)]:
        path=base/name
        status=json.loads((path/'status.json').read_text())
        assert status['status']=='completed' and status['rows']==count
        assert status['csv_sha256']==sha(path/'raw.csv')
        inputs=json.loads((path/'inputs.json').read_text())
        assert inputs['checkpoint_sha256']==inventory['results/h2/seed_2030/checkpoint.pt']
        assert inputs['calibration_sha256']==inventory['results/h2/seed_2030/decision.json']
        provenance=json.loads((path/'provenance.json').read_text())
        assert provenance['device']=='cpu' and provenance['msssim_version']=='1.0.0'
        for source,digest in provenance['source_sha256'].items():
            if source in inventory:
                assert digest==inventory[source]
        outputs[name]=dict(rows=count,sha256=status['csv_sha256'],device='cpu',job=provenance['slurm_job_id'])
    evidence=json.loads((ROOT/'experiments/spring_final_20260921/evidence.json').read_text())
    for name in outputs:
        assert evidence[name]['precision_audit']['csv_sha256']==outputs[name]['sha256']
        assert evidence[name]['precision_audit']['archived_replay_expected'] is False
        assert len(evidence[name]['sources'])==30
    report=dict(status='passed',checkpoint_and_calibration_files=20,source_files=len(runtime['sources']),
                original_archive_files=len(original['files']),audits=outputs,
                limitations='Hashes establish artifact identity, not scientific validity or model convergence. Neural weights are on Artemis, not in this archive.')
    (base/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
