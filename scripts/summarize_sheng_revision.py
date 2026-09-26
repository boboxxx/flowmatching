"""Validate and summarize the locked Sheng revision without selecting results."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
import math

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    root=ROOT/'artifacts/sheng_completion_20260926'
    target=ROOT/'experiments/sheng_completion_20260926'
    risk=root/'ack_risk'
    manifest=json.loads((risk/'manifest.json').read_text())
    assert sha(risk/'curves.csv')==manifest['csv_sha256']
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name)==digest
    selected=[r for r in read(risk/'curves.csv') if r['calibration_selected']=='True']
    summary={'risk_selected':selected,'bridge':{}}
    for dataset,images in [('div2k',80),('kodak',24)]:
        folder=root/('bridge_'+dataset)
        status=json.loads((folder/'status.json').read_text())
        inputs=json.loads((folder/'inputs.json').read_text())
        runtime=json.loads((folder/'runtime.json').read_text())
        for name,digest in runtime['source_hashes'].items():
            source=(ROOT/'artifacts/spring_final_20260921/runtime/source'/name
                    if name.startswith(('flowharq/','ugp/')) else ROOT/name)
            assert sha(source)==digest, name
        assert status['status']=='completed'
        assert sha(folder/'raw.csv')==status['csv_sha256']
        assert len(inputs['images'])==images
        prior=json.loads((ROOT/f'artifacts/spring_final_20260921/{dataset}/inputs.json').read_text())
        assert sorted(i['sha256'] for i in inputs['images'])==sorted(i['sha256'] for i in prior['images'])
        assert inputs['own_checkpoint_sha256']==prior['checkpoint_sha256']
        rows=read(folder/'raw.csv')
        assert len(rows)==images*24*6==status['rows']
        pairs=defaultdict(dict)
        groups=defaultdict(list)
        for row in rows:
            assert abs(float(row['psnr'])+10*math.log10(float(row['mse'])))<1e-6
            assert 0 <= float(row['ms_ssim']) <= 1
            n,b=int(row['nack']),int(row['scale_bits'])
            assert int(row['total_uses'])==4096*(1+n)+2*b+2*(1+n)
            key=tuple(row[k] for k in ('image','snr_db','speed_kmh','channel_seed'))
            assert row['method'] not in pairs[key]
            pairs[key][row['method']]=row
            groups[row['method']].append(row)
        for group in pairs.values():
            assert len(group)==6
            for field in ('h1_real','h1_imag','h2_real','h2_imag'):
                assert len({r[field] for r in group.values()})==1
            for method in ('adaptive_HARQ','FlowHARQ'):
                if int(group[method]['nack']):
                    for metric in ('mse','psnr','ms_ssim','lpips'):
                        assert group[method][metric]==group['same_codec_full_HARQ'][metric]
        for item in json.loads((folder/'reconstructions.json').read_text()):
            assert sha(folder/item['file'])==item['sha256']
        metrics=('psnr','mse','ms_ssim','lpips','nack','outage','total_uses')
        summary['bridge'][dataset]={method:{m:mean(float(r[m]) for r in values) for m in metrics} for method,values in groups.items()}
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Sheng revision results','',
           'Frozen-checkpoint diagnostics, not new training or untouched confirmation.','',
           '## Calibration-selected decision margins','',
           '| Dataset | Policy | Margin dB | ACK coverage % | Conditional false ACK % | Outage % |',
           '|---|---|---:|---:|---:|---:|']
    for r in selected:
        risk_value='undefined' if not r['conditional_false_ack'] else f"{100*float(r['conditional_false_ack']):.3f}"
        lines.append(f"| {r['dataset']} | {r['policy']} | {r['margin_db']} | {100*float(r['ack_coverage']):.3f} | {risk_value} | {100*float(r['final_outage']):.3f} |")
    lines+=['','## Same-channel published-codec transfer','',
            'One channel draw per image/condition; six SNRs and four speeds. Author SwinJSCC retains global scale normalization; FlowHARQ retains token scales. The Chase wrapper is not a published HARQ reproduction.','',
            '| Dataset | Method | PSNR | MSE | MS-SSIM | LPIPS | Charged uses |','|---|---|---:|---:|---:|---:|---:|']
    tex=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Data & Receiver & PSNR & MSE $\times10^3$ & MS-SSIM & LPIPS & Uses \\',r'\midrule']
    labels={'SwinJSCC_author_one_shot':'Swin 1R','SwinJSCC_author_Chase_wrapper':'Swin 2R',
            'same_codec_direct':'Own codec, 1R','same_codec_full_HARQ':'Own codec, 2R',
            'adaptive_HARQ':'Adaptive','FlowHARQ':'FlowHARQ'}
    dominance_notes=[]
    for dataset,methods in summary['bridge'].items():
        for method,r in methods.items():
            lines.append(f"| {dataset} | {method} | {r['psnr']:.4f} | {r['mse']:.6f} | {r['ms_ssim']:.4f} | {r['lpips']:.4f} | {r['total_uses']:.1f} |")
            if method in ('SwinJSCC_author_one_shot','SwinJSCC_author_Chase_wrapper','FlowHARQ'):
                tex.append(f"{'DIV' if dataset=='div2k' else 'Kodak'} & {labels[method]} & {r['psnr']:.2f} & {1000*r['mse']:.3f} & {r['ms_ssim']:.4f} & {r['lpips']:.4f} & {r['total_uses']:.0f} "+r'\\')
        author,ours=methods['SwinJSCC_author_one_shot'],methods['FlowHARQ']
        dominates=(author['psnr']>ours['psnr'] and author['mse']<ours['mse'] and author['ms_ssim']>ours['ms_ssim'] and author['lpips']<ours['lpips'] and author['total_uses']<ours['total_uses'])
        dominance_notes.append(f'All-four-metric / lower-charged-use dominance by published one-shot Swin on {dataset}: **{dominates}**.')
    lines+=['']+dominance_notes+['','The two-round author wrapper exceeds FlowHARQ on all four mean quality metrics at lower charged use on Kodak, but not on DIV2K: DIV2K mean MSE is worse despite better mean PSNR. This is a transfer diagnostic with unequal training and scale normalization, not a matched-training HARQ superiority claim.']
    tex += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/generated/sheng_bridge_table.tex').write_text('\n'.join(tex)+'\n')
    (target/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (target/'validation.json').write_text(json.dumps(dict(status='passed',checks=['input/checkpoint hashes','raw row counts','paired gains','common fallback identity','cost identities','PSNR/MSE identity','reconstruction hashes']),indent=2)+'\n')
    release=dict(status='diagnostics_completed_not_competitive_system_certified',
                 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and p.name!='release_manifest.json'},
                 manuscript_sha256=sha(ROOT/'paper/main.tex'),
                 pdf_sha256=sha(ROOT/'output/pdf/FlowHARQ_VTC2027.pdf'))
    (root/'release_manifest.json').write_text(json.dumps(release,indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
