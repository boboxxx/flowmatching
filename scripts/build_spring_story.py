"""Rebuild manuscript evidence from frozen H2 CSVs; no model/threshold selection."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS=('nack','psnr','lpips','ssim','outage','false_ack')
T975={3:4.3026527299,10:2.2621571629}


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values):
    m=mean(values)
    half=T975[len(values)]*stdev(values)/math.sqrt(len(values))
    return dict(mean=m,ci_low=m-half,ci_high=m+half,n=len(values))


def dataset_result(directory, images):
    files=sorted(directory.glob('train_20*_channel_*.csv'))
    assert len(files)==30,(directory,len(files))
    cells=defaultdict(lambda:defaultdict(list))
    transition=defaultdict(list)
    seeds=defaultdict(lambda:defaultdict(list))
    sources=[]
    for path in files:
        rows=load_csv(path)
        assert len(rows)==images*24*7,(path,len(rows))
        sources.append(dict(path=str(path),sha256=sha(path),rows=len(rows)))
        trial=defaultdict(dict)
        for row in rows:
            row['outage']=int(float(row['psnr'])<24)
            row['false_ack']=int(int(row['nack'])==0 and row['outage'])
            key=(row['image'],row['snr_db'],row['speed_kmh'])
            assert row['method'] not in trial[key]
            trial[key][row['method']]=row
            seed=int(row['training_seed'])
            for metric in METRICS:
                cells[(seed,row['method'],row['snr_db'])][metric].append(float(row[metric]))
                seeds[(seed,row['method'])][metric].append(float(row[metric]))
        for group in trial.values():
            a,f=group['adaptive_harq'],group['flowharq']
            category={(0,0):'both_ack',(1,0):'saved_round',(0,1):'extra_round',(1,1):'both_nack'}[(int(a['nack']),int(f['nack']))]
            transition[category].append({m:float(f[m])-float(a[m]) for m in ('psnr','lpips','ssim')})
            if category=='both_nack':
                assert float(a['psnr'])==float(f['psnr']) and float(a['lpips'])==float(f['lpips'])
    cohorts={}
    for n in (3,10):
        allowed=range(2030,2030+n)
        aggregates={method:{m:mean(mean(seeds[(seed,method)][m]) for seed in allowed) for m in METRICS}
                    for method in ('direct','full_harq','fm_only','adaptive_harq','flowharq')}
        for metrics in aggregates.values():
            metrics['conditional_false_ack']=metrics['false_ack']/(1-metrics['nack']) if metrics['nack']<1 else None
        differences={m:estimate([mean(seeds[(s,'flowharq')][m])-mean(seeds[(s,'adaptive_harq')][m]) for s in allowed]) for m in METRICS}
        cohorts[str(n)]=dict(absolute=aggregates,difference=differences)
    per_snr=[]
    for snr in ('0.0','3.0','6.0','9.0','12.0','15.0'):
        per_snr.append(dict(snr=float(snr),**{method:{m:mean(mean(cells[(s,method,snr)][m]) for s in range(2030,2040)) for m in METRICS} for method in ('adaptive_harq','flowharq')}))
    total=sum(map(len,transition.values()))
    transitions={name:dict(count=len(values),probability=len(values)/total,
                          **{m+'_conditional_delta':mean(v[m] for v in values) for m in ('psnr','lpips','ssim')},
                          **{m+'_contribution':sum(v[m] for v in values)/total for m in ('psnr','lpips','ssim')}) for name,values in transition.items()}
    for m in ('psnr','lpips','ssim'):
        assert abs(sum(row[m+'_contribution'] for row in transitions.values())-cohorts['10']['difference'][m]['mean'])<1e-10
    a,f=cohorts['10']['absolute']['adaptive_harq']['nack'],cohorts['10']['absolute']['flowharq']['nack']
    overhead=[]
    for eta in (.25,.5,1.,2.,4.):
        control=math.ceil(1/eta)
        metadata=math.ceil(256*16/eta)
        ca=4096*(1+a)+metadata+control*(1+a)
        cf=4096*(1+f)+metadata+control*(1+f)
        cra=(4096+metadata+control)*(1+a)
        crf=(4096+metadata+control)*(1+f)
        overhead.append(dict(eta=eta,adaptive_uses=ca,flow_uses=cf,relative_saving=(ca-cf)/ca,
                             metadata_once=True,repeated_metadata_saving=(cra-crf)/cra))
    return dict(images=images,cohorts=cohorts,per_snr=per_snr,transitions=transitions,overhead=overhead,
                payload_relative_saving=(a-f)/(1+a),second_round_relative_saving=(a-f)/a,sources=sources)


def precision_result(path, archive):
    if not path.exists():
        return None
    meta=json.loads((path/'status.json').read_text())
    assert meta['status']=='completed' and meta['csv_sha256']==sha(path/'raw.csv')
    rows=load_csv(path/'raw.csv')
    provenance=json.loads((path/'provenance.json').read_text())
    inputs=json.loads((path/'inputs.json').read_text())
    assert len(rows)==len(inputs['images'])*24*2*5
    sums=defaultdict(list)
    cases=defaultdict(dict)
    for row in rows:
        assert abs(float(row['psnr'])+10*math.log10(float(row['mse'])))<1e-4
        assert 0<=float(row['ms_ssim'])<=1
        assert int(row['total_uses'])==4096*(1+int(row['nack']))+8192+2*(1+int(row['nack']))
        assert row['scale_mode'] in ('ideal','binary16')
        sums[(row['scale_mode'],row['method'])].append(row)
        key=(row['image'],row['snr_db'],row['speed_kmh'],row['method'])
        assert row['scale_mode'] not in cases[key]
        cases[key][row['scale_mode']]=row
        assert int(row['false_ack'])==int(int(row['nack'])==0 and float(row['psnr'])<24)
    assert all(set(v)=={'ideal','binary16'} for v in cases.values())
    summary={}
    for (mode,method),group in sums.items():
        avg={m:mean(float(r[m]) for r in group) for m in ('psnr','mse','lpips','ms_ssim','nack','total_uses','outage','false_ack')}
        ack=sum(int(r['nack'])==0 for r in group)
        avg['conditional_false_ack']=sum(int(r['false_ack']) for r in group)/ack if ack else None
        summary[mode+'/'+method]=avg
    changed={method:sum(a['ideal']['nack']!=a['binary16']['nack'] for key,a in cases.items() if key[3]==method) for method in ('adaptive_harq','flowharq')}
    maximum={m:max(abs(float(a['ideal'][m])-float(a['binary16'][m])) for a in cases.values()) for m in ('psnr','mse','lpips','ms_ssim')}
    original={(r['image'],r['snr_db'],r['speed_kmh'],r['method']):r for r in load_csv(archive)}
    replay={m:max(abs(float(value['ideal'][m])-float(original[key][m])) for key,value in cases.items()) for m in ('psnr','lpips','nack')}
    return dict(summary=summary,changed_decisions=changed,max_quantization_difference=maximum,max_archived_replay_difference=replay,
                device=provenance.get('device','cuda'),archived_replay_expected=provenance.get('device','cuda')=='cuda',
                rows=len(rows),csv_sha256=meta['csv_sha256'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    args=p.parse_args()
    root=args.root
    out=root/'experiments/spring_final_20260921'
    out.mkdir(parents=True,exist_ok=True)
    result={name:dataset_result(root/f'results/h2/{folder}',count) for name,folder,count in [('div2k','test',80),('kodak','kodak',24)]}
    audits=root/'artifacts/spring_final_20260921'
    for name,folder,seed in [('div2k','test',8001),('kodak','kodak',9001)]:
        result[name]['precision_audit']=precision_result(audits/name,root/f'results/h2/{folder}/train_2030_channel_{seed}.csv')
    h4=json.loads((root/'results/h4/comparison.json').read_text())
    result['h4']=h4
    result['source_policy']='Original three-seed confirmation and locked ten-seed extension remain distinct. All metadata-efficiency points are reported; none selects a model.'
    (out/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    generated=root/'paper/generated'
    generated.mkdir(exist_ok=True)
    table=['\\begin{tabular}{llrrr}','\\toprule',r'Dataset & Runs & Saving (pp) [95\% CI] & $\Delta$PSNR & $\Delta$LPIPS \\',r'\midrule']
    macros=['% Generated by scripts/build_spring_story.py; no manually chosen result values.']
    for name,label,prefix in [('div2k','DIV2K','Div'),('kodak','Kodak24','Kod')]:
        for n in ('3','10'):
            d=result[name]['cohorts'][n]['difference']; r=d['nack']
            table.append(f'{label} & {n} & {-100*r["mean"]:.3f} [{-100*r["ci_high"]:.3f}, {-100*r["ci_low"]:.3f}] & {d["psnr"]["mean"]:+.4f} & {d["lpips"]["mean"]:+.5f} '+r'\\')
        for key,value in [('Saving',-100*result[name]['cohorts']['10']['difference']['nack']['mean']),
                          ('PayloadSaving',100*result[name]['payload_relative_saving']),('TotalSaving',100*result[name]['overhead'][1]['relative_saving']),
                          ('RequestSaving',100*result[name]['second_round_relative_saving']),
                          ('AdaptiveNack',100*result[name]['cohorts']['10']['absolute']['adaptive_harq']['nack']),
                          ('FlowNack',100*result[name]['cohorts']['10']['absolute']['flowharq']['nack'])]:
            macros.append(f'\\newcommand{{\\{prefix}{key}}}{{{value:.3f}}}')
    table += [r'\bottomrule',r'\end{tabular}']
    (generated/'spring_main_table.tex').write_text('\n'.join(table)+'\n')
    (generated/'spring_macros.tex').write_text('\n'.join(macros)+'\n')
    external=root/'artifacts/evidence_harq_20260920'
    published=[]
    for key,label in [('swin_mse_c32','SwinJSCC C32'),('swin_mse_c64','SwinJSCC C64'),('ntscc_quality1','NTSCC Q1'),('ntscc_quality2','NTSCC Q2')]:
        meta=json.loads((external/'paid_baselines'/f'{key}.json').read_text())
        source=external/'paid_baselines'/f'{key}.csv'
        assert meta['csv_sha256']==sha(source)
        rows=load_csv(source)
        assert len(rows)==240
        metrics={m:mean(float(r[m]) for r in rows) for m in ('psnr','mse','ms_ssim','lpips','cbr_total')}
        published.append((label,metrics))
    cddm_meta=json.loads((external/'cddm_kodak24.json').read_text())
    assert cddm_meta['csv_sha256']==sha(external/'cddm_kodak24.csv')
    rows=[r for r in load_csv(external/'cddm_kodak24.csv') if r['model']=='CDDM_author_C36_paid_scale']
    assert len(rows)==240
    published.append(('CDDM C36',{m:mean(float(r[m]) for r in rows) for m in ('psnr','mse','ms_ssim','lpips','cbr_total')}))
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Model & CBR & PSNR & MSE $\times10^3$ & MS-SSIM & LPIPS \\',r'\midrule']
    for label,r in published:
        lines.append(f'{label} & {r["cbr_total"]:.4f} & {r["psnr"]:.2f} & {1000*r["mse"]:.3f} & {r["ms_ssim"]:.4f} & {r["lpips"]:.4f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (generated/'spring_published_table.tex').write_text('\n'.join(lines)+'\n')
    if all(result[name]['precision_audit'] for name in ('div2k','kodak')):
        lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Policy & NACK (\%) & PSNR & MSE $\times10^3$ & MS-SSIM & LPIPS \\',r'\midrule']
        for name,label in [('div2k','DIV2K'),('kodak','Kodak24')]:
            audit=result[name]['precision_audit']
            lines.append(r'\multicolumn{6}{l}{\textit{'+label+r'}} \\')
            for key,policy in [('direct','Direct'),('full_harq','Full HARQ'),('fm_only','Repair only'),('adaptive_harq','Adaptive'),('flowharq','FlowHARQ')]:
                r=audit['summary']['binary16/'+key]
                lines.append(f'{policy} & {100*r["nack"]:.2f} & {r["psnr"]:.3f} & {1000*r["mse"]:.3f} & {r["ms_ssim"]:.4f} & {r["lpips"]:.4f} '+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (generated/'spring_precision_table.tex').write_text('\n'.join(lines)+'\n')
        da,ka=[result[n]['precision_audit'] for n in ('div2k','kodak')]
        changed=[sum(a['changed_decisions'].values()) for a in (da,ka)]
        max_psnr=max(a['max_quantization_difference']['psnr'] for a in (da,ka))
        paragraphs=[
            'The original archive records single-scale SSIM, not MS-SSIM. '
            'Table~\\ref{tab:precision} reports a separate frozen-checkpoint audit on Artemis: '
            'one existing trained receiver, one channel stream per dataset, all 24 SNR--speed '
            'conditions, and paired ideal/binary16 scale restoration. CPU random streams '
            'differ from the archived CUDA streams; this is a precision check, not their exact replay.',
            f'Quantization changes {changed[0]} of 3840 paired adaptive/FlowHARQ decisions on DIV2K '
            f'and {changed[1]} of 1152 on Kodak24. Across all five methods and both datasets, '
            f'the largest per-image PSNR change is {max_psnr:.4f} dB. '
            'PSNR and MSE alone do not establish perceptual benefit; the measured MS-SSIM '
            'and LPIPS must be considered separately. No threshold or weight is refitted '
            'using this audit, and it does not replace the multi-run confirmation.',
            r'\begin{table}[t]',r'\centering',
            r'\caption{Frozen binary16 audit under correlated Rayleigh fading. One existing receiver and one channel stream per dataset, with 1920 DIV2K and 576 Kodak image--channel cases per method. No uncertainty estimate or new training is inferred from these rows.}',
            r'\label{tab:precision}',r'\resizebox{\columnwidth}{!}{\input{generated/spring_precision_table}}',r'\end{table}',
        ]
        (generated/'spring_precision_text.tex').write_text('\n\n'.join(paragraphs)+'\n')
    # Report all results in compact human-readable form; raw evidence.json keeps precision.
    lines=['# Spring manuscript evidence audit','',
           'Frozen H2 results; original and expanded cohorts are distinct. No new training.','',
           '| Dataset | NACK saving (pp) | Payload saving (%) | Total saving at eta=0.5 (%) |',
           '|---|---:|---:|---:|']
    for name in ('div2k','kodak'):
        r=result[name]
        lines.append(f'| {name} | {-100*r["cohorts"]["10"]["difference"]["nack"]["mean"]:.6f} | {100*r["payload_relative_saving"]:.6f} | {100*r["overhead"][1]["relative_saving"]:.6f} |')
    for name in ('div2k','kodak'):
        r=result[name]
        lines += ['',f'## {name}: paired decision mechanism','',
                  '| Event | Fraction (%) | PSNR contribution (dB) | LPIPS contribution |',
                  '|---|---:|---:|---:|']
        for key,row in r['transitions'].items():
            lines.append(f'| {key} | {100*row["probability"]:.4f} | {row["psnr_contribution"]:+.6f} | {row["lpips_contribution"]:+.7f} |')
        lines += ['', '### Original ten-run stopping risk', '',
                  '| Policy | Final outage (%) | Joint false ACK (%) | Conditional false ACK (%) |',
                  '|---|---:|---:|---:|']
        for key in ('adaptive_harq','flowharq'):
            row=r['cohorts']['10']['absolute'][key]
            lines.append(f'| {key} | {100*row["outage"]:.4f} | {100*row["false_ack"]:.4f} | {100*row["conditional_false_ack"]:.4f} |')
        if r['precision_audit']:
            lines += ['', 'Finite-precision audit completed; changed ACK/NACK decisions: '+str(r['precision_audit']['changed_decisions'])+'.',
                      'Maximum ideal-branch difference from archived replay: '+str(r['precision_audit']['max_archived_replay_difference'])+'.']
            lines += ['', 'CPU draws differ from archived CUDA draws; the preceding replay differences are expected and are not a quantization error.', '',
                      '| Binary16 policy | NACK (%) | PSNR | MSE | MS-SSIM | LPIPS |',
                      '|---|---:|---:|---:|---:|---:|']
            for method in ('direct','full_harq','fm_only','adaptive_harq','flowharq'):
                row=r['precision_audit']['summary']['binary16/'+method]
                lines.append(f'| {method} | {100*row["nack"]:.4f} | {row["psnr"]:.6f} | {row["mse"]:.8f} | {row["ms_ssim"]:.6f} | {row["lpips"]:.6f} |')
    lines += ['', 'These costs assume reliable metadata/control and exclude pilots, headers, scheduling and retransmission of erroneous control messages.',
              'The finite-precision check uses one frozen training seed and one existing channel seed per dataset. It is not a replacement for the original confirmatory experiment.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({name:{'nack':result[name]['cohorts']['10']['difference']['nack'], 'payload_saving':result[name]['payload_relative_saving'],'total_saving':result[name]['overhead'][1]['relative_saving']} for name in ('div2k','kodak')},indent=2))


if __name__=='__main__':
    main()
