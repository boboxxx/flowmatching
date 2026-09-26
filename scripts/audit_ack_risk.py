"""Frozen, post-hoc decision diagnostic. Selection reads calibration only.

No torch dependency; intended to run on Sheng against the archived H2 CSVs.
This reports empirical risk, not a finite-sample reliability certificate.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

MARGINS = (0., .25, .5, 1., 2., 4., None)  # None means always NACK.
POLICIES = {'adaptive_harq': ('direct', 'direct'), 'flowharq': ('fm_only', 'fm')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(paths, split):
    cases = defaultdict(dict)
    for path in paths:
        with Path(path).open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['method'] not in ('direct', 'fm_only', 'full_harq'):
                    continue
                if row['split'] != split:
                    raise ValueError('Unexpected split: ' + row['split'])
                if int(row['training_seed']) != 2030 or int(row['flow_steps']) != 4:
                    raise ValueError('Wrong frozen receiver/configuration')
                if float(row['mask_threshold']) != .8 or float(row['target_psnr']) != 24:
                    raise ValueError('Wrong mask/target')
                key = tuple(row[k] for k in ('image', 'snr_db', 'speed_kmh', 'channel_seed'))
                if row['method'] in cases[key]:
                    raise ValueError('Duplicate case: ' + str(key))
                for metric in ('psnr', 'lpips', 'ssim', 'predicted_direct_psnr_raw', 'predicted_fm_psnr_raw'):
                    if not math.isfinite(float(row[metric])):
                        raise ValueError('Non-finite metric: ' + metric)
                cases[key][row['method']] = row
    if not cases:
        raise ValueError('Empty input')
    for group in cases.values():
        if set(group) != {'direct', 'fm_only', 'full_harq'}:
            raise ValueError('Incomplete paired case')
        for key in ('first_gain_sq', 'second_gain_sq', 'predicted_direct_psnr_raw', 'predicted_fm_psnr_raw'):
            if len({row[key] for row in group.values()}) != 1:
                raise ValueError('Unpaired observations/scores: ' + key)
    return cases


def evaluate(cases, policy, margin, bias):
    first_method, score_name = POLICIES[policy]
    ack = failures = outage = 0
    totals = dict(psnr=0., mse=0., lpips=0., ssim=0.)
    for group in cases.values():
        first = group[first_method]
        accept = margin is not None and float(first['predicted_'+score_name+'_psnr_raw']) + bias >= 24 + margin
        delivered = first if accept else group['full_harq']
        ack += int(accept)
        failures += int(accept and float(first['psnr']) < 24)
        outage += int(float(delivered['psnr']) < 24)
        for metric in ('psnr', 'lpips', 'ssim'):
            totals[metric] += float(delivered[metric])
        # Per-image MSE, never 10**(-mean PSNR/10).
        totals['mse'] += 10 ** (-float(delivered['psnr']) / 10)
    n = len(cases)
    nack = 1 - ack/n
    return dict(policy=policy, margin_db=margin, always_nack=margin is None,
                cases=n, ack_count=ack, ack_coverage=ack/n, nack=nack,
                joint_false_ack=failures/n,
                conditional_false_ack=failures/ack if ack else None,
                final_outage=outage/n, charged_uses=4096*(1+nack)+8192+2*(1+nack),
                **{key: value/n for key, value in totals.items()})


def select(curve):
    for row in curve:
        if (row['margin_db'] is not None and row['ack_coverage'] >= .05
                and row['conditional_false_ack'] is not None
                and row['conditional_false_ack'] <= .05):
            return dict(margin_db=row['margin_db'], status='empirical_calibration_gate_passed')
    return dict(margin_db=None, status='infeasible_nontrivial_policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root
    calibration = root/'results/h2/seed_2030/calibration.csv'
    decision = root/'results/h2/seed_2030/decision.json'
    selected = json.loads(decision.read_text())['selected']
    if selected['training_seed'] != 2030 or selected['target_psnr'] != 24:
        raise ValueError('Wrong calibration record')
    cases = read_cases([calibration], 'calibration')
    names = {key[0] for key in cases}
    if len(names) != 20 or len(cases) != 480:
        raise ValueError('Expected 20 calibration images and 480 cases')
    frozen, rows = {}, []
    for policy, (_, key) in POLICIES.items():
        curve = [evaluate(cases, policy, margin, selected[key+'_quality_bias']) for margin in MARGINS]
        frozen[policy] = select(curve)
        rows.extend(dict(dataset='calibration', **row) for row in curve)
    # Test data are not loaded until the operating points above have been fixed.
    sources = [calibration, decision]
    for dataset, folder, count, seeds in [('div2k', 'test', 80, (8001,8002,8003)),
                                           ('kodak', 'kodak', 24, (9001,9002,9003))]:
        paths = [root/f'results/h2/{folder}/train_2030_channel_{seed}.csv' for seed in seeds]
        test = read_cases(paths, 'test' if dataset == 'div2k' else 'all')
        if names & {key[0] for key in test}:
            raise ValueError('Calibration/test image overlap')
        if len(test) != count*24*3 or len({key[0] for key in test}) != count:
            raise ValueError('Incomplete test grid')
        for policy, (_, key) in POLICIES.items():
            rows.extend(dict(dataset=dataset, **evaluate(test, policy, margin, selected[key+'_quality_bias']))
                        for margin in MARGINS)
        sources.extend(paths)
    for row in rows:
        row['calibration_selected'] = row['margin_db'] == frozen[row['policy']]['margin_db']
    args.output.mkdir(parents=True)
    with (args.output/'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = dict(status='completed', interpretation='post_hoc_empirical_not_risk_certified',
                  selected=frozen, python=sys.version, host=platform.node(),
                  invocation=sys.argv, source_sha256=sha(__file__),
                  inputs={str(path.relative_to(root)):sha(path) for path in sources},
                  csv_sha256=sha(args.output/'curves.csv'),
                  note='No MS-SSIM in archived rows. MSE reconstructed per row from PSNR. No test fitting.')
    (args.output/'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
