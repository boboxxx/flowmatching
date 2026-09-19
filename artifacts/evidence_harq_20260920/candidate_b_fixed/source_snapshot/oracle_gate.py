"""Auditable, source-aware feasibility upper bound; keeps failed outcomes in averages."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .train_codec import save_json


def passes(row, psnr_target, lpips_target):
    return row["psnr"] >= psnr_target and row["lpips"] <= lpips_target


def summarize(rows, psnr_target, lpips_target):
    names = ["psnr","mse","lpips","ms_ssim","total_uses","cbr_total"]
    return dict(**{key:float(np.mean([r[key] for r in rows])) for key in names},
                outage=float(np.mean([not passes(r, psnr_target,lpips_target) for r in rows])))


def analyze(rows, psnr_target=24., lpips_target=.3):
    trials = defaultdict(list)
    for row in rows:
        trials[(row["image"],row["snr_db"],row["draw"])].append(row)
    if not trials:
        raise ValueError("no oracle trials")
    selected, reference, full, base, choices = [], [], [], [], []
    for key, trial in sorted(trials.items()):
        by_code = {int(r["subset"]):r for r in trial}
        if set(by_code) != set(range(16)) or len(trial) != 16:
            raise ValueError(f"missing/duplicate subset in {key}")
        zero, all_groups = by_code[0], by_code[15]
        ref = zero if passes(zero,psnr_target,lpips_target) else all_groups
        feasible = [r for r in trial if passes(r,psnr_target,lpips_target)
                    and r["psnr"] >= ref["psnr"]-.1
                    and r["lpips"] <= ref["lpips"]+.005
                    and r["ms_ssim"] >= ref["ms_ssim"]-.005]
        chosen = min(feasible, key=lambda r:(r["total_uses"],-r["psnr"],r["subset"])) if feasible else ref
        selected.append(chosen)
        reference.append(ref)
        base.append(zero)
        full.append(all_groups)
        choices.append(dict(image=key[0],snr_db=key[1],draw=key[2],reference_subset=ref["subset"],
                            oracle_subset=chosen["subset"],uses_saved=ref["total_uses"]-chosen["total_uses"]))
    summaries = {name:summarize(items,psnr_target,lpips_target) for name,items in
                 (("selective_clairvoyant",selected),("adaptive_base_full_clairvoyant",reference),
                  ("always_full",full),("base_only",base))}
    a, b = summaries["selective_clairvoyant"], summaries["adaptive_base_full_clairvoyant"]
    saving = 1-a["total_uses"]/b["total_uses"]
    criteria = dict(saving_at_least_10_percent=saving>=.10,
                    nontrivial_service_coverage=b["outage"]<=.80,
                    no_outage_increase=a["outage"]<=b["outage"]+1e-12,
                    psnr_noninferior=a["psnr"]>=b["psnr"]-.100001,
                    lpips_noninferior=a["lpips"]<=b["lpips"]+.005001,
                    ms_ssim_noninferior=a["ms_ssim"]>=b["ms_ssim"]-.005001,
                    full_refines_base=summaries["always_full"]["mse"]<summaries["base_only"]["mse"])
    return dict(psnr_target=psnr_target,lpips_target=lpips_target,trial_count=len(trials),
                savings_vs_adaptive_reference=saving,criteria=criteria,passed=all(criteria.values()),
                summaries=summaries), choices


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    meta=json.loads(args.csv.with_suffix(".json").read_text())
    if meta["status"]!="completed" or meta["split"]!="oracle_gate":
        raise ValueError("Only complete registered gate data may make continuation decisions")
    from .data import sha256
    if sha256(args.csv)!=meta["csv_sha256"]:
        raise ValueError("evaluation CSV integrity failure")
    with args.csv.open() as handle:
        rows=[{k:v if k in ("image","image_sha256","split") else float(v) for k,v in row.items()}
              for row in csv.DictReader(handle)]
    primary=[r for r in rows if r["snr_db"]==10.]
    main_result, choices=analyze(primary)
    sensitivity=[analyze(primary,p,l)[0] for p in (22.,24.,26.) for l in (.25,.30,.35)]
    result=dict(stage="oracle_gate",status="pass" if main_result["passed"] else "stop_candidate",
                primary=main_result,sensitivity=sensitivity,
                interpretation="Clairvoyant bound for this codec, channel, and service, not a conclusion about all active HARQ designs.",
                source_csv=str(args.csv),source_sha256=sha256(args.csv))
    save_json(args.output,result)
    with args.output.with_suffix(".choices.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=choices[0])
        writer.writeheader()
        writer.writerows(choices)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
