"""Build an honest research status/report from archived completed artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path


def read_if(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,default=Path("artifacts/evidence_harq_20260920"))
    p.add_argument("--output",type=Path,default=Path("experiments/evidence_harq_20260920"))
    args=p.parse_args()
    candidates={}
    for name in ("candidate_a","candidate_b_failed","candidate_b_fixed"):
        candidates[name]=dict(status=read_if(args.artifacts/name/"status.json"),
                              gate=read_if(args.artifacts/name/"gate_decision.json"))
    baselines=[]
    for path in sorted((args.artifacts/"paid_baselines").glob("*.json")):
        meta=json.loads(path.read_text())
        if meta.get("status")!="completed":
            continue
        csv_path=path.with_suffix(".csv")
        if hashlib.sha256(csv_path.read_bytes()).hexdigest()!=meta["csv_sha256"]:
            raise ValueError(f"CSV integrity failure: {csv_path}")
        baselines.append(dict(model=meta["checkpoint"]["name"],images=meta["images"],
                              draws=meta["draws"],**meta["summary"]))
    cddm=read_if(args.artifacts/"cddm_kodak24.json")
    if cddm and cddm.get("status")=="completed":
        if hashlib.sha256((args.artifacts/"cddm_kodak24.csv").read_bytes()).hexdigest()!=cddm["csv_sha256"]:
            raise ValueError("CDDM CSV integrity failure")
        baselines.extend(cddm["summary"])
    report=dict(candidates=candidates,published_baselines=baselines,
                posterior_training="not_started_gate_required",policy_training="not_started_gate_required",
                conditional_5_percent_ack_certificate="not_established; current independent calibration split too small",
                overall="feasibility_study; no successful new-method or publication-ready claim")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"RESULTS.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Evidence-HARQ feasibility results — 2026-09-20", "",
           "This is a gated development study, not a completed posterior-HARQ paper. All negative",
           "and numerically failed candidates are retained. Published baselines and own controls",
           "are distinguished. One training seed (2027); channel draws are not training seeds.","",
           "## Codec feasibility", "",
           "| Candidate | Execution | Oracle saving vs adaptive base/full | Gate |",
           "|---|---|---:|---|"]
    for name,result in candidates.items():
        status=result["status"] or {}
        gate=result["gate"]
        saving=f'{100*gate["primary"]["savings_vs_adaptive_reference"]:.3f}%' if gate else "not available"
        lines.append(f'| {name} | {status.get("status","not archived")} | {saving} | {gate["status"] if gate else "not evaluated"} |')
    for name,result in candidates.items():
        if not result["gate"]:
            continue
        gate=result["gate"]["primary"]
        lines += ["",f"### {name}","",
                  "40 development images × 4 noise draws × 16 subsets; 2,560 reconstruction records.",
                  "Oracle knows the source and future noises and is not deployable. All failed images",
                  "remain in cost and outage averages. Quality matching additionally limits per-trial",
                  "PSNR loss to 0.1dB and LPIPS/MS-SSIM deterioration to 0.005.","",
                  "| Control | Total CBR | PSNR | MSE | MS-SSIM | LPIPS | Outage |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for model,values in gate["summaries"].items():
            lines.append(f'| {model} | {values["cbr_total"]:.6f} | {values["psnr"]:.3f} | {values["mse"]:.6f} | {values["ms_ssim"]:.5f} | {values["lpips"]:.5f} | {100*values["outage"]:.2f}% |')
        failed=[key for key,passed in gate["criteria"].items() if not passed]
        lines += ["","Failed checks: "+(", ".join(failed) if failed else "none")+"."]
    lines += ["","## Published author baselines with paid metadata","",
              "Kodak24 center256, AWGN10dB; 10 paired channel draws per image. All weights are",
              "released author checkpoints; scale uses IEEE binary16 and metadata/control is charged",
              "at 0.5 information bits per complex channel use. NTSCC additionally pays for its rate",
              "map. These differ from the previously archived author-native free-scale assumptions.","",
              "Pairing means common image/draw identifiers and integer noise seeds; different",
              "architectures and packet dimensions do not receive identical-shaped noise tensors.","",
              "| Method | Payload CBR | Total CBR | PSNR | MSE | MS-SSIM | LPIPS |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in baselines:
        lines.append(f'| {row["model"]} | {row["cbr_payload"]:.6f} | {row["cbr_total"]:.6f} | {row["psnr"]:.3f} | {row["mse"]:.6f} | {row["ms_ssim"]:.5f} | {row["lpips"]:.5f} |')
    lines += ["","CDDM's two JSCC controls belong to the same published architecture study and do",
              "not represent two additional published methods. Different payload rates must be",
              "shown as frontier points rather than interpreted as equal-rate rankings. Ideal-output-",
              "feedback DeepJSCC-f remains in its original separate result package.","",
              "## Scope of conclusions","",
              "Failure of a learned progressive codec prevents interpreting its oracle as the ceiling",
              "of all active-HARQ systems. Stop this candidate's posterior/policy stage if its gate",
              "fails. A new codec construction requires its own declared study. Reusing the observed",
              "40 images for development cannot establish confirmatory performance. The current",
              "40-image risk-calibration pilot cannot certify 5% conditional false-ACK risk at 95%",
              "confidence, even at zero observed failures. No such guarantee is claimed.","",
              "SAFG-HARQ publication is verified, but its full journal implementation remains",
              "unreproduced; do not substitute a home-made selection rule under that name.","",
              "See `DECISION.md` for the stopped-candidate decision and prerequisites of a new study.","",
              "See `protocol.md`, `SYSTEM_MODEL.md`, `AMENDMENT_01.md`, `NUMERICAL_FIX_B.md`,",
              "`LITERATURE_NOTES.md`, and `REPRODUCE.md` for provenance and assumptions.",""]
    (args.output/"RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(candidates=list(candidates),completed_baseline_rows=len(baselines))))


if __name__=="__main__":
    main()
