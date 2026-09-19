"""Offline integrity/completeness audit of the released raw experiment records."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("psnr", "mse", "lpips", "ms_ssim", "cbr_total"):
            assert math.isfinite(float(row[key])), (path, key)
        assert float(row["mse"]) >= 0
        assert -1 <= float(row["ms_ssim"]) <= 1
    return rows


def audit(root):
    counts, retained_on_server = {}, []
    for run in ("candidate_a", "candidate_b_failed", "candidate_b_fixed"):
        directory = root/run
        inventory = json.loads((directory/"artifact_inventory.json").read_text())
        for entry in inventory["files"]:
            path = directory/entry["path"]
            if not path.exists():
                assert entry["path"] in ("best.pt", "last.pt"), path
                retained_on_server.append(f"{run}/{entry['path']}")
                continue
            assert path.stat().st_size == entry["bytes"], path
            assert sha(path) == entry["sha256"], path
        if run == "candidate_b_failed":
            assert json.loads((directory/"status.json").read_text())["status"] == "failed"
            continue
        assert (directory/"best.pt").exists()
        path = directory/"oracle_gate.csv"
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["csv_sha256"]
        rows = read_csv(path)
        trials = defaultdict(list)
        for row in rows:
            trials[(row["image_sha256"], row["draw"])].append(row)
            k = int(row["subset"]).bit_count() if hasattr(int, "bit_count") else bin(int(row["subset"])).count("1")
            total = 4096 + 1024*k + 32*(k+1) + 10 + 2*(k > 0)
            assert int(row["groups"]) == k
            assert abs(float(row["total_uses"])-total) < 1e-8
            assert abs(float(row["cbr_total"])-total/(3*256*256)) < 1e-12
        assert len(rows) == 2560 and len(trials) == 160
        for trial in trials.values():
            assert {int(row["subset"]) for row in trial} == set(range(16))
            assert len(trial) == 16 and len({row["noise_seed"] for row in trial}) == 1
        split = json.loads((directory/"split_manifest.json").read_text())
        seen = set()
        for records in split.values():
            hashes = {record["sha256"] for record in records}
            assert len(hashes) == len(records) and not hashes & seen
            seen |= hashes
        gate = json.loads((directory/"gate_decision.json").read_text())
        assert gate["source_sha256"] == sha(path)
        counts[run] = len(rows)
    all_baseline_keys = []
    for path in sorted((root/"paid_baselines").glob("*.csv")) + [root/"cddm_kodak24.csv"]:
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["status"] == "completed" and meta["csv_sha256"] == sha(path)
        rows = read_csv(path)
        groups = defaultdict(list)
        for row in rows:
            groups[row["model"]].append(row)
            overhead = sum(float(row.get(key, 0)) for key in ("scale_uses", "rate_map_uses", "feedback_uses"))
            assert abs(float(row["cbr_total"])-(float(row["payload_uses"])+overhead)/(3*256*256)) < 1e-12
        for model, group in groups.items():
            assert len(group) == 240
            keys = {(row["image_sha256"], row["draw"], row["noise_seed"]) for row in group}
            assert len(keys) == 240 and len({row["image_sha256"] for row in group}) == 24
            assert set(Counter(row["image_sha256"] for row in group).values()) == {10}
            all_baseline_keys.append(keys)
            summary = meta["summary"] if isinstance(meta["summary"], dict) else next(s for s in meta["summary"] if s["model"] == model)
            for key in ("psnr", "mse", "lpips", "ms_ssim", "cbr_payload", "cbr_total"):
                assert abs(mean(float(row[key]) for row in group)-summary[key]) < 1e-10, (path, key)
        counts[path.stem] = len(rows)
    assert len(all_baseline_keys) == 7 and all(keys == all_baseline_keys[0] for keys in all_baseline_keys)
    replay = read_csv(root/"replay_smoke.csv")
    original = {(row["image_sha256"], row["draw"], row["subset"]): row
                for row in read_csv(root/"candidate_b_fixed/oracle_gate.csv")}
    replay_max_error = 0.
    for row in replay:
        ref = original[(row["image_sha256"], row["draw"], row["subset"])]
        assert row["noise_seed"] == ref["noise_seed"]
        replay_max_error = max(replay_max_error, *(abs(float(row[key])-float(ref[key]))
                               for key in ("psnr", "mse", "lpips", "ms_ssim", "cbr_total")))
    assert len(replay) == 16 and replay_max_error < 1e-5
    runtime = json.loads((root/"runtime/runtime.json").read_text())
    for name, expected in runtime["source_sha256"].items():
        assert sha(Path(name)) == expected, name
    return dict(status="passed", row_counts=counts, total_reconstruction_rows=sum(counts.values()),
                relocated_replay_rows=len(replay), relocated_replay_max_metric_difference=replay_max_error,
                server_only_checkpoints=retained_on_server,
                interpretation="Integrity, accounting, split and completeness audit; not validation of scientific superiority.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts/evidence_harq_20260920"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.artifacts)
    text = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
