import csv

import numpy as np

from flowharq.compare_repair import interval, paired_rows


def test_interval_contains_mean():
    values = np.asarray([0.1, 0.2, 0.3])
    low, high = interval(values, 1000, np.random.default_rng(1))
    assert low <= values.mean() <= high


def test_paired_rows_matches_methods(tmp_path):
    path = tmp_path / "samples.csv"
    fields = [
        "image",
        "snr_db",
        "speed_kmh",
        "csi_age_ms",
        "channel_seed",
        "method",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in ("fm_only", "direct"):
            writer.writerow(
                {
                    "image": "x.png",
                    "snr_db": "6",
                    "speed_kmh": "60",
                    "csi_age_ms": "2",
                    "channel_seed": "9",
                    "method": method,
                }
            )
    pairs = paired_rows(str(path))
    assert len(pairs) == 1
    assert pairs[0][0]["method"] == "direct"
    assert pairs[0][1]["method"] == "fm_only"
