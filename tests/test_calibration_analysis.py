import csv

import pytest

from flowharq.analyze import interval
from flowharq.calibrate import evaluate


def test_student_interval_is_centered_on_mean():
    result = interval([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["ci_low"] < 2.0 < result["ci_high"]


def test_calibration_bias_uses_raw_predictions(tmp_path):
    destination = tmp_path / "calibration.csv"
    fieldnames = [
        "image",
        "method",
        "snr_db",
        "speed_kmh",
        "csi_age_ms",
        "psnr",
        "ssim",
        "lpips",
        "predicted_direct_psnr_raw",
        "predicted_fm_psnr_raw",
        "split",
        "flow_steps",
        "mask_threshold",
        "training_seed",
        "channel_seed",
    ]
    rows = []
    for name, direct_psnr, fm_psnr in (("a.png", 20.0, 22.0), ("b.png", 26.0, 27.0)):
        for method, value, lpips in (
            ("direct", direct_psnr, 0.30),
            ("fm_only", fm_psnr, 0.25),
            ("full_harq", 28.0, 0.15),
        ):
            rows.append(
                {
                    "image": name,
                    "method": method,
                    "snr_db": 6,
                    "speed_kmh": 90,
                    "csi_age_ms": 2,
                    "psnr": value,
                    "ssim": 0.8,
                    "lpips": lpips,
                    "predicted_direct_psnr_raw": direct_psnr - 1.0,
                    "predicted_fm_psnr_raw": fm_psnr - 1.0,
                    "split": "calibration",
                    "flow_steps": 4,
                    "mask_threshold": 0.8,
                    "training_seed": 2027,
                    "channel_seed": 4040,
                }
            )
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = evaluate(str(destination), target_psnr=24.0)
    assert result["direct_quality_bias"] == pytest.approx(1.0)
    assert result["fm_quality_bias"] == pytest.approx(1.0)
    assert result["adaptive_harq"]["nack"] == pytest.approx(0.5)
    assert result["flowharq"]["nack"] == pytest.approx(0.5)
