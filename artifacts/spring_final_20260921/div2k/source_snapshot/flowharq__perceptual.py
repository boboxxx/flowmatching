from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import Tensor, nn


class LPIPSMetric(nn.Module):
    """Standard learned perceptual image patch similarity (AlexNet backbone)."""

    def __init__(self, vendor: str = "vendor"):
        vendor_path = str(Path(vendor).resolve())
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is unavailable; install lpips or pass the correct --vendor path"
            ) from error
        super().__init__()
        self.metric = lpips.LPIPS(net="alex", verbose=False)
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad = False

    def forward(self, reference: Tensor, reconstruction: Tensor) -> Tensor:
        # LPIPS expects RGB tensors in [-1, 1].
        return self.metric(2.0 * reference - 1.0, 2.0 * reconstruction - 1.0).flatten()

