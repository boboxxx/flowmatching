import torch
from torch import Tensor


def psnr(reference: Tensor, reconstruction: Tensor) -> Tensor:
    mse = (reference - reconstruction).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def spearman_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    def ranks(x):
        order = x.argsort(dim=1)
        result = torch.empty_like(x)
        values = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[None].expand_as(x)
        result.scatter_(1, order, values)
        return result

    x = ranks(prediction)
    y = ranks(target)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    return (x * y).sum(dim=1) / (
        x.square().sum(dim=1).sqrt() * y.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)

