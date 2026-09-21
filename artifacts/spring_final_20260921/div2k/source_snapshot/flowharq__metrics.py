import torch
from torch import Tensor
import torch.nn.functional as F


def psnr(reference: Tensor, reconstruction: Tensor) -> Tensor:
    mse = (reference - reconstruction).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def ssim(
    reference: Tensor,
    reconstruction: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> Tensor:
    """Standard Gaussian-window SSIM, returned independently per image."""
    dtype = reference.dtype
    coordinates = torch.arange(window_size, device=reference.device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = reference.shape[1]
    kernel = kernel_2d.expand(channels, 1, -1, -1)
    padding = window_size // 2

    mu_x = F.conv2d(reference, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(reconstruction, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(reference.square(), kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(reconstruction.square(), kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(reference * reconstruction, kernel, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ).clamp_min(1e-12)
    return score.flatten(1).mean(dim=1)


def mask_f1(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.bool()
    target = target.bool()
    tp = (prediction & target).sum(dim=1).float()
    fp = (prediction & ~target).sum(dim=1).float()
    fn = (~prediction & target).sum(dim=1).float()
    return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
