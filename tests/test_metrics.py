import torch

from flowharq.metrics import psnr, ssim


def test_ssim_identity_and_degradation():
    torch.manual_seed(11)
    reference = torch.rand(2, 3, 32, 32)
    degraded = (reference + 0.1 * torch.randn_like(reference)).clamp(0.0, 1.0)
    identity = ssim(reference, reference)
    noisy = ssim(reference, degraded)
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)
    assert torch.all(noisy < identity)
    assert torch.all(psnr(reference, reference) > psnr(reference, degraded))
