import torch

from flowharq.metrics import psnr, ssim, mse, ms_ssim, ms_ssim_db


def test_ssim_identity_and_degradation():
    torch.manual_seed(11)
    reference = torch.rand(2, 3, 32, 32)
    degraded = (reference + 0.1 * torch.randn_like(reference)).clamp(0.0, 1.0)
    identity = ssim(reference, reference)
    noisy = ssim(reference, degraded)
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)
    assert torch.all(noisy < identity)
    assert torch.all(psnr(reference, reference) > psnr(reference, degraded))


def test_mse_psnr_are_per_image_not_aggregate_transforms():
    reference = torch.zeros(2, 3, 16, 16)
    reconstruction = torch.stack([torch.full((3, 16, 16), .1),
                                  torch.full((3, 16, 16), .2)])
    errors = mse(reference, reconstruction)
    assert torch.allclose(errors, torch.tensor([.01, .04]))
    assert torch.allclose(psnr(reference, reconstruction), -10 * errors.log10())
    assert not torch.isclose(psnr(reference, reconstruction).mean(),
                             -10 * errors.mean().log10())


def test_multiscale_identity_and_official_reference():
    from pytorch_msssim import ms_ssim as reference_metric
    torch.manual_seed(19)
    reference = torch.rand(2, 3, 192, 192)
    noisy = (reference + .08 * torch.randn_like(reference)).clamp(0, 1)
    scores = ms_ssim(reference, noisy)
    assert scores.shape == (2,)
    assert torch.allclose(scores, reference_metric(reference, noisy, data_range=1,
                                                    size_average=False), atol=1e-6)
    assert torch.allclose(ms_ssim(reference, reference), torch.ones(2), atol=1e-6)
    assert torch.all(scores < 1)
    assert torch.isfinite(ms_ssim_db(torch.ones(2))).all()


def test_multiscale_does_not_silently_use_fewer_scales():
    import pytest
    with pytest.raises(ValueError, match="five-scale"):
        ms_ssim(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
