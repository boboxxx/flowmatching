import torch
from evidence_harq.physical import LinkBudget, packet_awgn, subset_mask, enhancement_awgn
from evidence_harq.codec import EnhancementCodec


def test_unit_complex_power_and_paid_scale():
    z = torch.randn(3, 32, 16, 16) * 3
    received, info = packet_awgn(z, 10, torch.zeros_like(z))
    assert torch.allclose(info["normalized_power"], torch.ones(3), atol=1e-6)
    assert torch.allclose(z, received, atol=.008)
    budget = LinkBudget()
    assert budget.count(0)["total_uses"] == 4096 + 32 + 10
    assert budget.count(4)["total_uses"] == 8192 + 160 + 12
    assert budget.count(1)["total_uses"] > budget.count(0)["total_uses"]


def test_awgn_variance():
    z = torch.ones(32, 32, 16, 16)
    y, _ = packet_awgn(z, 10)
    # For real features, normalized complex noise variance / 2 times scale^2.
    assert abs((y-z).square().mean().item() - .1) < .003


def test_receiver_never_uses_unrequested_evidence():
    torch.manual_seed(7)
    model = EnhancementCodec(width=16)
    torch.nn.init.normal_(model.fusion[-1].weight, std=.01)
    y = torch.randn(2, 32, 16, 16)
    e = torch.randn_like(y)
    mask = subset_mask(torch.tensor([0, 5]))
    first = model.fuse(y, e, mask, 10)
    second = model.fuse(y, e + (1-mask)*100, mask, 10)
    assert torch.equal(first, second)
    assert torch.equal(first[0], y[0])
    assert mask[1].sum() == 128


def test_packet_noise_is_paired_and_group_local():
    e = torch.randn(2, 32, 16, 16)
    noise = torch.randn_like(e)
    first, _ = enhancement_awgn(e, 10, noise)
    second, _ = enhancement_awgn(e, 10, noise)
    assert torch.equal(first, second)


def test_bounded_evidence_has_finite_transmittable_scale():
    codec=EnhancementCodec(width=16,bounded_evidence=True)
    image=torch.rand(2,3,256,256)
    base=torch.randn(2,32,16,16)
    with torch.no_grad():
        codec.to_evidence[-1].weight.mul_(1e5)
        evidence=codec.encode(image,base)
        _,scale=enhancement_awgn(evidence,10)
    assert evidence.abs().max()<=1
    assert torch.isfinite(scale).all() and (scale>0).all()


def test_risk_calibration_does_not_invent_reliability():
    from evidence_harq.risk import calibrate_ack, upper_binomial
    assert upper_binomial(0,40,.05) > .05
    result=calibrate_ack(list(range(40)),[0.]*40,[False]*40,[.5])
    assert result["status"]=="abstain_no_certificate"
    result=calibrate_ack(list(range(100)),[0.]*100,[False]*100,[.5])
    assert result["status"]=="certified"


def test_repeated_noise_is_not_independent_calibration():
    import pytest
    from evidence_harq.risk import calibrate_ack
    with pytest.raises(ValueError,match="correlated"):
        calibrate_ack(["a","a"],[.1,.2],[False,False],[.5])


def test_oracle_counts_failed_cases_and_uses_adaptive_reference():
    from evidence_harq.oracle_gate import analyze
    rows=[]
    for image in ["easy","hard"]:
        for code in range(16):
            k=bin(code).count("1")
            rows.append(dict(image=image,snr_db=10.,draw=0.,subset=code,
                             psnr=30. if image=="easy" else 18.,mse=.001 if image=="easy" else .02,
                             lpips=.1 if image=="easy" else .6,ms_ssim=.95,
                             **LinkBudget().count(k)))
    result,_=analyze(rows)
    assert result["savings_vs_adaptive_reference"]==0
    assert result["summaries"]["selective_clairvoyant"]["outage"]==.5
    assert not result["passed"]
