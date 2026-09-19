import torch

from flowharq.modules import (
    QualityPredictor,
    ReliabilityAnchoredFlow,
    ReliabilityEstimator,
    oracle_unreliable_mask,
    quality_boundary_bce,
)


def test_oracle_mask_selects_requested_top_fraction():
    clean = torch.zeros(2, 10, 4)
    received = torch.arange(20, dtype=torch.float32).reshape(2, 10, 1).expand(-1, -1, 4)
    mask, error = oracle_unreliable_mask(clean, received, fraction=0.3)
    assert mask.shape == error.shape == (2, 10)
    assert torch.equal(mask.sum(dim=1), torch.tensor([3.0, 3.0]))
    assert torch.equal(mask[:, -3:], torch.ones(2, 3))


def test_reliable_tokens_are_exactly_anchored():
    torch.manual_seed(1)
    flow = ReliabilityAnchoredFlow(latent_dim=8, hidden_dim=32, depth=1, heads=4)
    received = torch.randn(2, 9, 8)
    mask = torch.zeros(2, 9)
    mask[:, :3] = 1.0
    context = torch.randn(2, 4)
    repaired = flow.integrate(received, mask, context, steps=3)
    assert torch.equal(repaired[:, 3:], received[:, 3:])
    assert not torch.equal(repaired[:, :3], received[:, :3])


def test_heads_have_expected_shapes_and_gradients():
    torch.manual_seed(2)
    received = torch.randn(2, 16, 8, requires_grad=True)
    context = torch.randn(2, 4)
    reliability = ReliabilityEstimator(latent_dim=8, hidden_dim=32)
    quality = QualityPredictor(latent_dim=8, hidden_dim=32)
    logits = reliability(received, context)
    prediction = quality(received, logits.sigmoid(), context)
    assert logits.shape == (2, 16)
    assert prediction.shape == (2,)
    prediction.sum().backward()
    assert received.grad is not None


def test_flow_matching_loss_is_finite():
    torch.manual_seed(3)
    clean = torch.randn(2, 9, 8)
    received = clean + 0.2 * torch.randn_like(clean)
    mask, _ = oracle_unreliable_mask(clean, received, fraction=0.4)
    context = torch.randn(2, 4)
    flow = ReliabilityAnchoredFlow(latent_dim=8, hidden_dim=32, depth=1, heads=4)
    loss = flow.matching_loss(clean, received, mask, context)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in flow.parameters())


def test_normalized_huber_flow_loss_handles_deep_fade_outlier():
    torch.manual_seed(4)
    clean = torch.randn(2, 9, 8)
    received = clean + 0.2 * torch.randn_like(clean)
    received[:, 0] += 100.0
    mask = torch.ones(2, 9)
    context = torch.randn(2, 4)
    flow = ReliabilityAnchoredFlow(latent_dim=8, hidden_dim=32, depth=1, heads=4)
    mse = flow.matching_loss(clean, received, mask, context, loss_type="mse")
    robust = flow.matching_loss(
        clean, received, mask, context, loss_type="normalized_huber"
    )
    assert torch.isfinite(robust)
    assert robust < mse
    robust.backward()
    assert any(parameter.grad is not None for parameter in flow.parameters())


def test_unknown_flow_loss_is_rejected():
    flow = ReliabilityAnchoredFlow(latent_dim=8, hidden_dim=32, depth=1, heads=4)
    clean = torch.randn(1, 4, 8)
    received = torch.randn_like(clean)
    mask = torch.ones(1, 4)
    context = torch.randn(1, 4)
    try:
        flow.matching_loss(clean, received, mask, context, loss_type="not-a-loss")
    except ValueError as error:
        assert "unknown flow-matching loss" in str(error)
    else:
        raise AssertionError("invalid loss type should raise ValueError")


def test_quality_boundary_loss_prefers_correct_ack_side():
    factor = -10.0 / torch.log(torch.tensor(10.0))
    actual_psnr = torch.tensor([22.0, 26.0])
    target_log_mse = actual_psnr / factor
    correct_prediction = torch.tensor([22.0, 26.0]) / factor
    wrong_prediction = torch.tensor([26.0, 22.0]) / factor
    correct = quality_boundary_bce(correct_prediction, target_log_mse, 24.0)
    wrong = quality_boundary_bce(wrong_prediction, target_log_mse, 24.0)
    assert correct < wrong


def test_quality_boundary_loss_rejects_nonpositive_temperature():
    values = torch.zeros(2)
    try:
        quality_boundary_bce(values, values, 24.0, temperature_db=0.0)
    except ValueError as error:
        assert "temperature_db" in str(error)
    else:
        raise AssertionError("nonpositive temperature should raise ValueError")
