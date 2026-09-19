import torch

from ugp.heads import RiskAwareProtection, heteroscedastic_error_nll


def test_power_budget_and_floor():
    head = RiskAwareProtection(8, average_power=1.0, minimum_power=0.2)
    result = head(torch.randn(3, 25, 8), torch.randn(3, 4))
    assert torch.all(result["power"] >= 0.2 - 1e-6)
    assert torch.allclose(result["power"].mean(dim=1), torch.ones(3), atol=1e-6)


def test_uncertainty_loss_has_gradients_only_through_prediction():
    variance = torch.ones(2, 4, requires_grad=True)
    target = torch.rand(2, 4, requires_grad=True)
    heteroscedastic_error_nll(variance, target).backward()
    assert variance.grad is not None
    assert target.grad is None

