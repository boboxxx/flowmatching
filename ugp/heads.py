import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )


class RiskAwareProtection(nn.Module):
    """Importance + predictive uncertainty + fixed-budget UEP."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 4,
        hidden_dim: int = 64,
        average_power: float = 1.0,
        minimum_power: float = 0.2,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= minimum_power < average_power:
            raise ValueError("minimum_power must be in [0, average_power)")
        self.importance = MLP(latent_dim, hidden_dim, 1)
        self.uncertainty = MLP(latent_dim + context_dim, hidden_dim, 1)
        self.risk = MLP(2 + context_dim, hidden_dim, 1)
        self.average_power = average_power
        self.minimum_power = minimum_power
        self.temperature = temperature
        self.eps = eps

    def forward(self, tokens: Tensor, context: Tensor, mode: str = "full"):
        if context.ndim != 2 or context.shape[0] != tokens.shape[0]:
            raise ValueError("context must have shape [B, C]")
        expanded = context[:, None, :].expand(-1, tokens.shape[1], -1)
        importance = torch.sigmoid(self.importance(tokens)).squeeze(-1)
        variance = F.softplus(
            self.uncertainty(torch.cat((tokens, expanded), dim=-1))
        ).squeeze(-1) + self.eps

        if mode == "equal":
            risk = torch.zeros_like(importance)
            power = torch.full_like(importance, self.average_power)
        else:
            importance_in = importance if mode != "uncertainty_only" else torch.zeros_like(importance)
            uncertainty_in = variance.log() if mode != "importance_only" else torch.zeros_like(variance)
            risk = self.risk(
                torch.cat(
                    (importance_in[..., None], uncertainty_in[..., None], expanded),
                    dim=-1,
                )
            ).squeeze(-1)
            weights = torch.softmax(risk / self.temperature, dim=1)
            n = tokens.shape[1]
            residual = n * (self.average_power - self.minimum_power)
            power = self.minimum_power + residual * weights
        return {
            "importance": importance,
            "uncertainty": variance,
            "risk": risk,
            "power": power,
        }


def heteroscedastic_error_nll(variance: Tensor, squared_error: Tensor) -> Tensor:
    target = squared_error.detach()
    return (0.5 * target / variance + 0.5 * variance.log()).mean()

