from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _expand_context(context: Tensor, token_count: int) -> Tensor:
    if context.ndim != 2:
        raise ValueError("context must have shape [B, C]")
    return context[:, None, :].expand(-1, token_count, -1)


class ReliabilityEstimator(nn.Module):
    """Predict the probability that each received semantic token is unreliable."""

    def __init__(self, latent_dim: int, context_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + context_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, received: Tensor, context: Tensor) -> Tensor:
        if received.ndim != 3:
            raise ValueError("received must have shape [B, N, D]")
        normalized = self.norm(received)
        global_token = normalized.mean(dim=1, keepdim=True).expand_as(normalized)
        token_norm = received.square().mean(dim=-1, keepdim=True).sqrt()
        deviation = (normalized - global_token).square().mean(dim=-1, keepdim=True)
        features = torch.cat(
            (normalized, global_token, token_norm, deviation, _expand_context(context, received.shape[1])),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class FlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.attn_norm(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.ffn(self.ffn_norm(x))


class ReliabilityAnchoredFlow(nn.Module):
    """Conditional velocity field that can only move masked latent tokens.

    The mask is applied outside the velocity network as a hard architectural
    constraint, so reliable tokens remain bitwise unchanged during integration.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 4,
        hidden_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        # z_t, received anchor, reliable global anchor, context, mask, and time features.
        input_dim = 3 * latent_dim + context_dim + 1 + 3
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            FlowBlock(hidden_dim, heads, dropout) for _ in range(depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    @staticmethod
    def time_features(t: Tensor, token_count: int) -> Tensor:
        if t.ndim == 1:
            t = t[:, None, None]
        elif t.ndim == 2:
            t = t[:, :, None]
        phase = 2.0 * math.pi * t
        features = torch.cat((t, phase.sin(), phase.cos()), dim=-1)
        return features.expand(-1, token_count, -1)

    @staticmethod
    def reliable_anchor(received: Tensor, mask: Tensor) -> Tensor:
        reliable = 1.0 - mask
        denominator = reliable.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (received * reliable[..., None]).sum(dim=1, keepdim=True)
        return (pooled / denominator[..., None]).expand_as(received)

    def velocity(
        self,
        z_t: Tensor,
        t: Tensor,
        received: Tensor,
        mask: Tensor,
        context: Tensor,
    ) -> Tensor:
        n = z_t.shape[1]
        features = torch.cat(
            (
                z_t,
                received,
                self.reliable_anchor(received, mask),
                _expand_context(context, n),
                mask[..., None],
                self.time_features(t, n),
            ),
            dim=-1,
        )
        hidden = self.input(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)

    def matching_loss(
        self,
        clean: Tensor,
        received: Tensor,
        mask: Tensor,
        context: Tensor,
        loss_type: str = "mse",
        time_mode: str = "uniform",
    ) -> Tensor:
        batch = clean.shape[0]
        if time_mode == "uniform":
            t = torch.rand(batch, device=clean.device, dtype=clean.dtype)
        elif time_mode == "zero":
            t = torch.zeros(batch, device=clean.device, dtype=clean.dtype)
        else:
            raise ValueError(f"unknown flow-matching time mode: {time_mode}")
        delta = clean - received
        z_t = received + t[:, None, None] * delta * mask[..., None]
        prediction = self.velocity(z_t, t, received, mask, context)
        if loss_type == "mse":
            token_loss = (prediction - delta).square().mean(dim=-1)
        elif loss_type == "normalized_huber":
            # ZF can create a few very large latent residuals in deep fades.
            # Normalize only those large targets, without amplifying already-small
            # residuals, and use a robust penalty.  The detached scale prevents the
            # target magnitude from becoming an optimization shortcut.
            scale = delta.square().mean(dim=-1, keepdim=True).sqrt().detach().clamp_min(1.0)
            token_loss = F.smooth_l1_loss(
                prediction / scale,
                delta / scale,
                beta=0.1,
                reduction="none",
            ).mean(dim=-1)
        else:
            raise ValueError(f"unknown flow-matching loss: {loss_type}")
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)

    def integrate(
        self,
        received: Tensor,
        mask: Tensor,
        context: Tensor,
        steps: int = 4,
        time_mode: str = "midpoint",
    ) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        z = received
        dt = 1.0 / steps
        for index in range(steps):
            if time_mode == "midpoint":
                time_value = (index + 0.5) * dt
            elif time_mode == "zero":
                time_value = 0.0
            else:
                raise ValueError(f"unknown integration time mode: {time_mode}")
            t = torch.full(
                (received.shape[0],),
                time_value,
                device=received.device,
                dtype=received.dtype,
            )
            z = z + dt * self.velocity(z, t, received, mask, context) * mask[..., None]
        # Re-apply the constraint to avoid any numerical drift on reliable tokens.
        return received + (z - received) * mask[..., None]


class QualityPredictor(nn.Module):
    """Predict per-image log-MSE for receiver-side ACK/NACK decisions."""

    def __init__(self, latent_dim: int, context_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.token = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim + context_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: Tensor, unreliability: Tensor, context: Tensor) -> Tensor:
        hidden = self.token(tokens)
        pooled = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=-1)
        confidence = torch.stack(
            (unreliability.mean(dim=1), unreliability.amax(dim=1)), dim=-1
        )
        return self.output(torch.cat((pooled, context, confidence), dim=-1)).squeeze(-1)

    @staticmethod
    def target_log_mse(reference: Tensor, reconstruction: Tensor) -> Tensor:
        mse = (reference - reconstruction).square().flatten(1).mean(dim=1)
        return mse.clamp_min(1e-8).log()

    @staticmethod
    def log_mse_to_psnr(log_mse: Tensor) -> Tensor:
        return -10.0 / math.log(10.0) * log_mse


def quality_boundary_bce(
    predicted_log_mse: Tensor,
    target_log_mse: Tensor,
    target_psnr: float,
    temperature_db: float = 1.0,
) -> Tensor:
    """Binary NACK loss focused on a receiver service-quality boundary."""
    if temperature_db <= 0:
        raise ValueError("temperature_db must be positive")
    predicted_psnr = QualityPredictor.log_mse_to_psnr(predicted_log_mse)
    actual_psnr = QualityPredictor.log_mse_to_psnr(target_log_mse)
    nack_logits = (target_psnr - predicted_psnr) / temperature_db
    nack_target = (actual_psnr < target_psnr).to(predicted_log_mse.dtype)
    return F.binary_cross_entropy_with_logits(nack_logits, nack_target)


def oracle_unreliable_mask(clean: Tensor, received: Tensor, fraction: float = 0.35) -> tuple[Tensor, Tensor]:
    """Return a per-sample top-error mask and its continuous token errors."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    error = (clean - received).square().mean(dim=-1)
    count = max(1, math.ceil(error.shape[1] * fraction))
    indices = error.topk(count, dim=1).indices
    mask = torch.zeros_like(error)
    mask.scatter_(1, indices, 1.0)
    return mask, error


def balanced_reliability_bce(logits: Tensor, target: Tensor) -> Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).detach()
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
