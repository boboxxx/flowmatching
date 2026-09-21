from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ChannelContext:
    """Per-image channel variables available to the transmitter."""

    snr_db: Tensor
    speed_kmh: Tensor
    csi_age_ms: Tensor
    rho: Tensor

    def features(self) -> Tensor:
        """Return bounded/scaled features used by the neural heads."""
        return torch.stack(
            (
                self.snr_db / 15.0,
                self.speed_kmh / 120.0,
                self.rho,
                self.csi_age_ms / 5.0,
            ),
            dim=-1,
        )


@dataclass
class ChannelObservation:
    """Equalized latent tokens and receiver-known channel state."""

    tokens: Tensor
    gain: Tensor
    gain_sq: Tensor
    effective_snr_db: Tensor
    equalized_noise_variance: Tensor

    def receiver_features(self, context: ChannelContext) -> Tensor:
        """Features available at the receiver, including instantaneous CSI."""
        gain_db = 10.0 * torch.log10(self.gain_sq.flatten(1).mean(dim=1).clamp_min(1e-8))
        return torch.cat(
            (
                context.features(),
                (gain_db / 20.0)[:, None],
                (self.effective_snr_db / 15.0)[:, None],
            ),
            dim=-1,
        )


def temporal_correlation(
    speed_kmh: Tensor,
    csi_age_ms: Tensor,
    carrier_hz: float = 5.9e9,
    light_speed: float = 299_792_458.0,
) -> Tensor:
    """Jakes correlation J0(2*pi*f_D*Delta)."""
    speed_ms = speed_kmh / 3.6
    doppler_hz = speed_ms * carrier_hz / light_speed
    argument = 2.0 * torch.pi * doppler_hz * csi_age_ms * 1e-3
    # Negative J0 values are valid correlations. Clamp only for numerical safety.
    return torch.special.bessel_j0(argument).clamp(-0.9999, 0.9999)


def sample_context(
    batch_size: int,
    device: torch.device,
    snr_range=(0.0, 15.0),
    speed_range=(0.0, 120.0),
    age_range=(0.0, 5.0),
) -> ChannelContext:
    def uniform(bounds):
        low, high = bounds
        return torch.empty(batch_size, device=device).uniform_(low, high)

    snr_db = uniform(snr_range)
    speed_kmh = uniform(speed_range)
    csi_age_ms = uniform(age_range)
    rho = temporal_correlation(speed_kmh, csi_age_ms)
    return ChannelContext(snr_db, speed_kmh, csi_age_ms, rho)


class TimeVaryingRayleigh(nn.Module):
    """Flat Rayleigh fading with temporally correlated HARQ rounds.

    Real-valued latent dimensions are paired into complex symbols.  The
    transmitter normalizes each token, applies the requested token power, and
    the receiver restores the original token scale after equalization.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _complex_normal(shape, device, dtype):
        real = torch.randn(shape, device=device, dtype=dtype)
        imag = torch.randn(shape, device=device, dtype=dtype)
        return torch.complex(real, imag) / (2.0**0.5)

    def forward_with_state(
        self,
        tokens: Tensor,
        power: Tensor,
        context: ChannelContext,
        previous_gain: Tensor | None = None,
        round_gap_ms: float = 1.0,
    ) -> ChannelObservation:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, D]")
        if tokens.shape[-1] % 2:
            raise ValueError("the latent width must be even for complex pairing")
        if power.shape != tokens.shape[:2]:
            raise ValueError("power must have shape [B, N]")

        b, n, d = tokens.shape
        output_dtype = tokens.dtype
        # ComplexHalf has incomplete operator coverage in PyTorch.  Keep the
        # differentiable channel simulation in complex64 under AMP.
        working = tokens.float() if tokens.dtype in (torch.float16, torch.bfloat16) else tokens
        working_power = power.to(working.dtype)
        paired = working.reshape(b, n, d // 2, 2)
        symbols = torch.complex(paired[..., 0], paired[..., 1])
        token_rms = symbols.abs().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        tx = symbols / token_rms * working_power.unsqueeze(-1).clamp_min(0.0).sqrt()

        innovation = self._complex_normal((b, 1, 1), working.device, working.dtype)
        if previous_gain is None:
            actual = innovation
        else:
            rho_round = temporal_correlation(
                context.speed_kmh,
                torch.full_like(context.speed_kmh, round_gap_ms),
            ).reshape(b, 1, 1).to(working.dtype)
            previous = previous_gain.to(device=working.device, dtype=actual_dtype(working))
            actual = rho_round * previous + (
                1.0 - rho_round.square()
            ).clamp_min(0.0).sqrt() * innovation

        snr_linear = torch.pow(10.0, context.snr_db.to(working.dtype) / 10.0)
        noise_std = (2.0 * snr_linear).reciprocal().sqrt().reshape(b, 1, 1)
        noise = torch.complex(
            torch.randn_like(tx.real) * noise_std,
            torch.randn_like(tx.real) * noise_std,
        )
        received = actual * tx + noise
        gain_sq = actual.abs().square()
        equalized = received * actual.conj() / gain_sq.add(self.eps)
        restored = equalized * token_rms
        restored_tokens = torch.stack((restored.real, restored.imag), dim=-1).reshape_as(working)
        effective_snr_db = context.snr_db.to(working.dtype) + 10.0 * torch.log10(
            gain_sq.flatten(1).mean(dim=1).clamp_min(self.eps)
        )
        noise_variance = noise_std.square() / gain_sq.add(self.eps)
        return ChannelObservation(
            tokens=restored_tokens.to(output_dtype),
            gain=actual,
            gain_sq=gain_sq,
            effective_snr_db=effective_snr_db,
            equalized_noise_variance=noise_variance,
        )

    def forward(self, tokens: Tensor, power: Tensor, context: ChannelContext) -> Tensor:
        return self.forward_with_state(tokens, power, context).tokens


def actual_dtype(reference: Tensor) -> torch.dtype:
    """Complex dtype corresponding to a real working tensor."""
    return torch.complex128 if reference.dtype == torch.float64 else torch.complex64
