"""Packet normalization and auditable resource accounting (complex channel uses)."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class LinkBudget:
    image_values: int = 3 * 256 * 256
    base_real_values: int = 32 * 16 * 16
    enhancement_real_values: int = 32 * 16 * 16
    groups: int = 4
    scale_bits: int = 16
    metadata_bits_per_use: float = 0.5
    feedback_bits_per_use: float = 0.5

    def count(self, selected: int):
        if not 0 <= selected <= self.groups:
            raise ValueError("invalid number of requested groups")
        forward = (self.base_real_values + selected * self.enhancement_real_values / self.groups) / 2
        side = math.ceil((1 + selected) * self.scale_bits / self.metadata_bits_per_use)
        # Fixed-length ACK/request flag + G-bit mask, even for ACK; final ACK is also charged.
        feedback_bits = 1 + self.groups + int(selected > 0)
        feedback = math.ceil(feedback_bits / self.feedback_bits_per_use)
        return dict(payload_uses=forward, scale_uses=side, feedback_uses=feedback,
                    total_uses=forward + side + feedback,
                    cbr_total=(forward + side + feedback) / self.image_values)


def packet_awgn(values, snr_db, noise=None):
    """Per-packet unit complex power; decoder receives *quantized* RMS only.

    Scalar is represented by IEEE binary16 (16 charged bits). Reliable metadata
    is an explicit idealization of a separate coded control link, not simulated LDPC.
    Receiver cannot access the unquantized transmitter RMS.
    """
    values = values.float()
    scale_tx = (2 * values.square().flatten(1).mean(1)).clamp_min(1e-12).sqrt()
    scale_rx = scale_tx.detach().half().float()
    if not torch.isfinite(scale_rx).all() or (scale_rx <= 0).any():
        raise FloatingPointError("packet scale is not representable in binary16")
    shape = (-1,) + (1,) * (values.ndim - 1)
    transmitted = values / scale_tx.reshape(shape)
    if noise is None:
        noise = torch.randn_like(transmitted)
    snr = torch.as_tensor(snr_db, device=values.device, dtype=torch.float32)
    sigma = (0.5 * torch.pow(10., -snr / 10)).sqrt()
    if sigma.ndim:
        sigma = sigma.reshape(shape)
    received = (transmitted + sigma * noise) * scale_rx.reshape(shape)
    return received, dict(scale_rx=scale_rx, normalized_power=2 * transmitted.square().flatten(1).mean(1))


def group_masks(height=16, width=16, device=None):
    """Four spatial quadrants, fixed and known at both terminals."""
    if height % 2 or width % 2:
        raise ValueError("even feature dimensions required")
    masks = torch.zeros(4, 1, height, width, device=device)
    for j in range(4):
        r, c = divmod(j, 2)
        masks[j, :, r*height//2:(r+1)*height//2, c*width//2:(c+1)*width//2] = 1
    return masks


def subset_mask(codes, height=16, width=16):
    bits = ((codes[:, None].long() >> torch.arange(4, device=codes.device)) & 1).float()
    return torch.einsum("bg,gchw->bchw", bits, group_masks(height, width, codes.device))


def enhancement_awgn(values, snr_db, noise=None):
    """Each potential group has a distinct noise draw; absent groups are never decoded."""
    result = torch.zeros_like(values)
    scales = []
    h, w = values.shape[-2:]
    for j in range(4):
        r, c = divmod(j, 2)
        sl = (..., slice(r*h//2, (r+1)*h//2), slice(c*w//2, (c+1)*w//2))
        y, info = packet_awgn(values[sl], snr_db, None if noise is None else noise[sl])
        result[sl] = y
        scales.append(info["scale_rx"])
    return result, torch.stack(scales, 1)
