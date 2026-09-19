from pathlib import Path
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .channel import ChannelContext, TimeVaryingRayleigh
from .heads import RiskAwareProtection, heteroscedastic_error_nll


def _load_upstream(upstream: str):
    path = str(Path(upstream).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from net.encoder import create_encoder
    from net.decoder import create_decoder

    return create_encoder, create_decoder


def small_swin_kwargs(latent_dim: int, image_size: int = 256):
    common = dict(
        model="SwinJSCC_w/o_SAandRA",
        img_size=(image_size, image_size),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    encoder = dict(
        **common,
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 2, 2],
        num_heads=[4, 6, 8, 10],
        C=latent_dim,
    )
    decoder = dict(
        **common,
        embed_dims=[320, 256, 192, 128],
        depths=[2, 2, 2, 2],
        num_heads=[10, 8, 6, 4],
        C=latent_dim,
    )
    return encoder, decoder


class UGPDeepJSCC(nn.Module):
    def __init__(
        self,
        upstream: str,
        latent_dim: int = 32,
        image_size: int = 256,
        average_power: float = 1.0,
        minimum_power: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()
        create_encoder, create_decoder = _load_upstream(upstream)
        encoder_kwargs, decoder_kwargs = small_swin_kwargs(latent_dim, image_size)
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.protection = RiskAwareProtection(
            latent_dim,
            average_power=average_power,
            minimum_power=minimum_power,
            temperature=temperature,
        )
        self.channel = TimeVaryingRayleigh()
        self.image_size = image_size
        self.downsample = 4
        self.latent_dim = latent_dim
        self.model_name = "SwinJSCC_w/o_SAandRA"
        self._resolution = image_size

    def _update_resolution(self, height: int, width: int):
        if height != width:
            raise ValueError("the current upstream decoder expects square inputs")
        if height != self._resolution:
            self.encoder.update_resolution(height, width)
            scale = 2**self.downsample
            self.decoder.update_resolution(height // scale, width // scale)
            self._resolution = height

    def forward(self, image: Tensor, context: ChannelContext, mode: str = "full"):
        _, _, height, width = image.shape
        self._update_resolution(height, width)
        # Upstream API accepts scalar SNR; SA is disabled, so this value is inert.
        nominal_snr = float(context.snr_db.detach().mean().item())
        tokens = self.encoder(image, nominal_snr, self.latent_dim, self.model_name)
        allocation = self.protection(tokens, context.features(), mode=mode)
        noisy_tokens = self.channel(tokens, allocation["power"], context)
        reconstruction = self.decoder(noisy_tokens, nominal_snr, self.model_name).clamp(0.0, 1.0)
        return reconstruction, tokens, allocation

    @staticmethod
    def local_error(image: Tensor, reconstruction: Tensor, token_count: int) -> Tensor:
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError("token grid must be square")
        pixel_error = (image - reconstruction).square().mean(dim=1, keepdim=True)
        return F.adaptive_avg_pool2d(pixel_error, (side, side)).flatten(1)

    def loss(
        self,
        image: Tensor,
        reconstruction: Tensor,
        allocation: dict,
        lambda_uncertainty: float = 0.05,
        lambda_multiscale: float = 0.1,
    ):
        l1 = F.l1_loss(reconstruction, image)
        # Dependency-free perceptual proxy for early runs; replace with LPIPS in final training.
        multiscale = sum(
            F.l1_loss(F.avg_pool2d(reconstruction, k), F.avg_pool2d(image, k))
            for k in (2, 4, 8)
        ) / 3.0
        error = self.local_error(image, reconstruction, allocation["uncertainty"].shape[1])
        uncertainty = heteroscedastic_error_nll(allocation["uncertainty"], error)
        total = l1 + lambda_multiscale * multiscale + lambda_uncertainty * uncertainty
        return {
            "total": total,
            "l1": l1,
            "multiscale": multiscale,
            "uncertainty_nll": uncertainty,
            "local_error": error,
        }

