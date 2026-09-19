from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ugp.channel import ChannelContext, ChannelObservation, TimeVaryingRayleigh
from ugp.model import small_swin_kwargs

from .modules import (
    QualityPredictor,
    ReliabilityAnchoredFlow,
    ReliabilityEstimator,
    balanced_reliability_bce,
    oracle_unreliable_mask,
    quality_boundary_bce,
)


def _load_upstream(upstream: str):
    path = str(Path(upstream).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from net.encoder import create_encoder
    from net.decoder import create_decoder

    return create_encoder, create_decoder


class FlowHARQJSCC(nn.Module):
    def __init__(
        self,
        upstream: str,
        latent_dim: int = 32,
        image_size: int = 256,
        mask_fraction: float = 0.35,
        flow_hidden_dim: int = 128,
        flow_depth: int = 2,
        flow_heads: int = 4,
    ):
        super().__init__()
        create_encoder, create_decoder = _load_upstream(upstream)
        encoder_kwargs, decoder_kwargs = small_swin_kwargs(latent_dim, image_size)
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.channel = TimeVaryingRayleigh()
        receiver_context_dim = 6
        self.reliability = ReliabilityEstimator(latent_dim, context_dim=receiver_context_dim)
        self.flow = ReliabilityAnchoredFlow(
            latent_dim,
            context_dim=receiver_context_dim,
            hidden_dim=flow_hidden_dim,
            depth=flow_depth,
            heads=flow_heads,
        )
        self.quality = QualityPredictor(latent_dim, context_dim=receiver_context_dim)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.mask_fraction = mask_fraction
        self.downsample = 4
        self.model_name = "SwinJSCC_w/o_SAandRA"
        self._resolution = image_size

    def _update_resolution(self, height: int, width: int) -> None:
        if height != width:
            raise ValueError("the current upstream decoder expects square inputs")
        if height != self._resolution:
            self.encoder.update_resolution(height, width)
            scale = 2**self.downsample
            self.decoder.update_resolution(height // scale, width // scale)
            self._resolution = height

    def encode(self, image: Tensor, context: ChannelContext) -> Tensor:
        self._update_resolution(image.shape[-2], image.shape[-1])
        nominal_snr = float(context.snr_db.detach().mean().item())
        return self.encoder(image, nominal_snr, self.latent_dim, self.model_name)

    def decode(self, tokens: Tensor, context: ChannelContext) -> Tensor:
        nominal_snr = float(context.snr_db.detach().mean().item())
        return self.decoder(tokens, nominal_snr, self.model_name).clamp(0.0, 1.0)

    def transmit(
        self,
        clean_tokens: Tensor,
        context: ChannelContext,
        previous: ChannelObservation | None = None,
        round_gap_ms: float = 1.0,
    ) -> ChannelObservation:
        power = torch.ones(clean_tokens.shape[:2], device=clean_tokens.device, dtype=clean_tokens.dtype)
        return self.channel.forward_with_state(
            clean_tokens,
            power,
            context,
            previous_gain=None if previous is None else previous.gain,
            round_gap_ms=round_gap_ms,
        )

    @staticmethod
    def combine(first: ChannelObservation, second: ChannelObservation) -> Tensor:
        """CSI-weighted maximum-ratio combining of equalized latent rounds."""
        first_weight = first.gain_sq.to(first.tokens.real.dtype)
        second_weight = second.gain_sq.to(second.tokens.real.dtype)
        denominator = (first_weight + second_weight).clamp_min(1e-8)
        return (first_weight * first.tokens + second_weight * second.tokens) / denominator

    def predict_unreliability(
        self, observation: ChannelObservation, context: ChannelContext
    ) -> tuple[Tensor, Tensor]:
        logits = self.reliability(observation.tokens, observation.receiver_features(context))
        return logits, logits.sigmoid()

    def repair(
        self,
        observation: ChannelObservation,
        context: ChannelContext,
        steps: int = 4,
        threshold: float = 0.8,
        soft_mask: bool = False,
        integration_time_mode: str = "midpoint",
    ) -> tuple[Tensor, Tensor, Tensor]:
        received = observation.tokens
        receiver_context = observation.receiver_features(context)
        logits, probabilities = self.predict_unreliability(observation, context)
        mask = probabilities if soft_mask else (probabilities >= threshold).to(probabilities.dtype)
        repaired = self.flow.integrate(
            received,
            mask,
            receiver_context,
            steps=steps,
            time_mode=integration_time_mode,
        )
        return repaired, probabilities, mask

    def load_backbone(self, checkpoint: str) -> dict:
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload)
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in own
            and own[key].shape == value.shape
            and (key.startswith("encoder.") or key.startswith("decoder."))
        }
        result = self.load_state_dict(compatible, strict=False)
        if not compatible:
            raise ValueError(f"no compatible encoder/decoder weights found in {checkpoint}")
        return {"loaded": len(compatible), "missing": result.missing_keys}

    def flow_losses(
        self,
        image: Tensor,
        clean: Tensor,
        observation: ChannelObservation,
        context: ChannelContext,
        flow_steps: int = 4,
        fm_loss_type: str = "mse",
        fm_time_mode: str = "uniform",
        decision_target_psnr: float = 24.0,
        decision_temperature_db: float = 1.0,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        received = observation.tokens
        receiver_context = observation.receiver_features(context)
        oracle_mask, token_error = oracle_unreliable_mask(
            clean, received, fraction=self.mask_fraction
        )
        reliability_logits, probabilities = self.predict_unreliability(observation, context)
        reliability_loss = balanced_reliability_bce(reliability_logits, oracle_mask)
        fm_loss = self.flow.matching_loss(
            clean,
            received,
            oracle_mask,
            receiver_context,
            loss_type=fm_loss_type,
            time_mode=fm_time_mode,
        )

        # Straight-through mask: hard behavior in the forward pass, useful
        # probability gradients in the backward pass.
        hard = (probabilities >= 0.5).to(probabilities.dtype)
        straight_through = hard + probabilities - probabilities.detach()
        repaired = self.flow.integrate(
            received,
            straight_through,
            receiver_context,
            steps=flow_steps,
            time_mode="zero" if fm_time_mode == "zero" else "midpoint",
        )
        reconstruction_pre = self.decode(received, context)
        reconstruction_post = self.decode(repaired, context)

        pre_log_mse = self.quality.target_log_mse(image, reconstruction_pre).detach()
        post_log_mse = self.quality.target_log_mse(image, reconstruction_post).detach()
        predicted_pre = self.quality(received, probabilities, receiver_context)
        predicted_post = self.quality(repaired, probabilities, receiver_context)
        quality_loss = 0.5 * (
            F.smooth_l1_loss(predicted_pre, pre_log_mse)
            + F.smooth_l1_loss(predicted_post, post_log_mse)
        )
        decision_loss = 0.5 * (
            quality_boundary_bce(
                predicted_pre,
                pre_log_mse,
                decision_target_psnr,
                decision_temperature_db,
            )
            + quality_boundary_bce(
                predicted_post,
                post_log_mse,
                decision_target_psnr,
                decision_temperature_db,
            )
        )
        reconstruction_loss = F.l1_loss(reconstruction_post, image)
        losses = {
            "fm": fm_loss,
            "reliability": reliability_loss,
            "quality": quality_loss,
            "decision": decision_loss,
            "reconstruction": reconstruction_loss,
        }
        artifacts = {
            "repaired": repaired,
            "reconstruction_pre": reconstruction_pre,
            "reconstruction_post": reconstruction_post,
            "probabilities": probabilities,
            "oracle_mask": oracle_mask,
            "token_error": token_error,
            "predicted_pre_psnr": self.quality.log_mse_to_psnr(predicted_pre),
            "predicted_post_psnr": self.quality.log_mse_to_psnr(predicted_post),
        }
        return losses, artifacts
