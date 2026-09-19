"""Frozen published SwinJSCC base plus a separately learned enhancement codec."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

from ugp.model import small_swin_kwargs


class PublishedSwin(nn.Module):
    def __init__(self, upstream, checkpoint):
        super().__init__()
        sys.path.insert(0, str(Path(upstream).resolve()))
        from net.encoder import create_encoder
        from net.decoder import create_decoder
        checkpoint = Path(checkpoint)
        manifest = json.loads(checkpoint.with_suffix(".json").read_text())
        if manifest["channels"] != 32 or manifest["objective"] != "MSE":
            raise ValueError("Stage A requires the preregistered MSE C32 author checkpoint")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("author checkpoint SHA256 mismatch")
        enc, dec = small_swin_kwargs(32)
        enc["depths"], dec["depths"] = [2, 2, 6, 2], [2, 6, 2, 2]
        self.encoder, self.decoder = create_encoder(**enc), create_decoder(**dec)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name, module in (("encoder", self.encoder), ("decoder", self.decoder)):
            selected = {k[len(name)+1:]: v for k, v in state.items()
                        if k.startswith(name + ".") and "attn_mask" not in k}
            missing, unexpected = module.load_state_dict(selected, strict=False)
            if unexpected or any("attn_mask" not in k for k in missing):
                raise ValueError(f"author model mismatch: {missing}, {unexpected}")
        self.encoder.update_resolution(256, 256)
        self.decoder.update_resolution(16, 16)
        self.requires_grad_(False).eval()

    def encode(self, x):
        z = self.encoder(x, 10., 32, "SwinJSCC_w/o_SAandRA")
        return z.transpose(1, 2).reshape(x.shape[0], 32, 16, 16)

    def decode(self, z):
        return self.decoder(z.flatten(2).transpose(1, 2), 10., "SwinJSCC_w/o_SAandRA").clamp(0, 1)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + .1 * self.net(x)


class EnhancementCodec(nn.Module):
    """E1(x,E0(x)) is transmitter-only and never sees receiver observations.

    Four spatial packets carry novel learned features. F(y0,yS,mask,SNR)
    corrects the base latent, then uses the unchanged published decoder.
    Empty mask is EXACTLY the published base receiver, with paid scale metadata.
    """
    def __init__(self, width=96, bounded_evidence=False):
        super().__init__()
        self.bounded_evidence = bounded_evidence
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 48, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(48, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(64, width, 5, stride=2, padding=2), nn.SiLU())
        self.to_evidence = nn.Sequential(nn.Conv2d(width+32, width, 3, padding=1), nn.SiLU(),
                                         ResidualBlock(width), nn.Conv2d(width, 32, 3, padding=1))
        self.fusion = nn.Sequential(nn.Conv2d(32+32+2, width, 3, padding=1), nn.SiLU(),
                                    ResidualBlock(width), ResidualBlock(width),
                                    nn.Conv2d(width, 32, 3, padding=1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def encode(self, image, base_clean):
        raw = self.to_evidence(torch.cat((self.image_encoder(image), base_clean), 1))
        if self.bounded_evidence:
            # Fix the unconstrained scale degree of freedom before packetization.
            # This operation is transmitter-only; its statistics are never passed to RX.
            return torch.tanh(F.layer_norm(raw.float(), raw.shape[1:]))
        return raw

    def fuse(self, base_received, enhancement_received, mask, snr):
        snr = torch.as_tensor(snr, device=base_received.device).float()
        if snr.ndim == 0:
            snr = snr.expand(base_received.shape[0])
        snr_map = (snr[:, None, None, None]/20).expand(-1, 1, *base_received.shape[-2:])
        # Mask before any convolution: changing unrequested evidence cannot affect output.
        context = torch.cat((base_received, enhancement_received * mask, mask, snr_map), 1)
        correction = self.fusion(context)
        any_received = (mask.flatten(1).sum(1) > 0).float()[:, None, None, None]
        return base_received + correction * any_received
