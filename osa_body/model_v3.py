"""BodyAvatar v3 — stronger refiner + multi-scale residual for real-video SOTA path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from osa.models.encoder import TemporalVideoEncoder


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class BodyResidualRefinerV3(nn.Module):
    """U-Net-lite refiner: base + frame hint + confidence → residual correction."""

    def __init__(self, image_size: int = 384) -> None:
        super().__init__()
        self.image_size = image_size
        self.enc1 = nn.Sequential(nn.Conv2d(7, 64, 3, padding=1), nn.GELU(), _ResBlock(64))
        self.down = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), _ResBlock(128), _ResBlock(128))
        self.up = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec = nn.Sequential(nn.GELU(), _ResBlock(64), nn.Conv2d(64, 3, 3, padding=1), nn.Tanh())
        self.residual_scale = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        base: torch.Tensor,
        confidence: torch.Tensor,
        frame_hint: torch.Tensor,
    ) -> torch.Tensor:
        b = base.shape[0]
        if confidence.dim() == 2:
            conf_map = confidence.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
            conf_map = conf_map.expand(-1, 1, self.image_size, self.image_size)
        else:
            conf_map = confidence
        inp = torch.cat([base, frame_hint, conf_map], dim=1)
        e1 = self.enc1(inp)
        m = self.mid(self.down(e1))
        out = self.dec(self.up(m) + e1)
        gate = (1.0 - conf_map).clamp(0.05, 1.0)
        return out * self.residual_scale.clamp(0.03, 0.35) * gate


class BodyAvatarModelV3(nn.Module):
    """KNN piecewise retrieval + temporal encoder + U-Net residual (v3)."""

    def __init__(
        self,
        identity_dim: int = 256,
        motion_dim: int = 128,
        image_size: int = 384,
        knn_k: int = 5,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.knn_k = knn_k
        self.encoder = TemporalVideoEncoder(identity_dim, motion_dim)
        self.refiner = BodyResidualRefinerV3(image_size)

    def forward(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor,
        retrieval_bank,
        subjects: list[str],
        exclude_frame_indices: list[set[int] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        retrieval = retrieval_bank.batch_knn_blend_warps(
            subjects,
            landmarks,
            k=self.knn_k,
            exclude_per_item=exclude_frame_indices,
        )
        enc = self.encoder(frames)
        conf = enc["confidence"].mean(dim=1)
        b, t, c, h, w = retrieval.shape
        base_flat = retrieval.reshape(b * t, c, h, w)
        frame_flat = frames.reshape(b * t, c, h, w)
        conf_flat = conf.unsqueeze(1).expand(-1, t, -1).reshape(b * t, -1)
        residual = self.refiner(base_flat, conf_flat, frame_flat)
        pred = (base_flat + residual).clamp(0.0, 1.0).view(b, t, c, h, w)
        return {
            "pred": pred,
            "retrieval": retrieval,
            "residual": residual.view(b, t, c, h, w),
            "confidence": enc["confidence"],
            "z_id": enc["z_id"],
            "z_motion": enc["z_motion"],
        }


def v3_composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn,
    confidence: torch.Tensor,
    w_l1: float = 1.0,
    w_lpips: float = 0.22,
    w_ssim: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stronger perceptual weighting for real-video quality."""
    from osa.utils.losses import composite_loss, ssim_loss

    base_loss, parts = composite_loss(
        pred, target, lpips_fn, confidence, w_l1=w_l1, w_lpips=w_lpips, w_ssim=w_ssim
    )
    # Multi-scale L1 (half-res) for global structure
    if pred.shape[-1] >= 192:
        pred_h = F.interpolate(pred, scale_factor=0.5, mode="bilinear", align_corners=False)
        tgt_h = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False)
        ms = F.l1_loss(pred_h, tgt_h) * 0.15
        base_loss = base_loss + ms
        parts["ms_l1"] = ms.item()
    return base_loss, parts
