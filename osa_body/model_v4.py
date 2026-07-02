"""BodyAvatar v4 — landmark heatmaps + deeper refiner + temporal-aware training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from osa.models.encoder import TemporalVideoEncoder


def landmarks_to_heatmap(landmarks: torch.Tensor, size: int, sigma: float = 4.0) -> torch.Tensor:
    """Render pose points to (B, 1, H, W) Gaussian heatmap (vectorized)."""
    b, n, _ = landmarks.shape
    device, dtype = landmarks.device, landmarks.dtype
    pts = landmarks[..., :2].clamp(0, size - 1)
    grid_y = torch.arange(size, device=device, dtype=dtype)
    grid_x = torch.arange(size, device=device, dtype=dtype)
    yy = grid_y.view(1, 1, size, 1).expand(b, n, -1, size)
    xx = grid_x.view(1, 1, 1, size).expand(b, n, size, -1)
    py = pts[:, :, 1].view(b, n, 1, 1)
    px = pts[:, :, 0].view(b, n, 1, 1)
    dist2 = (yy - py) ** 2 + (xx - px) ** 2
    heat = torch.exp(-dist2 / (2 * sigma**2)).sum(dim=1, keepdim=True)
    return heat.clamp(0, 1)


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


class BodyResidualRefinerV4(nn.Module):
    """U-Net refiner with pose heatmap + error map (frame - base)."""

    def __init__(self, image_size: int = 384) -> None:
        super().__init__()
        self.image_size = image_size
        # base(3) + frame(3) + err(3) + heatmap(1) + conf(1) = 11
        self.enc1 = nn.Sequential(nn.Conv2d(11, 96, 3, padding=1), nn.GELU(), _ResBlock(96), _ResBlock(96))
        self.down1 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), _ResBlock(128), _ResBlock(128), _ResBlock(128))
        self.up1 = nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1)
        self.dec = nn.Sequential(nn.GELU(), _ResBlock(96), nn.Conv2d(96, 3, 3, padding=1), nn.Tanh())
        self.residual_scale = nn.Parameter(torch.tensor(0.18))

    def forward(
        self,
        base: torch.Tensor,
        confidence: torch.Tensor,
        frame_hint: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        b = base.shape[0]
        if confidence.dim() == 2:
            conf_map = confidence.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
            conf_map = conf_map.expand(-1, 1, self.image_size, self.image_size)
        else:
            conf_map = confidence
        err = (frame_hint - base).detach()
        heat = landmarks_to_heatmap(landmarks, self.image_size)
        inp = torch.cat([base, frame_hint, err, heat, conf_map], dim=1)
        e1 = self.enc1(inp)
        m = self.mid(self.down1(e1))
        out = self.dec(self.up1(m) + e1)
        gate = (1.0 - conf_map).clamp(0.05, 1.0)
        return out * self.residual_scale.clamp(0.04, 0.40) * gate


class BodyAvatarModelV4(nn.Module):
    """KNN retrieval + temporal encoder + v4 pose-aware refiner."""

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
        self.refiner = BodyResidualRefinerV4(image_size)

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
        lmk_flat = landmarks.reshape(b * t, landmarks.shape[-2], landmarks.shape[-1])
        conf_flat = conf.unsqueeze(1).expand(-1, t, -1).reshape(b * t, -1)
        residual = self.refiner(base_flat, conf_flat, frame_flat, lmk_flat)
        pred = (base_flat + residual).clamp(0.0, 1.0).view(b, t, c, h, w)
        return {
            "pred": pred,
            "retrieval": retrieval,
            "residual": residual.view(b, t, c, h, w),
            "confidence": enc["confidence"],
            "z_id": enc["z_id"],
            "z_motion": enc["z_motion"],
        }


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def v4_composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn,
    confidence: torch.Tensor,
    mask: torch.Tensor | None = None,
    w_charb: float = 1.0,
    w_lpips: float = 0.28,
    w_ssim: float = 0.32,
) -> tuple[torch.Tensor, dict[str, float]]:
    from osa.utils.losses import composite_loss, ssim_loss

    if mask is not None:
        m = mask.unsqueeze(1)
        pred_m = pred * m
        tgt_m = target * m
    else:
        pred_m, tgt_m = pred, target

    charb = charbonnier(pred_m - tgt_m).mean()
    l1 = F.l1_loss(pred_m, tgt_m)
    lp = lpips_fn(pred_m * 2 - 1, tgt_m * 2 - 1).mean()
    ss = ssim_loss(pred_m, tgt_m)
    conf_reg = (1.0 - confidence.mean()).abs() * 0.02
    loss = w_charb * charb + 0.35 * l1 + w_lpips * lp + w_ssim * ss + conf_reg

    parts = {"charb": charb.item(), "l1": l1.item(), "lpips": lp.item(), "ssim_loss": ss.item(), "conf": conf_reg.item()}

    if pred.shape[-1] >= 192:
        pred_h = F.interpolate(pred_m, scale_factor=0.5, mode="bilinear", align_corners=False)
        tgt_h = F.interpolate(tgt_m, scale_factor=0.5, mode="bilinear", align_corners=False)
        ms = charbonnier(pred_h - tgt_h).mean() * 0.12
        loss = loss + ms
        parts["ms_charb"] = ms.item()

    return loss, parts


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match frame-to-frame motion magnitude between pred and target clips (B,T,C,H,W)."""
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    dp = (pred[:, 1:] - pred[:, :-1]).abs().mean(dim=(2, 3, 4))
    dt = (target[:, 1:] - target[:, :-1]).abs().mean(dim=(2, 3, 4))
    return F.l1_loss(dp, dt)
