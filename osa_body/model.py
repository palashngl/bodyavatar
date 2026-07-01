from __future__ import annotations

import torch
import torch.nn as nn

from osa.models.encoder import TemporalVideoEncoder


class BodyResidualRefiner(nn.Module):
    """Residual correction on KNN warp base; uses input frames + confidence."""

    def __init__(self, image_size: int = 384, use_frame_hint: bool = True) -> None:
        super().__init__()
        self.image_size = image_size
        self.use_frame_hint = use_frame_hint
        in_ch = 7 if use_frame_hint else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(
        self,
        base: torch.Tensor,
        confidence: torch.Tensor,
        frame_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = base.shape[0]
        if confidence.dim() == 2:
            conf_map = confidence.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
            conf_map = conf_map.expand(-1, 1, self.image_size, self.image_size)
        else:
            conf_map = confidence
        if self.use_frame_hint and frame_hint is not None:
            inp = torch.cat([base, frame_hint, conf_map], dim=1)
        else:
            inp = torch.cat([base, conf_map], dim=1)
        gate = (1.0 - conf_map).clamp(0.05, 1.0)
        return self.net(inp) * self.residual_scale.clamp(0.02, 0.30) * gate


class BodyAvatarModel(nn.Module):
    """
    Full-body video-to-avatar model.

    Pipeline: KNN train retrieval + piecewise warp + confidence-guided residual.
    """

    def __init__(
        self,
        identity_dim: int = 256,
        motion_dim: int = 128,
        image_size: int = 384,
        knn_k: int = 5,
        use_frame_hint: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.knn_k = knn_k
        self.encoder = TemporalVideoEncoder(identity_dim, motion_dim)
        self.refiner = BodyResidualRefiner(image_size, use_frame_hint=use_frame_hint)

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
