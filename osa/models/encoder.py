from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TemporalVideoEncoder(nn.Module):
    """Encodes a short clip into identity, motion, and confidence latents."""

    def __init__(
        self,
        identity_dim: int = 256,
        motion_dim: int = 128,
        num_parts: int = 4,
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        feat_dim = 512
        self.temporal = nn.GRU(feat_dim, 256, batch_first=True, bidirectional=True)
        self.identity_head = nn.Sequential(
            nn.Linear(512, identity_dim),
            nn.LayerNorm(identity_dim),
        )
        self.motion_head = nn.Linear(512, motion_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(512, num_parts),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            frames: [B, T, 3, H, W] in [0, 1]
        """
        b, t, c, h, w = frames.shape
        feats = self.stem(frames.reshape(b * t, c, h, w)).reshape(b, t, -1)
        temporal, _ = self.temporal(feats)
        pooled = temporal.mean(dim=1)
        z_id = self.identity_head(pooled)
        z_motion = self.motion_head(temporal)
        confidence = self.confidence_head(temporal)
        return {
            "z_id": z_id,
            "z_motion": z_motion,
            "confidence": confidence,
        }
