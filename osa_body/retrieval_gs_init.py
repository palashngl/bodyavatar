"""Retrieval initialization for 3D Gaussian avatar optimization (CVPR SOTA path)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from osa_body.retrieval_bank import BodyTrainBank


@dataclass
class RetrievalInitConfig:
    knn_k: int = 5
    image_size: int = 384
    warp: str = "piecewise"


class RetrievalGSInitializer:
    """
    Stage-1: KNN retrieval warps provide dense 2D appearance targets.
    Stage-2: 3DGS optimizer fits Gaussians to match retrieval + photometric loss.

    This class packages per-frame retrieval targets for an external 3DGS trainer
    (GauHuman / gsplat). Full 3DGS training requires smplx + optional gsplat install.
    """

    def __init__(
        self,
        processed_root: Path,
        split_file: Path,
        cfg: RetrievalInitConfig | None = None,
    ) -> None:
        self.cfg = cfg or RetrievalInitConfig()
        self.bank = BodyTrainBank(processed_root, split_file, warp=self.cfg.warp)
        self.processed_root = Path(processed_root)

    def preload(self, subjects: list[str]) -> None:
        self.bank.preload(subjects)

    @torch.no_grad()
    def build_retrieval_targets(
        self,
        subject: str,
        landmarks: torch.Tensor,
        exclude_indices: set[int] | None = None,
    ) -> torch.Tensor:
        """Return (T, 3, H, W) retrieval warps as 3DGS photometric initialization."""
        outs = []
        for t in range(landmarks.shape[0]):
            outs.append(
                self.bank.knn_blend_warp(
                    subject,
                    landmarks[t],
                    k=self.cfg.knn_k,
                    exclude_indices=exclude_indices,
                )
            )
        return torch.cat(outs, dim=0)

    def export_init_package(
        self,
        subject: str,
        out_dir: Path,
        frame_indices: list[int],
        landmarks: torch.Tensor,
    ) -> Path:
        """
        Save retrieval targets + GT frames for 3DGS fine-tuning.
        Output: out_dir/{retrieval,gt,meta}.npz
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.processed_root / subject / "processed.npz"
        data = np.load(npz_path)
        exclude = set(frame_indices)
        retr = self.build_retrieval_targets(subject, landmarks, exclude).cpu().numpy()
        gt = data["frames"][frame_indices]
        pack = {
            "retrieval": retr.transpose(0, 2, 3, 1).astype(np.float32),
            "gt": gt.astype(np.float32),
            "frame_indices": np.array(frame_indices, dtype=np.int32),
            "subject": subject,
        }
        if "smpl_joints" in data:
            pack["smpl_joints"] = data["smpl_joints"][frame_indices]
        out_path = out_dir / f"{subject}_gs_init.npz"
        np.savez_compressed(out_path, **pack)
        return out_path
