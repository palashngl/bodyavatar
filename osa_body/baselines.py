from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from osa.baselines.warp_utils import piecewise_affine_warp_image, procrustes_warp_image
from osa.baselines.tps_warp import tps_warp_image
from osa_body.retrieval_bank import BodyTrainBank, BODY_LMK_WEIGHTS


class _AffineWarper:
    def __init__(self, image_size: int = 384) -> None:
        self.image_size = image_size

    def warp(self, atlas: torch.Tensor, src_lmk: torch.Tensor, dst_lmk: torch.Tensor) -> torch.Tensor:
        b = atlas.shape[0]
        src_ctr = src_lmk.mean(dim=1)
        dst_ctr = dst_lmk.mean(dim=1)
        src_scale = (src_lmk - src_ctr.unsqueeze(1)).norm(dim=-1).mean(dim=1).clamp(min=1e-3)
        dst_scale = (dst_lmk - dst_ctr.unsqueeze(1)).norm(dim=-1).mean(dim=1).clamp(min=1e-3)
        s = (dst_scale / src_scale).view(b)
        tx = (dst_ctr[:, 0] - src_ctr[:, 0] * s) / self.image_size
        ty = (dst_ctr[:, 1] - src_ctr[:, 1] * s) / self.image_size
        affine = torch.zeros(b, 2, 3, device=atlas.device, dtype=atlas.dtype)
        affine[:, 0, 0] = s
        affine[:, 1, 1] = s
        affine[:, 0, 2] = tx * 2
        affine[:, 1, 2] = ty * 2
        grid = F.affine_grid(affine, size=(b, 3, self.image_size, self.image_size), align_corners=True)
        return F.grid_sample(atlas, grid, align_corners=True)


class _StoreMixin:
    store: BodyTrainBank

    def _nearest(
        self,
        subject: str,
        dst_lmk: torch.Tensor,
        exclude_indices: set[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.store._load(subject)
        train_lmks = self.store._landmarks[subject].to(dst_lmk.device)
        train_frames = self.store._frames[subject]
        orig = self.store._orig_idx[subject].tolist()
        if exclude_indices:
            keep = [i for i, frame_idx in enumerate(orig) if frame_idx not in exclude_indices]
            if keep:
                train_lmks = train_lmks[keep]
                train_frames = train_frames[keep]
        dist = ((train_lmks - dst_lmk.unsqueeze(0)) ** 2)
        w = BODY_LMK_WEIGHTS.to(train_lmks.device)[: train_lmks.shape[1]]
        dist = (dist[..., :2] * w.view(1, -1, 1)).sum(dim=-1).mean(dim=-1)
        idx = int(dist.argmin().item())
        return train_frames[idx : idx + 1], train_lmks[idx : idx + 1]


class FirstFrameBodyWarp(_StoreMixin):
    name = "FirstFrameBodyWarp"

    def __init__(self, data_root: Path, image_size: int = 384) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")
        self.warper = _AffineWarper(image_size)
        self._first: dict[str, torch.Tensor] = {}
        self._first_lmk: dict[str, torch.Tensor] = {}

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        del exclude_indices
        if subject not in self._first:
            npz = np.load(self.store.data_root / subject / "processed.npz")
            self._first[subject] = torch.from_numpy(npz["frames"][0:1]).permute(0, 3, 1, 2).float()
            self._first_lmk[subject] = torch.from_numpy(npz["landmarks"][0:1]).float()
        t = landmarks.shape[0]
        atlas = self._first[subject].to(device).expand(t, -1, -1, -1)
        src = self._first_lmk[subject].to(device).expand(t, -1, -1)
        return self.warper.warp(atlas, src, landmarks.to(device))


class NearestTrainBodyWarp(_StoreMixin):
    name = "NearestTrainBodyWarp"

    def __init__(self, data_root: Path, image_size: int = 384) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")
        self.warper = _AffineWarper(image_size)

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            frame, src = self._nearest(subject, landmarks[t], exclude_indices)
            outs.append(self.warper.warp(frame.to(device), src.to(device), landmarks[t : t + 1].to(device)))
        return torch.cat(outs, dim=0)


class NearestTrainBodyProcrustes(_StoreMixin):
    name = "NearestTrainBodyProcrustes"

    def __init__(self, data_root: Path) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            frame, src = self._nearest(subject, landmarks[t], exclude_indices)
            outs.append(
                procrustes_warp_image(
                    frame.to(device), src.to(device), landmarks[t : t + 1].to(device)
                )
            )
        return torch.cat(outs, dim=0)


class _NearestKMixin(_StoreMixin):
    def _nearest_k(
        self,
        subject: str,
        dst_lmk: torch.Tensor,
        exclude_indices: set[int] | None,
        k: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        self.store._load(subject)
        train_lmks = self.store._landmarks[subject].to(dst_lmk.device)
        train_frames = self.store._frames[subject]
        orig = self.store._orig_idx[subject].tolist()
        if exclude_indices:
            keep = [i for i, frame_idx in enumerate(orig) if frame_idx not in exclude_indices]
            if keep:
                train_lmks = train_lmks[keep]
                train_frames = train_frames[keep]
        dist = ((train_lmks - dst_lmk.unsqueeze(0)) ** 2)
        w = BODY_LMK_WEIGHTS.to(train_lmks.device)[: train_lmks.shape[1]]
        dist = (dist[..., :2] * w.view(1, -1, 1)).sum(dim=-1).mean(dim=-1)
        k = min(k, len(dist))
        vals, idxs = torch.topk(dist, k=k, largest=False)
        out = []
        for d, idx in zip(vals, idxs):
            i = int(idx.item())
            out.append((train_frames[i : i + 1], train_lmks[i : i + 1], float(d.item())))
        return out


class NearestTrainBodyPiecewise(_StoreMixin):
    """Mesh-style Delaunay piecewise warp (geometry baseline, InstantAvatar-class proxy)."""

    name = "NearestTrainBodyPiecewise"

    def __init__(self, data_root: Path) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            frame, src = self._nearest(subject, landmarks[t], exclude_indices)
            outs.append(
                piecewise_affine_warp_image(
                    frame.to(device), src.to(device), landmarks[t : t + 1].to(device)
                )
            )
        return torch.cat(outs, dim=0)


class NearestTrainBodyTPS(_StoreMixin):
    """Thin-plate spline warp (classic non-rigid baseline)."""

    name = "NearestTrainBodyTPS"

    def __init__(self, data_root: Path) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            frame, src = self._nearest(subject, landmarks[t], exclude_indices)
            outs.append(tps_warp_image(frame.to(device), src.to(device), landmarks[t : t + 1].to(device)))
        return torch.cat(outs, dim=0)


class KNNBodyPiecewiseBlend(_NearestKMixin):
    """KNN blend with piecewise warps (strong retrieval + mesh geometry)."""

    name = "KNNBodyPiecewiseBlend"

    def __init__(self, data_root: Path, k: int = 3) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")
        self.k = k

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            dst = landmarks[t : t + 1].to(device)
            neighbors = self._nearest_k(subject, landmarks[t], exclude_indices, self.k)
            warped, weights = [], []
            for frame, src_lmk, dist in neighbors:
                warped.append(
                    piecewise_affine_warp_image(frame.to(device), src_lmk.to(device), dst)
                )
                weights.append(1.0 / (dist + 1e-4))
            w = torch.tensor(weights, device=device, dtype=warped[0].dtype)
            w = w / w.sum()
            outs.append(sum(w[i] * warped[i] for i in range(len(warped))))
        return torch.cat(outs, dim=0)


class NeutralAtlasBodyPiecewise(_StoreMixin):
    """Mean train texture + piecewise warp (NeuralBody/INSTA-lite proxy)."""

    name = "NeutralAtlasBodyPiecewise"

    def __init__(self, data_root: Path) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")
        self._atlas: dict[str, torch.Tensor] = {}

    def _load_atlas(self, subject: str) -> None:
        if subject in self._atlas:
            return
        self.store._load(subject)
        self._atlas[subject] = self.store._frames[subject].mean(dim=0, keepdim=True)

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        del exclude_indices
        self._load_atlas(subject)
        atlas = self._atlas[subject].to(device)
        neutral = torch.from_numpy(
            np.load(self.store.data_root / subject / "processed.npz")["neutral_landmarks"]
        ).float().to(device).unsqueeze(0)
        outs = []
        for t in range(landmarks.shape[0]):
            outs.append(
                piecewise_affine_warp_image(atlas, neutral, landmarks[t : t + 1].to(device))
            )
        return torch.cat(outs, dim=0)


class KNNBodyBlendWarp:
    name = "KNNBodyBlendWarp"

    def __init__(self, data_root: Path, k: int = 3) -> None:
        self.store = BodyTrainBank(data_root, data_root.parent / "splits" / "bench_split.json")
        self.k = k

    def predict(self, subject: str, landmarks: torch.Tensor, device: torch.device, exclude_indices=None):
        outs = []
        for t in range(landmarks.shape[0]):
            outs.append(
                self.store.knn_blend_warp(subject, landmarks[t].to(device), k=self.k, exclude_indices=exclude_indices)
            )
        return torch.cat(outs, dim=0)
