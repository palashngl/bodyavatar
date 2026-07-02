from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from osa.baselines.warp_utils import piecewise_affine_warp_image, procrustes_warp_image
from osa.data.splits import build_clips, load_split, save_split, split_clips, train_starts_by_subject

# MediaPipe pose: emphasize torso / limbs over noisy extremities.
BODY_LMK_WEIGHTS = torch.tensor(
    [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8, 0.8,
    ],
    dtype=torch.float32,
)


class BodyTrainBank:
    """Per-subject train frame bank for full-body KNN retrieval."""

    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        warp: str = "piecewise",
    ) -> None:
        self.data_root = Path(data_root)
        self.warp = warp
        split_path = Path(split_file)
        if split_path.exists():
            train, _ = load_split(split_path)
        else:
            clips = build_clips(self.data_root)
            train, val = split_clips(clips)
            save_split(split_path, train, val)
        self._train_clips = train
        self._starts = train_starts_by_subject(train)
        self._train_frames = self._frame_indices_from_clips(train)
        self._frames: dict[str, torch.Tensor] = {}
        self._landmarks: dict[str, torch.Tensor] = {}
        self._orig_idx: dict[str, torch.Tensor] = {}
        self._warp_cache: dict[tuple, torch.Tensor] = {}
        self._cache_max = 4096
        self._gpu_subject: str | None = None

    @staticmethod
    def _frame_indices_from_clips(clips: list[dict]) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for c in clips:
            subj = c["subject"]
            out.setdefault(subj, set())
            for i in range(c["start"], c["start"] + c["clip_length"]):
                out[subj].add(i)
        return out

    def _load(self, subject: str) -> None:
        if subject in self._frames:
            return
        npz = np.load(self.data_root / subject / "processed.npz")
        frames = torch.from_numpy(npz["frames"]).permute(0, 3, 1, 2).float()
        landmarks = torch.from_numpy(npz["landmarks"]).float()
        indices = sorted(self._train_frames.get(subject, set(range(len(frames)))))
        if not indices:
            indices = list(range(len(frames)))
        indices = [i for i in indices if i < len(frames)]
        frames = frames[indices]
        landmarks = landmarks[indices]
        self._frames[subject] = frames
        self._landmarks[subject] = landmarks
        self._orig_idx[subject] = torch.tensor(indices, dtype=torch.long)

    def preload(self, subjects: list[str] | None = None) -> None:
        paths = sorted(self.data_root.glob("*/processed.npz"))
        if subjects:
            allow = set(subjects)
            paths = [p for p in paths if p.parent.name in allow]
        for p in paths:
            self._load(p.parent.name)

    def pin_to_gpu(self, subject: str, device: torch.device | None = None) -> None:
        """Keep train bank on GPU for faster inference."""
        self._load(subject)
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._frames[subject] = self._frames[subject].to(dev, non_blocking=True)
        self._landmarks[subject] = self._landmarks[subject].to(dev, non_blocking=True)
        self._orig_idx[subject] = self._orig_idx[subject].to(dev)
        self._gpu_subject = subject

    def _cache_key(
        self,
        subject: str,
        dst_lmk: torch.Tensor,
        exclude_indices: set[int] | None,
        k: int,
    ) -> tuple:
        lm = dst_lmk.detach().cpu().numpy().round(1).tobytes()
        ex = tuple(sorted(exclude_indices or ()))
        return (subject, lm, ex, k)

    def clear_cache(self) -> None:
        self._warp_cache.clear()

    def subjects(self) -> list[str]:
        return sorted({p.parent.name for p in self.data_root.glob("*/processed.npz")})

    def _warp(self, frame: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if self.warp == "procrustes":
            return procrustes_warp_image(frame, src, dst)
        return piecewise_affine_warp_image(frame, src, dst)

    def _landmark_distance(self, bank_lmks: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        w = BODY_LMK_WEIGHTS.to(bank_lmks.device)[: bank_lmks.shape[1]]
        diff = (bank_lmks[..., :2] - dst[..., :2].unsqueeze(0)) ** 2
        return (diff * w.view(1, -1, 1)).sum(dim=-1).mean(dim=-1)

    def _filter_bank(
        self,
        subject: str,
        device: torch.device,
        exclude_indices: set[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._load(subject)
        bank_lmks = self._landmarks[subject].to(device)
        bank_frames = self._frames[subject].to(device)
        orig = self._orig_idx[subject].to(device)
        if exclude_indices:
            keep = torch.tensor(
                [i for i, o in enumerate(orig.tolist()) if o not in exclude_indices],
                device=device,
                dtype=torch.long,
            )
            if len(keep) > 0:
                bank_lmks = bank_lmks[keep]
                bank_frames = bank_frames[keep]
        return bank_frames, bank_lmks

    @torch.no_grad()
    def knn_blend_warp(
        self,
        subject: str,
        dst_lmk: torch.Tensor,
        k: int = 3,
        exclude_indices: set[int] | None = None,
    ) -> torch.Tensor:
        if dst_lmk.dim() == 3:
            dst_lmk = dst_lmk[0]
        device = dst_lmk.device
        key = self._cache_key(subject, dst_lmk, exclude_indices, k)
        cached = self._warp_cache.get(key)
        if cached is not None:
            return cached.to(device)
        bank_frames, bank_lmks = self._filter_bank(subject, device, exclude_indices)
        dist = self._landmark_distance(bank_lmks, dst_lmk)
        k = min(k, len(dist))
        vals, idxs = torch.topk(dist, k=k, largest=False)
        warped = []
        # Landmark distances are O(10²–10³); tau must match that scale (v2 used 1/d).
        tau = max(float(vals.min().item()) * 0.35, 8.0)
        log_w = -(vals / tau)
        log_w = log_w - log_w.max()
        weights = torch.exp(log_w)
        weights = weights / weights.sum().clamp(min=1e-8)
        dst = dst_lmk.unsqueeze(0)
        for i, idx in enumerate(idxs):
            j = int(idx.item())
            warped.append(self._warp(bank_frames[j : j + 1], bank_lmks[j : j + 1], dst))
        out = sum(weights[i] * warped[i] for i in range(len(warped)))
        if len(self._warp_cache) >= self._cache_max:
            self._warp_cache.pop(next(iter(self._warp_cache)))
        self._warp_cache[key] = out.detach().cpu()
        return out

    @torch.no_grad()
    def batch_knn_blend_warps(
        self,
        subjects: list[str],
        landmarks: torch.Tensor,
        k: int = 3,
        exclude_per_item: list[set[int] | None] | None = None,
    ) -> torch.Tensor:
        b, tlen = landmarks.shape[:2]
        rows = []
        for bi in range(b):
            subject = subjects[bi]
            exclude = exclude_per_item[bi] if exclude_per_item else None
            frames_t = []
            for ti in range(tlen):
                frames_t.append(
                    self.knn_blend_warp(subject, landmarks[bi, ti], k=k, exclude_indices=exclude)
                )
            rows.append(torch.cat(frames_t, dim=0))
        return torch.stack(rows, dim=0)
