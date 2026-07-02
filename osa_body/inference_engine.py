"""Train and fast-render BodyAvatar v4 on real user videos."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from osa.data.splits import build_clips, save_split, split_clips_temporal
from osa.utils.metrics import compute_lpips_masked, compute_psnr_masked, compute_ssim_masked
from osa.utils.seed import set_seed
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model_v4 import BodyAvatarModelV4, temporal_consistency_loss, v4_composite_loss
from osa_body.paths import default_pretrained_checkpoint
from osa_body.preprocess import BodyPoseTracker
from osa_body.retrieval_bank import BodyTrainBank


def combined_score(psnr: float, ssim: float, lpips: float) -> float:
    return psnr + 5.0 * ssim - 15.0 * lpips


def load_compatible(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = ckpt["model"]
    dst = model.state_dict()
    compatible = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def _flat_mask(batch: dict, device: torch.device):
    masks = batch.get("masks")
    if masks is None:
        return None
    b, t, h, w = masks.shape
    return masks.to(device).reshape(b * t, h, w)


class BodyAvatarEngine:
    """
    End-to-end: real video → preprocess → v4 finetune → avatar MP4 at source FPS.

    After training, inference uses a GPU-pinned retrieval bank and clip batching
    to reach ~10–30 FPS on a 3090 for 384×384 output.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        clip_length: int = 8,
        knn_k: int = 5,
        image_size: int = 384,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_length = clip_length
        self.knn_k = knn_k
        self.image_size = image_size
        self.model = BodyAvatarModelV4(image_size=image_size, knn_k=knn_k).to(self.device)
        self.bank: BodyTrainBank | None = None
        self.subject: str | None = None
        self._frame_buf: deque[torch.Tensor] = deque(maxlen=clip_length)
        self._lmk_buf: deque[torch.Tensor] = deque(maxlen=clip_length)

    def preprocess_video(
        self,
        video_path: Path,
        output_dir: Path,
        max_frames: int = 480,
    ) -> dict:
        tracker = BodyPoseTracker(image_size=self.image_size)
        return tracker.process_video(video_path, output_dir, max_frames=max_frames)

    def build_split(self, processed_root: Path, subject: str, split_file: Path) -> tuple[list, list]:
        clips = build_clips(processed_root, self.clip_length)
        subj_clips = [c for c in clips if c["subject"] == subject]
        train, val = split_clips_temporal(subj_clips, train_ratio=0.85)
        save_split(split_file, train, val)
        return train, val

    def train(
        self,
        subject: str,
        processed_root: Path,
        split_file: Path,
        ckpt_path: Path,
        epochs: int = 22,
        init_checkpoint: Path | None = None,
        lr: float = 1.5e-4,
        temporal_weight: float = 0.08,
        retrieval_anchor: float = 0.05,
        residual_reg: float = 0.006,
    ) -> dict[str, float]:
        import lpips

        set_seed(42)
        init_ckpt = init_checkpoint or default_pretrained_checkpoint()
        if init_ckpt and init_ckpt.exists():
            n = load_compatible(self.model, init_ckpt, self.device)
            print(f"  init from {init_ckpt.name} ({n} tensors)")

        bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
        bank.preload([subject])
        bank.pin_to_gpu(subject, self.device)

        train_loader = DataLoader(
            BodyVideoDataset(processed_root, self.clip_length, "train", split_file),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            BodyVideoDataset(processed_root, self.clip_length, "val", split_file),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lpips_fn = lpips.LPIPS(net="alex").to(self.device).eval()

        best_metrics = {"combined": -1e9}
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                frames = batch["frames"].to(self.device)
                landmarks = batch["landmarks"].to(self.device)
                mask_flat = _flat_mask(batch, self.device)
                out = self.model(
                    frames=frames,
                    landmarks=landmarks,
                    retrieval_bank=bank,
                    subjects=batch["subject"],
                    exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
                )
                pred = out["pred"]
                b, t, c, h, w = pred.shape
                pred_flat = pred.reshape(b * t, c, h, w)
                target_flat = frames.reshape(b * t, c, h, w)
                if mask_flat is not None:
                    m = mask_flat.unsqueeze(1)
                    pred_flat = pred_flat * m
                    target_flat = target_flat * m
                loss, _ = v4_composite_loss(
                    pred_flat, target_flat, lpips_fn, out["confidence"], mask=mask_flat
                )
                loss = loss + temporal_consistency_loss(pred, frames) * temporal_weight
                retr_flat = out["retrieval"].reshape(b * t, c, h, w)
                if mask_flat is not None:
                    retr_flat = retr_flat * m
                loss = loss + F.l1_loss(pred_flat, retr_flat) * retrieval_anchor
                loss = loss + out["residual"].abs().mean() * residual_reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            sched.step()

            metrics = self._evaluate(val_loader, bank, lpips_fn)
            metrics["combined"] = combined_score(metrics["psnr"], metrics["ssim"], metrics["lpips"])
            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.3f} lpips={metrics['lpips']:.3f}"
            )
            if metrics["combined"] >= best_metrics.get("combined", -1e9):
                best_metrics = metrics
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "metrics": metrics,
                        "subject": subject,
                        "model_type": "body_v4",
                    },
                    ckpt_path,
                )

        self.bank = bank
        self.subject = subject
        return best_metrics

    @torch.no_grad()
    def _evaluate(self, loader, bank, lpips_fn) -> dict[str, float]:
        self.model.eval()
        psnrs, ssims, lpipss = [], [], []
        for batch in loader:
            frames = batch["frames"].to(self.device)
            mask_flat = _flat_mask(batch, self.device)
            out = self.model(
                frames=frames,
                landmarks=batch["landmarks"].to(self.device),
                retrieval_bank=bank,
                subjects=batch["subject"],
                exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
            )
            pred = out["pred"]
            b, t, c, h, w = pred.shape
            pred_flat = pred.reshape(b * t, c, h, w)
            target_flat = frames.reshape(b * t, c, h, w)
            psnrs.append(compute_psnr_masked(pred_flat, target_flat, mask_flat))
            ssims.append(compute_ssim_masked(pred_flat, target_flat, mask_flat))
            lpipss.append(compute_lpips_masked(pred_flat, target_flat, lpips_fn, mask_flat))
        n = max(len(psnrs), 1)
        return {"psnr": sum(psnrs) / n, "ssim": sum(ssims) / n, "lpips": sum(lpipss) / n}

    def load_checkpoint(self, ckpt_path: Path) -> None:
        load_compatible(self.model, ckpt_path, self.device)
        self.model.eval()

    def setup_inference(
        self,
        subject: str,
        processed_root: Path,
        split_file: Path,
    ) -> None:
        self.subject = subject
        self.bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
        self.bank.preload([subject])
        self.bank.pin_to_gpu(subject, self.device)
        self.bank.clear_cache()
        self.model.eval()

    @torch.no_grad()
    def render_video(
        self,
        npz_path: Path,
        out_mp4: Path,
        fps: float,
        side_by_side: bool = False,
        avatar_only: bool = True,
    ) -> dict:
        if self.bank is None or self.subject is None:
            raise RuntimeError("Call setup_inference() or train() first")

        data = np.load(npz_path)
        n = int(len(data["frames"]))
        all_frames = torch.from_numpy(data["frames"]).permute(0, 3, 1, 2).float()
        all_lmks = torch.from_numpy(data["landmarks"]).float()

        rendered: list[np.ndarray] = []
        t0 = time.perf_counter()

        for start in range(0, n, self.clip_length):
            end = min(start + self.clip_length, n)
            clip_f = all_frames[start:end]
            clip_l = all_lmks[start:end]
            actual = end - start
            if actual < self.clip_length:
                pad = self.clip_length - actual
                clip_f = torch.cat([clip_f, clip_f[-1:].repeat(pad, 1, 1, 1)], dim=0)
                clip_l = torch.cat([clip_l, clip_l[-1:].repeat(pad, 1, 1)], dim=0)

            frames_b = clip_f.unsqueeze(0).to(self.device, non_blocking=True)
            lmks_b = clip_l.unsqueeze(0).to(self.device, non_blocking=True)
            exclude = set(range(start, end))
            out = self.model(
                frames=frames_b,
                landmarks=lmks_b,
                retrieval_bank=self.bank,
                subjects=[self.subject],
                exclude_frame_indices=[exclude],
            )
            pred = out["pred"][0, :actual].cpu()
            gt = all_frames[start:end]
            for t in range(actual):
                p = (np.clip(pred[t].numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                if side_by_side:
                    g = (np.clip(gt[t].numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    p = np.concatenate([g, p], axis=1)
                rendered.append(cv2.cvtColor(p, cv2.COLOR_RGB2BGR))

        elapsed = time.perf_counter() - t0
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        h, w = rendered[0].shape[:2]
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in rendered:
            writer.write(frame)
        writer.release()

        return {
            "frames": n,
            "fps": fps,
            "render_fps": n / max(elapsed, 1e-6),
            "output": str(out_mp4),
            "avatar_only": avatar_only and not side_by_side,
        }

    def reset_stream(self) -> None:
        self._frame_buf.clear()
        self._lmk_buf.clear()

    @torch.no_grad()
    def render_stream_frame(
        self,
        frame_chw: torch.Tensor,
        landmarks: torch.Tensor,
        frame_index: int,
    ) -> np.ndarray:
        """Single-frame avatar for webcam / live pipelines (rolling clip buffer)."""
        if self.bank is None or self.subject is None:
            raise RuntimeError("Call setup_inference() first")

        self._frame_buf.append(frame_chw)
        self._lmk_buf.append(landmarks)
        while len(self._frame_buf) < self.clip_length:
            self._frame_buf.appendleft(frame_chw)
            self._lmk_buf.appendleft(landmarks)

        clip_f = torch.stack(list(self._frame_buf), dim=0).unsqueeze(0).to(self.device)
        clip_l = torch.stack(list(self._lmk_buf), dim=0).unsqueeze(0).to(self.device)
        out = self.model(
            frames=clip_f,
            landmarks=clip_l,
            retrieval_bank=self.bank,
            subjects=[self.subject],
            exclude_frame_indices=[{frame_index}],
        )
        pred = out["pred"][0, -1].cpu().numpy().transpose(1, 2, 0)
        return (np.clip(pred, 0, 1) * 255).astype(np.uint8)
