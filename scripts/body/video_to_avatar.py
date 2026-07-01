#!/usr/bin/env python3
"""
Video → avatar video (one command).

Give an MP4 of a person moving; get back a full-body avatar MP4.

  python scripts/body/video_to_avatar.py \\
    --input /path/to/your_video.mp4 \\
    --output runs/my_avatar/avatar.mp4

Pipeline: MediaPipe pose extract → train BodyAvatar on your clip → render all frames.
First run on a new video takes ~30–90 min (GPU). Re-render with --skip-train.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from osa.data.splits import build_clips, save_split, split_clips_temporal
from osa_body.model_v3 import BodyAvatarModelV3, v3_composite_loss
from osa_body.preprocess import BodyPoseTracker
from osa_body.retrieval_bank import BodyTrainBank


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return s[:48] or "video"


def _train(
    subject: str,
    processed_root: Path,
    split_file: Path,
    ckpt_path: Path,
    epochs: int,
    device: torch.device,
) -> None:
    import lpips
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from osa.utils.metrics import compute_psnr_masked
    from osa.utils.seed import set_seed
    from osa_body.dataset import BodyVideoDataset, collate_fn

    set_seed(42)
    bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
    bank.preload([subject])
    train_loader = DataLoader(
        BodyVideoDataset(processed_root, 8, "train", split_file),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        BodyVideoDataset(processed_root, 8, "val", split_file),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = BodyAvatarModelV3(image_size=384, knn_k=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    def flat_mask(batch):
        masks = batch.get("masks")
        if masks is None:
            return None
        b, t, h, w = masks.shape
        return masks.to(device).reshape(b * t, h, w)

    best = -1e9
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            frames = batch["frames"].to(device)
            mask_flat = flat_mask(batch)
            out = model(
                frames=frames,
                landmarks=batch["landmarks"].to(device),
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
            loss, _ = v3_composite_loss(pred_flat, target_flat, lpips_fn, out["confidence"])
            retr = out["retrieval"].reshape(b * t, c, h, w)
            if mask_flat is not None:
                retr = retr * m
            loss = loss + F.l1_loss(pred_flat, retr) * 0.06 + out["residual"].abs().mean() * 0.008
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        psnrs = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                mask_flat = flat_mask(batch)
                out = model(
                    frames=frames,
                    landmarks=batch["landmarks"].to(device),
                    retrieval_bank=bank,
                    subjects=batch["subject"],
                    exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
                )
                pred = out["pred"].reshape(-1, 3, 384, 384)
                target = frames.reshape(-1, 3, 384, 384)
                psnrs.append(compute_psnr_masked(pred, target, mask_flat))
        val_psnr = sum(psnrs) / max(len(psnrs), 1)
        if val_psnr >= best:
            best = val_psnr
            torch.save({"model": model.state_dict(), "subject": subject, "model_type": "body_v3"}, ckpt_path)
        print(f"  train epoch {epoch + 1}/{epochs}  val_psnr={val_psnr:.2f}")


@torch.no_grad()
def render_avatar_video(
    model: BodyAvatarModelV3,
    bank: BodyTrainBank,
    subject: str,
    npz_path: Path,
    out_mp4: Path,
    device: torch.device,
    fps: float,
    clip_length: int = 8,
    side_by_side: bool = False,
) -> int:
    data = np.load(npz_path)
    n = int(len(data["frames"]))
    all_frames = torch.from_numpy(data["frames"]).permute(0, 3, 1, 2).float()
    all_lmks = torch.from_numpy(data["landmarks"]).float()
    model.eval()

    rendered: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, clip_length):
            end = min(start + clip_length, n)
            clip_f = all_frames[start:end]
            clip_l = all_lmks[start:end]
            actual = end - start
            if actual < clip_length:
                pad = clip_length - actual
                clip_f = torch.cat([clip_f, clip_f[-1:].repeat(pad, 1, 1, 1)], dim=0)
                clip_l = torch.cat([clip_l, clip_l[-1:].repeat(pad, 1, 1)], dim=0)
            frames_b = clip_f.unsqueeze(0).to(device)
            lmks_b = clip_l.unsqueeze(0).to(device)
            exclude = set(range(start, end))
            out = model(
                frames=frames_b,
                landmarks=lmks_b,
                retrieval_bank=bank,
                subjects=[subject],
                exclude_frame_indices=[exclude],
            )
            pred = out["pred"][0].cpu()
            gt = all_frames[start:end]
            for t in range(actual):
                p = (np.clip(pred[t].numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                if side_by_side:
                    g = (np.clip(gt[t].numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    p = np.concatenate([g, p], axis=1)
                rendered.append(cv2.cvtColor(p, cv2.COLOR_RGB2BGR))

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w = rendered[0].shape[:2]
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in rendered:
        writer.write(frame)
    writer.release()
    return len(rendered)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a person video into a BodyAvatar MP4")
    p.add_argument("--input", required=True, help="Input MP4 (person visible, full or upper body)")
    p.add_argument("--output", required=True, help="Output avatar MP4 path")
    p.add_argument("--work-dir", default="", help="Cache dir (default: same folder as --output)")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs on your video")
    p.add_argument("--max-frames", type=int, default=360)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-train", action="store_true", help="Reuse checkpoint in work-dir")
    p.add_argument(
        "--side-by-side",
        action="store_true",
        help="Output original|avatar side-by-side instead of avatar only",
    )
    args = p.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_mp4 = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else out_mp4.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    subject = f"user_{_slug(input_path.stem)}"
    processed_root = work_dir / "processed"
    subject_dir = processed_root / subject
    split_file = work_dir / "split.json"
    ckpt_path = work_dir / "best.pt"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not (subject_dir / "processed.npz").exists():
        print(f"[1/3] Extracting pose + cropping from {input_path.name} ...")
        tracker = BodyPoseTracker(image_size=384)
        meta = tracker.process_video(input_path, subject_dir, max_frames=args.max_frames)
        fps = float(meta["fps"])
        print(f"      {meta['num_frames']} frames @ {fps:.1f} fps")
    else:
        meta = json.loads((subject_dir / "meta.json").read_text())
        fps = float(meta["fps"])
        print(f"[1/3] Using cached preprocess ({meta['num_frames']} frames)")

    clips = build_clips(processed_root, 8)
    subj_clips = [c for c in clips if c["subject"] == subject]
    train, val = split_clips_temporal(subj_clips, train_ratio=0.85)
    save_split(split_file, train, val)

    if args.skip_train:
        if not ckpt_path.exists():
            raise SystemExit(f"No checkpoint at {ckpt_path} — remove --skip-train to train first")
        print(f"[2/3] Skipping train (using {ckpt_path})")
    else:
        print(f"[2/3] Training BodyAvatar on your video ({len(train)} train clips, ~{args.epochs} epochs) ...")
        _train(subject, processed_root, split_file, ckpt_path, args.epochs, device)

    print("[3/3] Rendering avatar video ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BodyAvatarModelV3(image_size=384, knn_k=5).to(device)
    model.load_state_dict(ckpt["model"])
    bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
    bank.preload([subject])

    n_frames = render_avatar_video(
        model,
        bank,
        subject,
        subject_dir / "processed.npz",
        out_mp4,
        device,
        fps=fps,
        side_by_side=args.side_by_side,
    )

    summary = {
        "input": str(input_path),
        "avatar_video": str(out_mp4),
        "frames": n_frames,
        "fps": fps,
        "subject": subject,
        "checkpoint": str(ckpt_path),
        "work_dir": str(work_dir),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone — avatar video: {out_mp4}  ({n_frames} frames @ {fps:.1f} fps)")
    if not args.side_by_side:
        print("Tip: add --side-by-side to compare original vs avatar in one file.")


if __name__ == "__main__":
    main()
