#!/usr/bin/env python3
"""
Run BodyAvatar on a real MP4: preprocess → train (optional) → render output video.

Examples:
  python scripts/body/infer_real_video.py \\
    --video data/body/raw/neuman/bike.mp4 \\
    --subject neuman_bike \\
    --checkpoint runs/body_neuman_v2/neuman_bike/best.pt \\
    --skip-train \\
    --output runs/infer_bike

  python scripts/body/infer_real_video.py \\
    --video /path/to/your/video.mp4 \\
    --subject my_dance \\
    --train-epochs 15 \\
    --output runs/infer_my_dance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from osa.data.splits import build_clips, save_split, split_clips_temporal
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.preprocess import BodyPoseTracker
from osa_body.retrieval_bank import BodyTrainBank


def _train_subject(
    subject: str,
    processed_root: Path,
    split_file: Path,
    out_ckpt: Path,
    epochs: int,
    device: torch.device,
) -> Path:
    import lpips
    import torch.nn.functional as F

    from osa.utils.losses import composite_loss
    from osa.utils.metrics import compute_psnr_masked
    from osa.utils.seed import set_seed

    set_seed(42)
    bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
    bank.preload([subject])
    train_ds = BodyVideoDataset(processed_root, 8, "train", split_file)
    val_ds = BodyVideoDataset(processed_root, 8, "val", split_file)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = BodyAvatarModel(image_size=384, knn_k=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    def flat_mask(batch):
        masks = batch.get("masks")
        if masks is None:
            return None
        b, t, h, w = masks.shape
        return masks.to(device).reshape(b * t, h, w)

    best_score = -1e9
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
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
            loss, _ = composite_loss(pred_flat, target_flat, lpips_fn, out["confidence"])
            retr = out["retrieval"].reshape(b * t, c, h, w)
            if mask_flat is not None:
                retr = retr * m
            loss = loss + F.l1_loss(pred_flat, retr) * 0.12 + out["residual"].abs().mean() * 0.01
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
        if val_psnr >= best_score:
            best_score = val_psnr
            torch.save({"model": model.state_dict(), "subject": subject}, out_ckpt)
        print(f"  epoch {epoch+1}/{epochs} val_psnr={val_psnr:.2f}")
    return out_ckpt


@torch.no_grad()
def render_video(
    model: BodyAvatarModel,
    bank: BodyTrainBank,
    processed_root: Path,
    split_file: Path,
    device: torch.device,
    out_mp4: Path,
    split: str = "val",
) -> None:
    loader = DataLoader(
        BodyVideoDataset(processed_root, split=split, split_file=split_file),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    frames_out: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        frames = batch["frames"].to(device)
        out = model(
            frames=frames,
            landmarks=batch["landmarks"].to(device),
            retrieval_bank=bank,
            subjects=batch["subject"],
            exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
        )
        pred = out["pred"][0].cpu().numpy()
        gt = frames[0].cpu().numpy()
        for t in range(pred.shape[0]):
            p = (np.clip(pred[t].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            g = (np.clip(gt[t].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            panel = np.concatenate([g, p], axis=1)
            frames_out.append(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    if not frames_out:
        raise RuntimeError(f"No frames to render for split={split}")

    h, w = frames_out[0].shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    for f in frames_out:
        writer.write(f)
    writer.release()
    print(f"Saved side-by-side GT|pred video: {out_mp4} ({len(frames_out)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description="BodyAvatar real-video inference pipeline")
    p.add_argument("--video", required=True, help="Input MP4 path")
    p.add_argument("--subject", required=True, help="Subject id (e.g. neuman_bike or my_video)")
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--output", default="runs/infer_real")
    p.add_argument("--checkpoint", default="", help="Existing checkpoint; if empty, train first")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-epochs", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=360)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--render-split", default="val", choices=["train", "val"])
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    processed_root = Path(args.processed_dir)
    subject_dir = processed_root / args.subject
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (subject_dir / "processed.npz").exists():
        print(f"Preprocessing {args.video} → {subject_dir}")
        tracker = BodyPoseTracker(image_size=384)
        meta = tracker.process_video(Path(args.video), subject_dir, max_frames=args.max_frames)
        print(f"  {meta['num_frames']} frames @ {meta['fps']:.1f} fps")

    clips = build_clips(processed_root, 8)
    subj_clips = [c for c in clips if c["subject"] == args.subject]
    train, val = split_clips_temporal(subj_clips, train_ratio=0.85)
    split_file = out_dir / f"{args.subject}_split.json"
    save_split(split_file, train, val)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"
    if not args.skip_train and not args.checkpoint:
        print(f"Training BodyAvatar on {args.subject} ({len(train)} train clips)...")
        _train_subject(args.subject, processed_root, split_file, ckpt_path, args.train_epochs, device)
    elif not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BodyAvatarModel(image_size=384, knn_k=5).to(device)
    model.load_state_dict(ckpt["model"])
    bank = BodyTrainBank(processed_root, split_file, warp="piecewise")
    bank.preload([args.subject])

    out_mp4 = out_dir / f"{args.subject}_{args.render_split}.mp4"
    render_video(model, bank, processed_root, split_file, device, out_mp4, split=args.render_split)

    summary = {
        "video": str(args.video),
        "subject": args.subject,
        "checkpoint": str(ckpt_path),
        "output_video": str(out_mp4),
        "split_file": str(split_file),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
