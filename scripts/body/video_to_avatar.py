#!/usr/bin/env python3
"""
Real video → avatar video (BodyAvatar v4).

  python scripts/body/video_to_avatar.py \\
    --input /path/to/person.mp4 \\
    --output runs/my_avatar/avatar.mp4

First run on a new video: preprocess + finetune (~15–40 min GPU).
Re-render only: add --skip-train.

Uses v4 with synthetic pretrain init when available — beats in-repo baselines on NeuMan.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from osa_body.inference_engine import BodyAvatarEngine
from osa_body.paths import default_pretrained_checkpoint


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return s[:48] or "video"


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a real person video into a BodyAvatar MP4")
    p.add_argument("--input", required=True, help="Input MP4 (person visible)")
    p.add_argument("--output", required=True, help="Output avatar MP4 path")
    p.add_argument("--work-dir", default="", help="Cache dir (default: parent of --output)")
    p.add_argument("--epochs", type=int, default=22, help="Finetune epochs on your video")
    p.add_argument("--max-frames", type=int, default=480)
    p.add_argument("--init-checkpoint", default="", help="Override pretrained init (.pt)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-train", action="store_true", help="Reuse checkpoint in work-dir")
    p.add_argument("--side-by-side", action="store_true", help="Output original|avatar")
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
    engine = BodyAvatarEngine(device=device)
    print(f"Device: {device}")

    if not (subject_dir / "processed.npz").exists():
        print(f"[1/3] Extracting pose from {input_path.name} ...")
        meta = engine.preprocess_video(input_path, subject_dir, max_frames=args.max_frames)
        fps = float(meta["fps"])
        print(f"      {meta['num_frames']} frames @ {fps:.1f} fps")
    else:
        meta = json.loads((subject_dir / "meta.json").read_text())
        fps = float(meta["fps"])
        print(f"[1/3] Using cached preprocess ({meta['num_frames']} frames)")

    train, val = engine.build_split(processed_root, subject, split_file)
    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else default_pretrained_checkpoint()
    if init_ckpt:
        print(f"      Pretrained init: {init_ckpt}")

    if args.skip_train:
        if not ckpt_path.exists():
            raise SystemExit(f"No checkpoint at {ckpt_path} — remove --skip-train")
        print(f"[2/3] Skipping train (using {ckpt_path})")
        engine.load_checkpoint(ckpt_path)
        engine.setup_inference(subject, processed_root, split_file)
    else:
        print(f"[2/3] Training BodyAvatar v4 ({len(train)} train / {len(val)} val clips, {args.epochs} epochs) ...")
        metrics = engine.train(
            subject,
            processed_root,
            split_file,
            ckpt_path,
            epochs=args.epochs,
            init_checkpoint=init_ckpt,
        )
        print(f"      best val psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.3f} lpips={metrics['lpips']:.3f}")

    print("[3/3] Rendering avatar video ...")
    render_stats = engine.render_video(
        subject_dir / "processed.npz",
        out_mp4,
        fps=fps,
        side_by_side=args.side_by_side,
    )

    summary = {
        "input": str(input_path),
        "avatar_video": str(out_mp4),
        "model": "BodyAvatar v4",
        "subject": subject,
        "checkpoint": str(ckpt_path),
        "work_dir": str(work_dir),
        "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        **render_stats,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone — {out_mp4}")
    print(f"  {render_stats['frames']} frames @ {fps:.1f} fps  (render {render_stats['render_fps']:.1f} fps)")
    if render_stats["render_fps"] >= fps:
        print("  Inference is faster than video FPS — suitable for real-time playback.")


if __name__ == "__main__":
    main()
