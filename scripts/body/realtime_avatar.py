#!/usr/bin/env python3
"""
Live or file-stream avatar rendering (BodyAvatar v4).

Train once on a reference video, then drive the avatar from:
  - a second MP4 (--drive-video), or
  - webcam (--webcam 0)

Example:
  # Train on reference clip, render driven by same or new video
  python scripts/body/realtime_avatar.py \\
    --reference /path/to/reference.mp4 \\
    --drive-video /path/to/motion.mp4 \\
    --output runs/live/avatar.mp4

  # Webcam after training on reference
  python scripts/body/realtime_avatar.py \\
    --reference /path/to/reference.mp4 \\
    --webcam 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from osa_body.inference_engine import BodyAvatarEngine
from osa_body.paths import default_pretrained_checkpoint
from osa_body.preprocess import LiveBodyPoseTracker


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return s[:48] or "video"


def _train_reference(
    engine: BodyAvatarEngine,
    reference: Path,
    work_dir: Path,
    epochs: int,
    max_frames: int,
    init_ckpt: Path | None,
) -> tuple[str, Path, float]:
    subject = f"ref_{_slug(reference.stem)}"
    processed_root = work_dir / "processed"
    subject_dir = processed_root / subject
    split_file = work_dir / "split.json"
    ckpt_path = work_dir / "best.pt"

    if not (subject_dir / "processed.npz").exists():
        print(f"Preprocessing reference: {reference.name}")
        engine.preprocess_video(reference, subject_dir, max_frames=max_frames)
    else:
        print(f"Using cached reference preprocess")

    engine.build_split(processed_root, subject, split_file)

    if ckpt_path.exists():
        print(f"Loading existing checkpoint {ckpt_path}")
        engine.load_checkpoint(ckpt_path)
    else:
        print(f"Training v4 on reference ({epochs} epochs) ...")
        engine.train(subject, processed_root, split_file, ckpt_path, epochs=epochs, init_checkpoint=init_ckpt)

    meta = json.loads((subject_dir / "meta.json").read_text())
    fps = float(meta["fps"])
    engine.setup_inference(subject, processed_root, split_file)
    return subject, subject_dir / "processed.npz", fps


def _render_drive_video(
    engine: BodyAvatarEngine,
    tracker: LiveBodyPoseTracker,
    drive_path: Path,
    out_mp4: Path,
    fps: float,
    max_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(drive_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {drive_path}")

    engine.reset_stream()
    frames_out: list[np.ndarray] = []
    idx = 0
    while cap.isOpened() and idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        parsed = tracker.process_frame(frame)
        if parsed is None:
            continue
        frame_f, lmk, _ = parsed
        frame_t = torch.from_numpy(frame_f).permute(2, 0, 1).float()
        lmk_t = torch.from_numpy(lmk).float()
        avatar_rgb = engine.render_stream_frame(frame_t, lmk_t, idx)
        frames_out.append(cv2.cvtColor(avatar_rgb, cv2.COLOR_RGB2BGR))
        idx += 1
    cap.release()

    if not frames_out:
        raise RuntimeError("No frames rendered from drive video")

    h, w = frames_out[0].shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_out:
        writer.write(f)
    writer.release()
    print(f"Saved avatar video: {out_mp4} ({len(frames_out)} frames)")


def _run_webcam(engine: BodyAvatarEngine, tracker: LiveBodyPoseTracker, cam_id: int) -> None:
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam {cam_id}")

    engine.reset_stream()
    idx = 0
    print("Webcam avatar — press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        parsed = tracker.process_frame(frame)
        if parsed is not None:
            frame_f, lmk, _ = parsed
            frame_t = torch.from_numpy(frame_f).permute(2, 0, 1).float()
            lmk_t = torch.from_numpy(lmk).float()
            avatar_rgb = engine.render_stream_frame(frame_t, lmk_t, idx)
            avatar_bgr = cv2.cvtColor(avatar_rgb, cv2.COLOR_RGB2BGR)
            panel = np.concatenate([frame, cv2.resize(avatar_bgr, (384, 384))], axis=1)
            cv2.imshow("BodyAvatar live (camera | avatar)", panel)
            idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Stream BodyAvatar v4 from video or webcam")
    p.add_argument("--reference", required=True, help="Reference MP4 to train avatar identity")
    p.add_argument("--work-dir", default="runs/realtime_avatar")
    p.add_argument("--drive-video", default="", help="Motion source MP4 (optional)")
    p.add_argument("--output", default="", help="Output MP4 when using --drive-video")
    p.add_argument("--webcam", type=int, default=-1, help="Webcam device id (e.g. 0)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=360)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    engine = BodyAvatarEngine(device=device)
    tracker = LiveBodyPoseTracker(image_size=384)
    work_dir = Path(args.work_dir)
    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else default_pretrained_checkpoint()

    _, _, fps = _train_reference(
        engine, Path(args.reference).resolve(), work_dir, args.epochs, args.max_frames, init_ckpt
    )

    if args.webcam >= 0:
        _run_webcam(engine, tracker, args.webcam)
    elif args.drive_video:
        out = Path(args.output) if args.output else work_dir / "avatar_driven.mp4"
        _render_drive_video(engine, tracker, Path(args.drive_video).resolve(), out, fps, args.max_frames)
    else:
        ref_npz = work_dir / "processed" / f"ref_{_slug(Path(args.reference).stem)}" / "processed.npz"
        out = Path(args.output) if args.output else work_dir / "avatar.mp4"
        stats = engine.render_video(ref_npz, out, fps=fps)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
