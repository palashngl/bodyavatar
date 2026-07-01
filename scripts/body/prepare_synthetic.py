#!/usr/bin/env python3
"""Build synthetic full-body video subjects for BodyAvatar training."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import cv2
import numpy as np
import requests

from osa_body.preprocess import BodyPoseTracker


PORTRAITS = [
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=640",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=640",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640",
]


def download_image(url: str, dest: Path) -> np.ndarray:
    headers = {"User-Agent": "BodyAvatar/0.1 research"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_body_motion_video(base_rgb: np.ndarray, out_video: Path, frames: int = 240) -> None:
    h, w = base_rgb.shape[:2]
    scale = 720 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    base = cv2.resize(base_rgb, (nw, nh))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, 25.0, (nw, nh))
    cx, cy = nw / 2, nh / 2
    for t in range(frames):
        angle = 6 * np.sin(2 * np.pi * t / frames)
        scale_t = 1.0 + 0.08 * np.sin(4 * np.pi * t / frames)
        tx = int(10 * np.sin(2 * np.pi * t / 60))
        ty = int(6 * np.cos(2 * np.pi * t / 45))
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale_t)
        M[0, 2] += tx
        M[1, 2] += ty
        warped = cv2.warpAffine(base, M, (nw, nh), borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    raw = Path("data/body/raw")
    out = Path("data/body/processed")
    tracker = BodyPoseTracker(image_size=384)
    ok = 0
    for i, url in enumerate(PORTRAITS):
        subject = f"body_{i}"
        img_path = raw / f"{subject}.jpg"
        vid_path = raw / f"{subject}.mp4"
        try:
            rgb = download_image(url, img_path)
            make_body_motion_video(rgb, vid_path)
            tracker.process_video(vid_path, out / subject, max_frames=240)
            print(f"OK {subject}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {subject}: {exc}")
    print(f"Prepared {ok}/{len(PORTRAITS)} body subjects -> {out}")


if __name__ == "__main__":
    main()
