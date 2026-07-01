#!/usr/bin/env python3
"""Download NeuMan monocular videos and preprocess for BodyAvatar training."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from osa.data.splits import build_clips, save_split, split_clips_temporal
from osa_body.preprocess import BodyPoseTracker

# Standard NeuMan sequences used in avatar papers (bike, citron, jogging, seattle).
NEUMAN_VIDEOS = {
    "bike": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/bike.mp4",
    "citron": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/citron.mp4",
    "jogging": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/jogging.mp4",
    "seattle": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/seattle.mp4",
    "lab": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/lab.mp4",
    "parkinglot": "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/parkinglot.mp4",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  skip download (exists): {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequences",
        nargs="+",
        default=["bike", "citron", "jogging", "seattle"],
        choices=sorted(NEUMAN_VIDEOS.keys()),
    )
    p.add_argument("--raw-dir", default="data/body/raw/neuman")
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/neuman_split.json")
    p.add_argument("--clip-length", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=360)
    p.add_argument("--image-size", type=int, default=384)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    tracker = BodyPoseTracker(image_size=args.image_size)
    manifest = {"sequences": {}, "split_file": args.split_file}

    for name in args.sequences:
        url = NEUMAN_VIDEOS[name]
        subject = f"neuman_{name}"
        mp4 = raw_dir / f"{name}.mp4"
        out = processed_dir / subject
        download(url, mp4)
        print(f"Processing {name} -> {out}")
        meta = tracker.process_video(mp4, out, max_frames=args.max_frames)
        meta["dataset"] = "NeuMan"
        meta["sequence"] = name
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        manifest["sequences"][name] = {
            "subject": subject,
            "frames": meta["num_frames"],
            "source": str(mp4),
        }
        print(f"  {subject}: {meta['num_frames']} frames")

    clips = build_clips(processed_dir, clip_length=args.clip_length)
    neuman_clips = [c for c in clips if c["subject"].startswith("neuman_")]
    train, val = split_clips_temporal(neuman_clips, train_ratio=0.85)
    split_path = Path(args.split_file)
    save_split(split_path, train, val)
    manifest["train_clips"] = len(train)
    manifest["val_clips"] = len(val)
    manifest_path = processed_dir.parent / "neuman_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSplit: {len(train)} train / {len(val)} val clips -> {split_path}")
    print(f"Manifest: {manifest_path}")
    print("\nNext:")
    print("  python scripts/body/run_neuman_benchmark.py --epochs 25 --gpu 0")


if __name__ == "__main__":
    main()
