#!/usr/bin/env python3
"""Export retrieval initialization packages for 3DGS fine-tuning (SOTA path)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from osa.data.splits import build_clips, load_split, split_clips_temporal, save_split
from osa_body.retrieval_gs_init import RetrievalGSInitializer, RetrievalInitConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. neuman_bike")
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--split-file", default="")
    p.add_argument("--output", default="runs/gs_init")
    p.add_argument("--knn-k", type=int, default=5)
    args = p.parse_args()

    processed = Path(args.processed_dir)
    subject = args.subject
    npz = processed / subject / "processed.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}")

    if args.split_file:
        split_file = Path(args.split_file)
    else:
        clips = build_clips(processed, 8)
        subj = [c for c in clips if c["subject"] == subject]
        train, val = split_clips_temporal(subj, 0.85)
        split_file = Path(args.output) / f"{subject}_split.json"
        save_split(split_file, train, val)

    data = np.load(npz)
    landmarks = torch.from_numpy(data["landmarks"]).float()
    train_clips, _ = load_split(split_file)
    frame_indices = sorted(
        {i for c in train_clips if c["subject"] == subject for i in range(c["start"], c["start"] + c["clip_length"])}
    )

    init = RetrievalGSInitializer(processed, split_file, RetrievalInitConfig(knn_k=args.knn_k))
    init.preload([subject])
    out_path = init.export_init_package(subject, Path(args.output), frame_indices, landmarks)
    print(f"Saved 3DGS init package: {out_path}")
    print("Next: fine-tune 3DGS with gsplat/GauHuman using retrieval as photometric prior.")


if __name__ == "__main__":
    main()
