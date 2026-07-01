#!/usr/bin/env python3
"""Save BodyAvatar visual comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/body_avatar/best.pt")
    p.add_argument("--data", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/bench_split.json")
    p.add_argument("--output", default="docs/body_results")
    p.add_argument("--max-clips", type=int, default=6)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BodyAvatarModel(
        image_size=ckpt.get("args", {}).get("image_size", 384),
        knn_k=ckpt.get("args", {}).get("knn_k", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bank = BodyTrainBank(args.data, args.split_file)
    bank.preload()
    loader = DataLoader(
        BodyVideoDataset(args.data, split="val", split_file=args.split_file),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if i >= args.max_clips:
            break
        frames = batch["frames"].to(device)
        out = model(
            frames=frames,
            landmarks=batch["landmarks"].to(device),
            retrieval_bank=bank,
            subjects=batch["subject"],
            exclude_frame_indices=[set(batch["frame_indices"][0])],
        )
        n = min(4, out["pred"].shape[1])
        panel = torch.cat([frames[0, :n], out["pred"][0, :n], out["retrieval"][0, :n]], dim=0)
        subject = batch["subject"][0]
        vutils.save_image(panel, out_dir / f"{subject}_clip{i:02d}.png", nrow=n)

    (out_dir / "README.txt").write_text(
        "Each PNG: ground_truth | BodyAvatar | retrieval_base (left to right)\n"
    )
    print(f"Saved panels to {out_dir}")


if __name__ == "__main__":
    main()
