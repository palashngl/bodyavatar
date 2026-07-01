#!/usr/bin/env python3
"""Benchmark BodyAvatar vs classical full-body baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from osa.utils.metrics import compute_lpips, compute_psnr, compute_ssim
from osa_body.baselines import (
    FirstFrameBodyWarp,
    KNNBodyBlendWarp,
    NearestTrainBodyProcrustes,
    NearestTrainBodyWarp,
)
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank


def combined(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 10.0 * m["lpips"]


@torch.no_grad()
def eval_model(model, loader, device, lpips_fn, bank=None, label="BodyAvatar"):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in tqdm(loader, desc=label, leave=False):
        frames = batch["frames"].to(device)
        if bank is not None:
            out = model(
                frames=frames,
                landmarks=batch["landmarks"].to(device),
                retrieval_bank=bank,
                subjects=batch["subject"],
                exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
            )
            pred = out["pred"]
        else:
            raise ValueError("bank required")
        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(b * t, c, h, w)
        target_flat = frames.reshape(b * t, c, h, w)
        psnrs.append(compute_psnr(pred_flat, target_flat))
        ssims.append(compute_ssim(pred_flat, target_flat))
        lpipss.append(compute_lpips(pred_flat, target_flat, lpips_fn))
    return {"psnr": sum(psnrs) / len(psnrs), "ssim": sum(ssims) / len(ssims), "lpips": sum(lpipss) / len(lpipss)}


@torch.no_grad()
def eval_baseline(baseline, loader, device, lpips_fn):
    psnrs, ssims, lpipss = [], [], []
    for batch in tqdm(loader, desc=baseline.name, leave=False):
        frames = batch["frames"].to(device)
        subject = batch["subject"][0]
        landmarks = batch["landmarks"][0].to(device)
        exclude = set(batch["frame_indices"][0])
        pred = baseline.predict(subject, landmarks, device, exclude_indices=exclude).unsqueeze(0)
        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(b * t, c, h, w)
        target_flat = frames.reshape(b * t, c, h, w)
        psnrs.append(compute_psnr(pred_flat, target_flat))
        ssims.append(compute_ssim(pred_flat, target_flat))
        lpipss.append(compute_lpips(pred_flat, target_flat, lpips_fn))
    return {"psnr": sum(psnrs) / len(psnrs), "ssim": sum(ssims) / len(ssims), "lpips": sum(lpipss) / len(lpipss)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/bench_split.json")
    p.add_argument("--checkpoint", default="runs/body_avatar/best.pt")
    p.add_argument("--output", default="runs/body_comparison.json")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data)
    val_loader = DataLoader(
        BodyVideoDataset(args.data, split="val", split_file=args.split_file),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    bank = BodyTrainBank(args.data, args.split_file)
    bank.preload()

    methods = {}
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = BodyAvatarModel(
            image_size=ckpt.get("args", {}).get("image_size", 384),
            knn_k=ckpt.get("args", {}).get("knn_k", 3),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        methods["BodyAvatar"] = eval_model(model, val_loader, device, lpips_fn, bank)

    baselines = [
        FirstFrameBodyWarp(data_root),
        NearestTrainBodyWarp(data_root),
        NearestTrainBodyProcrustes(data_root),
        KNNBodyBlendWarp(data_root),
    ]
    for base in baselines:
        methods[base.name] = eval_baseline(base, val_loader, device, lpips_fn)

    ranked = sorted(methods.keys(), key=lambda k: combined(methods[k]), reverse=True)
    results = {"methods": methods, "ranking": ranked, "best": ranked[0]}
    Path(args.output).write_text(json.dumps(results, indent=2))

    print("\n=== Full-body benchmark (validation) ===")
    print(f"{'Method':<28} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Combined':>10}")
    print("-" * 66)
    for i, name in enumerate(ranked, 1):
        m = methods[name]
        print(f"{name:<28} {m['psnr']:8.2f} {m['ssim']:8.3f} {m['lpips']:8.3f} {combined(m):10.2f}  #{i}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
