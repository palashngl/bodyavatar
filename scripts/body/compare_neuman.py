#!/usr/bin/env python3
"""Compare BodyAvatar vs baselines on NeuMan per-subject val splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from osa.utils.metrics import (
    compute_lpips_masked,
    compute_psnr_masked,
    compute_ssim_masked,
)
from osa_body.baselines import (
    FirstFrameBodyWarp,
    KNNBodyBlendWarp,
    KNNBodyPiecewiseBlend,
    NearestTrainBodyPiecewise,
    NearestTrainBodyProcrustes,
)
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.model_v3 import BodyAvatarModelV3
from osa_body.model_v4 import BodyAvatarModelV4
from osa_body.retrieval_bank import BodyTrainBank


def load_model(ckpt_path: Path, device: torch.device, version: str = "auto"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "body_v2")
    if version == "v4" or (version == "auto" and "v4" in model_type):
        model = BodyAvatarModelV4(
            image_size=ckpt.get("args", {}).get("image_size", 384),
            knn_k=ckpt.get("args", {}).get("knn_k", 5),
        ).to(device)
    elif version == "v3" or (version == "auto" and "v3" in model_type):
        model = BodyAvatarModelV3(
            image_size=ckpt.get("args", {}).get("image_size", 384),
            knn_k=ckpt.get("args", {}).get("knn_k", 5),
        ).to(device)
    else:
        model = BodyAvatarModel(
            image_size=ckpt.get("args", {}).get("image_size", 384),
            knn_k=ckpt.get("args", {}).get("knn_k", 5),
        ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def combined(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 15.0 * m["lpips"]


def _metrics(pred, target, mask, lpips_fn):
    return {
        "psnr": compute_psnr_masked(pred, target, mask),
        "ssim": compute_ssim_masked(pred, target, mask),
        "lpips": compute_lpips_masked(pred, target, lpips_fn, mask),
    }


@torch.no_grad()
def eval_model(model, loader, device, lpips_fn, bank):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in tqdm(loader, desc="BodyAvatar", leave=False):
        frames = batch["frames"].to(device)
        masks = batch.get("masks")
        if masks is not None:
            masks = masks.to(device)
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
        mask_flat = masks.reshape(b * t, h, w) if masks is not None else None
        m = _metrics(pred_flat, target_flat, mask_flat, lpips_fn)
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
        lpipss.append(m["lpips"])
    n = max(len(psnrs), 1)
    return {"psnr": sum(psnrs) / n, "ssim": sum(ssims) / n, "lpips": sum(lpipss) / n}


@torch.no_grad()
def eval_baseline(baseline, loader, device, lpips_fn):
    psnrs, ssims, lpipss = [], [], []
    for batch in tqdm(loader, desc=baseline.name, leave=False):
        frames = batch["frames"].to(device)
        masks = batch.get("masks")
        if masks is not None:
            masks = masks.to(device)
        subject = batch["subject"][0]
        landmarks = batch["landmarks"][0].to(device)
        exclude = set(batch["frame_indices"][0])
        pred = baseline.predict(subject, landmarks, device, exclude_indices=exclude).unsqueeze(0)
        b, t, c, h, w = pred.shape
        pred_flat = pred.reshape(b * t, c, h, w)
        target_flat = frames.reshape(b * t, c, h, w)
        mask_flat = masks.reshape(b * t, h, w) if masks is not None else None
        m = _metrics(pred_flat, target_flat, mask_flat, lpips_fn)
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
        lpipss.append(m["lpips"])
    n = max(len(psnrs), 1)
    return {"psnr": sum(psnrs) / n, "ssim": sum(ssims) / n, "lpips": sum(lpipss) / n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--sequences", nargs="+", default=["bike", "citron", "jogging", "seattle"])
    p.add_argument("--split-dir", default="runs/body_neuman/splits")
    p.add_argument("--checkpoint-dir", default="runs/body_neuman_v2")
    p.add_argument("--output", default="runs/body_neuman_v2/comparison.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model-version", default="auto", choices=["auto", "v2", "v3", "v4"])
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    processed = Path(args.processed_dir)
    all_results: dict[str, dict] = {}

    for seq in args.sequences:
        subject = f"neuman_{seq}"
        split_file = Path(args.split_dir) / f"{subject}.json"
        ckpt_path = Path(args.checkpoint_dir) / subject / "best.pt"
        if not split_file.exists():
            print(f"skip {subject}: missing split {split_file}")
            continue

        val_loader = DataLoader(
            BodyVideoDataset(processed, split="val", split_file=split_file),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        bank = BodyTrainBank(processed, split_file, warp="piecewise")
        bank.preload([subject])

        methods: dict[str, dict] = {}
        baselines = [
            KNNBodyPiecewiseBlend(processed, k=5),
            KNNBodyBlendWarp(processed, k=5),
            NearestTrainBodyPiecewise(processed),
            NearestTrainBodyProcrustes(processed),
            FirstFrameBodyWarp(processed),
        ]
        for b in baselines:
            b.store = bank
            methods[b.name] = eval_baseline(b, val_loader, device, lpips_fn)

        if ckpt_path.exists():
            model = load_model(ckpt_path, device, args.model_version)
            methods["BodyAvatar"] = eval_model(model, val_loader, device, lpips_fn, bank)

        for name, m in methods.items():
            m["combined"] = combined(m)
        ranked = sorted(methods, key=lambda k: methods[k]["combined"], reverse=True)
        all_results[subject] = {"methods": methods, "ranking": ranked, "best": ranked[0]}

        print(f"\n=== {subject} (masked metrics) ===")
        for name in ranked:
            m = methods[name]
            tag = " *" if name == "BodyAvatar" else ""
            print(f"  {name:<28} PSNR={m['psnr']:6.2f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f}{tag}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
