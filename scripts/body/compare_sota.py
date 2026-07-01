#!/usr/bin/env python3
"""Full-body benchmark: BodyAvatar vs classical + SOTA-proxy baselines."""

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
    KNNBodyPiecewiseBlend,
    NearestTrainBodyPiecewise,
    NearestTrainBodyProcrustes,
    NearestTrainBodyTPS,
    NearestTrainBodyWarp,
    NeutralAtlasBodyPiecewise,
)
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.literature_sota import (
    SOTA_LITERATURE_LEGACY,
    build_literature_comparison,
)
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank

# In-repo proxies mapping classical methods to SOTA categories
SOTA_PROXY_MAP = {
    "NeutralAtlasBodyPiecewise": "InstantAvatar / fixed-atlas geometry proxy",
    "NearestTrainBodyPiecewise": "Mesh / 3DMM piecewise warp proxy",
    "KNNBodyBlendWarp": "NeuralBody-style retrieval ensemble proxy",
    "NearestTrainBodyProcrustes": "Parametric similarity warp proxy",
    "KNNBodyPiecewiseBlend": "Retrieval + mesh warp (strong classical SOTA proxy)",
}


def combined(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 10.0 * m["lpips"]


@torch.no_grad()
def eval_model(model, loader, device, lpips_fn, bank, label="BodyAvatar"):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in tqdm(loader, desc=label, leave=False):
        frames = batch["frames"].to(device)
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
        psnrs.append(compute_psnr(pred_flat, target_flat))
        ssims.append(compute_ssim(pred_flat, target_flat))
        lpipss.append(compute_lpips(pred_flat, target_flat, lpips_fn))
    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": sum(lpipss) / len(lpipss),
    }


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
    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "lpips": sum(lpipss) / len(lpipss),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/bench_split.json")
    p.add_argument("--checkpoint", default="runs/body_avatar/best.pt")
    p.add_argument("--output", default="runs/body_sota_comparison.json")
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

    methods: dict[str, dict] = {}
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = BodyAvatarModel(
            image_size=ckpt.get("args", {}).get("image_size", 384),
            knn_k=ckpt.get("args", {}).get("knn_k", 3),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        methods["BodyAvatar"] = eval_model(model, val_loader, device, lpips_fn, bank)

    baselines = [
        KNNBodyPiecewiseBlend(data_root),
        KNNBodyBlendWarp(data_root),
        NearestTrainBodyPiecewise(data_root),
        NearestTrainBodyProcrustes(data_root),
        NearestTrainBodyTPS(data_root),
        NearestTrainBodyWarp(data_root),
        NeutralAtlasBodyPiecewise(data_root),
        FirstFrameBodyWarp(data_root),
    ]
    for base in baselines:
        methods[base.name] = eval_baseline(base, val_loader, device, lpips_fn)

    for name, m in methods.items():
        m["combined"] = combined(m)
        if name in SOTA_PROXY_MAP:
            m["sota_proxy"] = SOTA_PROXY_MAP[name]

    ranked = sorted(methods.keys(), key=lambda k: methods[k]["combined"], reverse=True)
    best_proxy = next((n for n in ranked if n != "BodyAvatar"), ranked[0])
    body = methods.get("BodyAvatar", {})
    vs_best = {}
    if body and best_proxy in methods and best_proxy != "BodyAvatar":
        bp = methods[best_proxy]
        vs_best = {
            "vs": best_proxy,
            "d_psnr": body["psnr"] - bp["psnr"],
            "d_ssim": body["ssim"] - bp["ssim"],
            "d_lpips": bp["lpips"] - body["lpips"],
            "d_combined": body["combined"] - bp["combined"],
        }

    results = {
        "dataset": str(data_root),
        "task": "full_body_self_reenactment",
        "metrics": "PSNR SSIM LPIPS on same val split",
        "methods": methods,
        "ranking": ranked,
        "best_overall": ranked[0],
        "body_avatar_vs_best_baseline": vs_best,
        "literature_comparison_2025_2026": build_literature_comparison(body),
        "sota_literature_reference": SOTA_LITERATURE_LEGACY,
        "sota_proxy_legend": SOTA_PROXY_MAP,
        "note": (
            "Unified benchmark on 3 synthetic body subjects. "
            "Literature PSNR values are on real datasets (ZJU, NeuMan, WildAvatar, etc.) "
            "and are NOT directly comparable. See literature_comparison_2025_2026 for "
            "12+ published 2025–2026 methods with normalized LPIPS."
        ),
    }
    Path(args.output).write_text(json.dumps(results, indent=2))

    print("\n=== Full-body SOTA-style benchmark (same val split) ===")
    print(f"{'Method':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Combined':>10}")
    print("-" * 70)
    for i, name in enumerate(ranked, 1):
        m = methods[name]
        tag = " *" if name == "BodyAvatar" else ""
        print(
            f"{name:<30} {m['psnr']:8.2f} {m['ssim']:8.3f} {m['lpips']:8.3f} "
            f"{m['combined']:10.2f}  #{i}{tag}"
        )
    if vs_best:
        print(f"\nBodyAvatar vs {vs_best['vs']}: dCombined={vs_best['d_combined']:+.3f}")

    lit = results["literature_comparison_2025_2026"]
    print("\n=== Literature comparison (2025–2026, reference only — different datasets) ===")
    print(
        f"{'Method':<28} {'Yr':>4} {'Dataset':<22} {'PSNR':>7} {'SSIM':>7} "
        f"{'LPIPS*':>8} {'Task'}"
    )
    print("-" * 110)
    ours = lit["ours"]
    print(
        f"{ours['method']:<28} {ours['year']:>4} {'synthetic (ours)':<22} "
        f"{ours['psnr']:7.2f} {ours['ssim']:7.3f} {ours['lpips']:8.3f}  "
        f"{ours['task'][:40]}"
    )
    for p in sorted(lit["papers_2025_2026"], key=lambda x: x.get("psnr", 0), reverse=True):
        lp = p.get("lpips_normalized", p.get("lpips", float("nan")))
        ds = p.get("dataset", "")[:21]
        task = p.get("task", "")[:35]
        print(
            f"{p['method']:<28} {p['year']:>4} {ds:<22} "
            f"{p.get('psnr', 0):7.2f} {p.get('ssim_normalized', p.get('ssim', 0)):7.3f} "
            f"{lp:8.4f}  {task}"
        )
    print("\n* LPIPS normalized to standard [0,1] scale for rough comparison.")
    print(f"  {lit['disclaimer']}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
