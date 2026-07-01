#!/usr/bin/env python3
"""Train BodyAvatar v3 on synthetic full-body data (pretrain for real-video finetune)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from osa.utils.metrics import compute_lpips, compute_psnr, compute_ssim
from osa.utils.seed import set_seed
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model_v3 import BodyAvatarModelV3, v3_composite_loss
from osa_body.retrieval_bank import BodyTrainBank


def score(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 15.0 * m["lpips"]


@torch.no_grad()
def evaluate(model, loader, device, lpips_fn, bank) -> dict[str, float]:
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in loader:
        out = model(
            frames=batch["frames"].to(device),
            landmarks=batch["landmarks"].to(device),
            retrieval_bank=bank,
            subjects=batch["subject"],
            exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
        )
        pred = out["pred"]
        target = batch["frames"].to(device)
        b, t, c, h, w = pred.shape
        pf = pred.reshape(b * t, c, h, w)
        tf = target.reshape(b * t, c, h, w)
        psnrs.append(compute_psnr(pf, tf))
        ssims.append(compute_ssim(pf, tf))
        lpipss.append(compute_lpips(pf, tf, lpips_fn))
    n = max(len(psnrs), 1)
    return {"psnr": sum(psnrs) / n, "ssim": sum(ssims) / n, "lpips": sum(lpipss) / n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/bench_split.json")
    p.add_argument("--output", default="runs/body_v3_synthetic")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--clip-length", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--retrieval-anchor", type=float, default=0.06)
    p.add_argument("--residual-reg", type=float, default=0.008)
    args = p.parse_args()

    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = BodyTrainBank(args.data, args.split_file, warp="piecewise")
    bank.preload()
    train_loader = DataLoader(
        BodyVideoDataset(args.data, args.clip_length, "train", args.split_file),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        BodyVideoDataset(args.data, args.clip_length, "val", args.split_file),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = BodyAvatarModelV3(knn_k=args.knn_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    best = -1e9
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            frames = batch["frames"].to(device)
            model_out = model(
                frames=frames,
                landmarks=batch["landmarks"].to(device),
                retrieval_bank=bank,
                subjects=batch["subject"],
                exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
            )
            pred = model_out["pred"]
            b, t, c, h, w = pred.shape
            pf = pred.reshape(b * t, c, h, w)
            tf = frames.reshape(b * t, c, h, w)
            loss, _ = v3_composite_loss(pf, tf, lpips_fn, model_out["confidence"])
            retr = model_out["retrieval"].reshape(b * t, c, h, w)
            loss = loss + F.l1_loss(pf, retr) * args.retrieval_anchor
            loss = loss + model_out["residual"].abs().mean() * args.residual_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        metrics = evaluate(model, val_loader, device, lpips_fn, bank)
        metrics["combined"] = score(metrics)
        if metrics["combined"] >= best:
            best = metrics["combined"]
            torch.save(
                {"model": model.state_dict(), "metrics": metrics, "model_type": "body_v3", "args": vars(args)},
                out_dir / "best.pt",
            )
        print(f"ep {epoch+1}/{args.epochs} val psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.3f} lpips={metrics['lpips']:.3f}")

    (out_dir / "train_summary.json").write_text(json.dumps({"best_combined": best, "checkpoint": str(out_dir / "best.pt")}, indent=2))
    print(f"Saved: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
