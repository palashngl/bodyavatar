#!/usr/bin/env python3
"""Train BodyAvatar full-body video-to-avatar model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from osa.utils.losses import composite_loss
from osa.utils.metrics import compute_lpips, compute_psnr, compute_ssim
from osa.utils.seed import set_seed
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank


def score(metrics: dict[str, float]) -> float:
    return metrics["psnr"] + 5.0 * metrics["ssim"] - 15.0 * metrics["lpips"]


def batch_to_model(batch: dict, device: torch.device) -> dict:
    return {
        "frames": batch["frames"].to(device),
        "landmarks": batch["landmarks"].to(device),
        "subjects": batch["subject"],
        "exclude_frame_indices": [set(fi) for fi in batch["frame_indices"]],
    }


@torch.no_grad()
def evaluate(model, loader, device, lpips_fn, bank) -> dict[str, float]:
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in loader:
        b = batch_to_model(batch, device)
        out = model(
            frames=b["frames"],
            landmarks=b["landmarks"],
            retrieval_bank=bank,
            subjects=b["subjects"],
            exclude_frame_indices=b["exclude_frame_indices"],
        )
        pred = out["pred"]
        target = batch["frames"].to(device)
        bsz, t, c, h, w = pred.shape
        pred_flat = pred.reshape(bsz * t, c, h, w)
        target_flat = target.reshape(bsz * t, c, h, w)
        psnrs.append(compute_psnr(pred_flat, target_flat))
        ssims.append(compute_ssim(pred_flat, target_flat))
        lpipss.append(compute_lpips(pred_flat, target_flat, lpips_fn))
    return {
        "psnr": float(sum(psnrs) / max(len(psnrs), 1)),
        "ssim": float(sum(ssims) / max(len(ssims), 1)),
        "lpips": float(sum(lpipss) / max(len(lpipss), 1)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = BodyTrainBank(args.data, args.split_file)
    bank.preload()
    print("Body bank subjects:", bank.subjects())

    train_ds = BodyVideoDataset(args.data, args.clip_length, "train", args.split_file)
    val_ds = BodyVideoDataset(args.data, args.clip_length, "val", args.split_file)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    model = BodyAvatarModel(image_size=args.image_size, knn_k=args.knn_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    writer = SummaryWriter(out_dir / "tb")

    best_score = -1e9
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            b = batch_to_model(batch, device)
            out = model(
                frames=b["frames"],
                landmarks=b["landmarks"],
                retrieval_bank=bank,
                subjects=b["subjects"],
                exclude_frame_indices=b["exclude_frame_indices"],
            )
            pred = out["pred"]
            target = batch["frames"].to(device)
            bsz, tlen, c, h, w = pred.shape
            pred_flat = pred.reshape(bsz * tlen, c, h, w)
            target_flat = target.reshape(bsz * tlen, c, h, w)
            loss, parts = composite_loss(pred_flat, target_flat, lpips_fn, out["confidence"])
            retr_flat = out["retrieval"].reshape(bsz * tlen, c, h, w)
            anchor = F.l1_loss(pred_flat, retr_flat) * args.retrieval_anchor
            res_reg = out["residual"].abs().mean() * args.residual_reg
            loss = loss + anchor + res_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr_anchor=f"{anchor.item():.4f}")

        sched.step()
        metrics = evaluate(model, val_loader, device, lpips_fn, bank)
        metrics["combined"] = score(metrics)
        writer.add_scalar("val/combined", metrics["combined"], epoch)
        print("val:", metrics)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "metrics": metrics, "args": vars(args), "model_type": "body_v1"}
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["combined"] >= best_score:
            best_score = metrics["combined"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"saved best combined={best_score:.3f} psnr={metrics['psnr']:.2f}")

    (out_dir / "train_summary.json").write_text(json.dumps({"best_combined": best_score}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/body/processed")
    p.add_argument("--split-file", default="data/body/splits/bench_split.json")
    p.add_argument("--output", default="runs/body_avatar")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--clip-length", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--knn-k", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval-anchor", type=float, default=0.35)
    p.add_argument("--residual-reg", type=float, default=0.02)
    train(p.parse_args())


if __name__ == "__main__":
    main()
