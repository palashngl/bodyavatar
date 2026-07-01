#!/usr/bin/env python3
"""
Train + evaluate BodyAvatar v2 on NeuMan real video.

Improvements vs v1:
- Piecewise KNN retrieval + weighted landmarks
- Frame-hint residual refiner, k=5
- Foreground-masked metrics
- Train from scratch on real data (no synthetic init)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from osa.data.splits import build_clips, save_split, split_clips_temporal
from osa.utils.losses import composite_loss
from osa.utils.metrics import compute_lpips_masked, compute_psnr_masked, compute_ssim_masked
from osa.utils.seed import set_seed
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.literature_sota import build_literature_comparison
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank

DEFAULT_SEQUENCES = ["bike", "citron", "jogging", "seattle"]


def score(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 15.0 * m["lpips"]


def _flat_mask(batch: dict, device: torch.device):
    masks = batch.get("masks")
    if masks is None:
        return None
    b, t, h, w = masks.shape
    return masks.to(device).reshape(b * t, h, w)


@torch.no_grad()
def evaluate(model, loader, device, lpips_fn, bank) -> dict[str, float]:
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        mask_flat = _flat_mask(batch, device)
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
        psnrs.append(compute_psnr_masked(pred_flat, target_flat, mask_flat))
        ssims.append(compute_ssim_masked(pred_flat, target_flat, mask_flat))
        lpipss.append(compute_lpips_masked(pred_flat, target_flat, lpips_fn, mask_flat))
    n = max(len(psnrs), 1)
    return {"psnr": sum(psnrs) / n, "ssim": sum(ssims) / n, "lpips": sum(lpipss) / n}


def train_one_subject(
    subject: str,
    processed_root: Path,
    split_file: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    init_checkpoint: Path | None,
) -> dict[str, float]:
    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = BodyTrainBank(str(processed_root), str(split_file), warp="piecewise")
    bank.preload([subject])
    train_ds = BodyVideoDataset(processed_root, args.clip_length, "train", split_file)
    val_ds = BodyVideoDataset(processed_root, args.clip_length, "val", split_file)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = BodyAvatarModel(image_size=args.image_size, knn_k=args.knn_k).to(device)
    if init_checkpoint and init_checkpoint.exists():
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  partial init from {init_checkpoint}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    best_metrics = {"combined": -1e9}
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            frames = batch["frames"].to(device)
            landmarks = batch["landmarks"].to(device)
            mask_flat = _flat_mask(batch, device)
            out = model(
                frames=frames,
                landmarks=landmarks,
                retrieval_bank=bank,
                subjects=batch["subject"],
                exclude_frame_indices=[set(fi) for fi in batch["frame_indices"]],
            )
            pred = out["pred"]
            b, t, c, h, w = pred.shape
            pred_flat = pred.reshape(b * t, c, h, w)
            target_flat = frames.reshape(b * t, c, h, w)
            if mask_flat is not None:
                m = mask_flat.unsqueeze(1)
                pred_flat = pred_flat * m
                target_flat = target_flat * m
            loss, _ = composite_loss(pred_flat, target_flat, lpips_fn, out["confidence"])
            retr_flat = out["retrieval"].reshape(b * t, c, h, w)
            if mask_flat is not None:
                retr_flat = retr_flat * m
            loss = loss + F.l1_loss(pred_flat, retr_flat) * args.retrieval_anchor
            loss = loss + out["residual"].abs().mean() * args.residual_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        metrics = evaluate(model, val_loader, device, lpips_fn, bank)
        metrics["combined"] = score(metrics)
        if metrics["combined"] >= best_metrics.get("combined", -1e9):
            best_metrics = metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "subject": subject,
                    "args": vars(args),
                    "model_type": "body_v2",
                },
                out_dir / "best.pt",
            )
        print(
            f"  [{subject}] ep {epoch+1}/{args.epochs} "
            f"val psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.3f} lpips={metrics['lpips']:.3f}"
        )

    return best_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    p.add_argument("--output", default="runs/body_neuman_v2")
    p.add_argument("--init-checkpoint", default="", help="Optional; empty = train from scratch")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--clip-length", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval-anchor", type=float, default=0.12)
    p.add_argument("--residual-reg", type=float, default=0.01)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    processed_root = Path(args.processed_dir)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else None
    results: dict[str, dict] = {}

    for seq in args.sequences:
        subject = f"neuman_{seq}"
        subject_dir = processed_root / subject
        if not (subject_dir / "processed.npz").exists():
            print(f"Missing {subject} — run prepare_neuman.py")
            continue
        data = __import__("numpy").load(subject_dir / "processed.npz")
        if "masks" not in data:
            print(f"Missing masks in {subject} — re-run prepare_neuman.py --sequences {seq}")
            continue

        clips = build_clips(processed_root, args.clip_length)
        subj_clips = [c for c in clips if c["subject"] == subject]
        train, val = split_clips_temporal(subj_clips, train_ratio=0.85)
        split_file = Path(args.output) / f"splits/{subject}.json"
        save_split(split_file, train, val)

        print(f"\n=== {subject} ({len(train)} train / {len(val)} val clips) ===")
        metrics = train_one_subject(
            subject, processed_root, split_file, Path(args.output) / subject, args, device, init_ckpt
        )
        metrics["combined"] = score(metrics)
        results[subject] = metrics

    if not results:
        raise SystemExit("No sequences trained.")

    mean = {
        "psnr": sum(m["psnr"] for m in results.values()) / len(results),
        "ssim": sum(m["ssim"] for m in results.values()) / len(results),
        "lpips": sum(m["lpips"] for m in results.values()) / len(results),
    }
    mean["combined"] = score(mean)

    report = {
        "dataset": "NeuMan",
        "model": "BodyAvatar v2 (piecewise KNN + frame-hint refiner)",
        "metrics": "foreground-masked PSNR/SSIM/LPIPS",
        "per_subject": results,
        "mean": mean,
        "literature_context": build_literature_comparison(mean),
    }
    out_json = Path(args.output) / "neuman_benchmark.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("\n=== NeuMan v2 summary (masked) ===")
    for subj, m in results.items():
        print(f"  {subj}: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f}")
    print(f"  MEAN: PSNR={mean['psnr']:.2f} SSIM={mean['ssim']:.3f} LPIPS={mean['lpips']:.3f}")
    print(f"Saved: {out_json}")
    print("\nRun baseline comparison:")
    print(f"  python scripts/body/compare_neuman.py --checkpoint-dir {args.output}")


if __name__ == "__main__":
    main()
