#!/usr/bin/env python3
"""Quick BodyAvatar vs KNN on neuman_bike val."""

from __future__ import annotations

from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader

from osa.utils.metrics import compute_lpips_masked, compute_psnr_masked, compute_ssim_masked
from osa_body.baselines import KNNBodyPiecewiseBlend
from osa_body.dataset import BodyVideoDataset, collate_fn
from osa_body.model import BodyAvatarModel
from osa_body.retrieval_bank import BodyTrainBank


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processed = Path("data/body/processed")
    split_file = Path("runs/body_neuman_v2/splits/neuman_bike.json")
    loader = DataLoader(
        BodyVideoDataset(processed, split="val", split_file=split_file),
        batch_size=1,
        collate_fn=collate_fn,
    )
    bank = BodyTrainBank(processed, split_file, warp="piecewise")
    bank.preload(["neuman_bike"])
    lp = lpips.LPIPS(net="alex").to(device).eval()
    baseline = KNNBodyPiecewiseBlend(processed, k=5)
    baseline.store = bank
    ckpt = torch.load(
        "runs/body_neuman_v2/neuman_bike/best.pt", map_location=device, weights_only=False
    )
    model = BodyAvatarModel(image_size=384, knn_k=5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def score(pred, frames, masks):
        b, t, c, h, w = pred.shape
        pf = pred.reshape(b * t, c, h, w)
        tf = frames.reshape(b * t, c, h, w)
        mf = masks.reshape(b * t, h, w)
        m = {
            "psnr": compute_psnr_masked(pf, tf, mf),
            "ssim": compute_ssim_masked(pf, tf, mf),
            "lpips": compute_lpips_masked(pf, tf, lp, mf),
        }
        m["combined"] = m["psnr"] + 5 * m["ssim"] - 15 * m["lpips"]
        return m

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            masks = batch["masks"].to(device)
            pred = baseline.predict(
                batch["subject"][0],
                batch["landmarks"][0].to(device),
                device,
                exclude_indices=set(batch["frame_indices"][0]),
            ).unsqueeze(0)
            m = score(pred, frames, masks)
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
            lpipss.append(m["lpips"])
    n = len(psnrs)
    b_m = {
        "psnr": sum(psnrs) / n,
        "ssim": sum(ssims) / n,
        "lpips": sum(lpipss) / n,
    }
    b_m["combined"] = b_m["psnr"] + 5 * b_m["ssim"] - 15 * b_m["lpips"]
    print(f"KNNBodyPiecewiseBlend: {b_m}")

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            masks = batch["masks"].to(device)
            out = model(
                frames=frames,
                landmarks=batch["landmarks"].to(device),
                retrieval_bank=bank,
                subjects=batch["subject"],
                exclude_frame_indices=[set(batch["frame_indices"][0])],
            )
            m = score(out["pred"], frames, masks)
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
            lpipss.append(m["lpips"])
    n = len(psnrs)
    o_m = {
        "psnr": sum(psnrs) / n,
        "ssim": sum(ssims) / n,
        "lpips": sum(lpipss) / n,
    }
    o_m["combined"] = o_m["psnr"] + 5 * o_m["ssim"] - 15 * o_m["lpips"]
    print(f"BodyAvatar: {o_m}")
    print(f"Delta combined: {o_m['combined'] - b_m['combined']:+.3f}")


if __name__ == "__main__":
    main()
