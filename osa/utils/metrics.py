from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return float(10 * torch.log10(torch.tensor(1.0 / mse)))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity

    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    scores = []
    for p, t in zip(pred_np, target_np):
        scores.append(
            structural_similarity(p, t, channel_axis=-1, data_range=1.0)
        )
    return float(sum(scores) / len(scores))


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_fn) -> float:
    pred = pred.clamp(0, 1) * 2 - 1
    target = target.clamp(0, 1) * 2 - 1
    return float(lpips_fn(pred, target).mean().item())


def apply_mask(t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """mask: (B, 1, H, W) or (B, H, W)."""
    if mask is None:
        return t
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return t * mask


def compute_psnr_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    pred = apply_mask(pred.clamp(0, 1), mask)
    target = apply_mask(target.clamp(0, 1), mask)
    if mask is not None:
        m = mask if mask.dim() == 4 else mask.unsqueeze(1)
        mse = ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1.0)
    else:
        mse = torch.mean((pred - target) ** 2)
    mse = float(mse.item())
    if mse == 0:
        return 99.0
    return float(10 * torch.log10(torch.tensor(1.0 / mse)))


def compute_ssim_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    return compute_ssim(apply_mask(pred, mask), apply_mask(target, mask))


def compute_lpips_masked(pred: torch.Tensor, target: torch.Tensor, lpips_fn, mask: torch.Tensor | None) -> float:
    return compute_lpips(apply_mask(pred, mask), apply_mask(target, mask), lpips_fn)
