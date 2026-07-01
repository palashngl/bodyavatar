from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device: torch.device, channels: int) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable 1 - SSIM (lower is better)."""
    c = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, pred.device, c)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu12
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1.0 - ssim_map.mean()


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn,
    confidence: torch.Tensor,
    w_l1: float = 1.0,
    w_lpips: float = 0.15,
    w_ssim: float = 0.25,
    w_conf: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(pred, target)
    lp = lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
    ss = ssim_loss(pred, target)
    conf_reg = (1.0 - confidence.mean()).abs() * w_conf
    loss = w_l1 * l1 + w_lpips * lp + w_ssim * ss + conf_reg
    return loss, {
        "l1": l1.item(),
        "lpips": lp.item(),
        "ssim_loss": ss.item(),
        "conf": conf_reg.item(),
    }
