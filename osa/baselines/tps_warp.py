from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    r2 = np.maximum(r2, 1e-12)
    return r2 * np.log(r2)


def _dedupe_control(control: np.ndarray, target: np.ndarray, min_dist: float = 1.0):
    """Drop near-duplicate control points that make the TPS system singular."""
    keep = [0]
    for i in range(1, control.shape[0]):
        if np.min(np.linalg.norm(control[keep] - control[i], axis=1)) >= min_dist:
            keep.append(i)
    if len(keep) < 3:
        return control, target
    idx = np.asarray(keep, dtype=np.int64)
    return control[idx], target[idx]


def _fit_tps(control: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit TPS mapping control -> target. control/target: (N, 2)."""
    control, target = _dedupe_control(control, target)
    n = control.shape[0]
    diff = control[:, None, :] - control[None, :, :]
    k = _tps_kernel((diff**2).sum(axis=-1))
    k = k + 1e-3 * np.eye(n)
    p = np.concatenate([np.ones((n, 1)), control], axis=1)
    upper = np.concatenate([k, p], axis=1)
    lower = np.concatenate([p.T, np.zeros((3, 3))], axis=1)
    system = np.concatenate([upper, lower], axis=0)
    rhs = np.concatenate([target, np.zeros((3, 2))], axis=0)
    try:
        params = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return params[:n], params[n:], control


def _apply_tps(points: np.ndarray, w: np.ndarray, a: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Apply TPS to points (M, 2)."""
    diff = points[:, None, :] - control[None, :, :]
    k = _tps_kernel((diff**2).sum(axis=-1))
    return k @ w + np.concatenate([np.ones((points.shape[0], 1)), points], axis=1) @ a


def tps_warp_image(
    image: torch.Tensor,
    src_lmk: torch.Tensor,
    dst_lmk: torch.Tensor,
) -> torch.Tensor:
    """
    Warp image so src landmarks align to dst positions (inverse sampling).
    image: (1, C, H, W), landmarks: (1, N, 2) in pixel coords.
    """
    _, _, h, w = image.shape
    device, dtype = image.device, image.dtype
    src = src_lmk[0, :, :2].detach().cpu().numpy()
    dst = dst_lmk[0, :, :2].detach().cpu().numpy()
    w_coef, a_coef, ctrl = _fit_tps(dst, src)

    ys = np.linspace(0, h - 1, h, dtype=np.float32)
    xs = np.linspace(0, w - 1, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    src_pts = _apply_tps(pts, w_coef, a_coef, ctrl)
    src_x = src_pts[:, 0].reshape(h, w)
    src_y = src_pts[:, 1].reshape(h, w)
    grid_x_norm = (src_x / (w - 1)) * 2 - 1
    grid_y_norm = (src_y / (h - 1)) * 2 - 1
    grid = np.stack([grid_x_norm, grid_y_norm], axis=-1)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype).unsqueeze(0)
    return F.grid_sample(image, grid_t, align_corners=True, padding_mode="border")
