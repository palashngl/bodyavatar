from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay


def _to_xy(landmarks: torch.Tensor) -> torch.Tensor:
    return landmarks[..., :2]


def procrustes_warp_image(
    image: torch.Tensor,
    src_lmk: torch.Tensor,
    dst_lmk: torch.Tensor,
) -> torch.Tensor:
    """Similarity transform aligning src landmarks to dst (inverse sampling)."""
    _, _, h, w = image.shape
    device, dtype = image.device, image.dtype
    src = _to_xy(src_lmk)[0].detach().cpu().numpy()
    dst = _to_xy(dst_lmk)[0].detach().cpu().numpy()
    src_pts = procrustes_src_points(src, dst, _pixel_grid(h, w))
    grid = _norm_grid(src_pts, h, w)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype).unsqueeze(0)
    return F.grid_sample(image, grid_t, align_corners=True, padding_mode="border")


def _pixel_grid(h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, h - 1, h, dtype=np.float64)
    xs = np.linspace(0, w - 1, w, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def _norm_grid(src_pts: np.ndarray, h: int, w: int) -> np.ndarray:
    grid_x_norm = (src_pts[:, 0].reshape(h, w) / (w - 1)) * 2 - 1
    grid_y_norm = (src_pts[:, 1].reshape(h, w) / (h - 1)) * 2 - 1
    return np.stack([grid_x_norm, grid_y_norm], axis=-1).astype(np.float32)


def _barycentric_single(tri: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric weights for points inside one triangle tri (3,2), pts (N,2)."""
    a, b, c = tri[0], tri[1], tri[2]
    v0 = b - a
    v1 = c - a
    v2 = pts - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = (v2 * v0).sum(axis=-1)
    d21 = (v2 * v1).sum(axis=-1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def piecewise_affine_warp_image(
    image: torch.Tensor,
    src_lmk: torch.Tensor,
    dst_lmk: torch.Tensor,
    faces: np.ndarray | None = None,
) -> torch.Tensor:
    """Delaunay piecewise-affine warp (stronger than global affine / TPS on faces)."""
    del faces
    _, _, h, w = image.shape
    device, dtype = image.device, image.dtype
    src = _to_xy(src_lmk)[0].detach().cpu().numpy().astype(np.float64)
    dst = _to_xy(dst_lmk)[0].detach().cpu().numpy().astype(np.float64)

    delaunay = Delaunay(dst)
    simplices = delaunay.simplices
    dst_tri = dst[simplices]
    src_tri = src[simplices]
    pts = _pixel_grid(h, w)

    simplex_idx = delaunay.find_simplex(pts)
    src_pts = np.zeros_like(pts)

    for si in range(len(simplices)):
        mask = simplex_idx == si
        if not mask.any():
            continue
        dt = dst_tri[si]
        st = src_tri[si]
        u, v, w_b = _barycentric_single(dt, pts[mask])
        src_pts[mask] = (
            u[:, None] * st[0] + v[:, None] * st[1] + w_b[:, None] * st[2]
        )

    outside = simplex_idx < 0
    if outside.any():
        src_pts[outside] = procrustes_src_points(src, dst, pts[outside])

    grid = _norm_grid(src_pts, h, w)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=dtype).unsqueeze(0)
    return F.grid_sample(image, grid_t, align_corners=True, padding_mode="border")


def procrustes_src_points(
    src: np.ndarray, dst: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    """Map dst-space points to src-space via similarity transform."""
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)
    src_c = src - src_ctr
    dst_c = dst - dst_ctr
    num = (dst_c * src_c).sum()
    den = (src_c**2).sum() + 1e-8
    scale = float(np.sqrt((dst_c**2).sum() / den))
    angle = float(np.arctan2(
        (dst_c[:, 0] * src_c[:, 1] - dst_c[:, 1] * src_c[:, 0]).sum(),
        num,
    ))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    inv_scale = 1.0 / max(scale, 1e-6)
    centered = pts - dst_ctr
    return (centered @ rot.T) * inv_scale + src_ctr
