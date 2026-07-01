"""Load NeuMan official SMPL-X parameters and project 2D joints for retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# SMPL-X body joint count used for 2D reprojection (22 body joints, no hands/face detail)
SMPL_BODY_JOINTS = 22


def _find_smpl_json(seq_dir: Path) -> Path | None:
    """NeuMan official layout: smpl/smpl_params.json or per-frame npz."""
    candidates = [
        seq_dir / "smpl" / "smpl_params.json",
        seq_dir / "smpl_params.json",
        seq_dir / "smplx_params" / "smpl_params.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    npz_dir = seq_dir / "smpl"
    if npz_dir.is_dir() and list(npz_dir.glob("*.npz")):
        return npz_dir  # type: ignore[return-value]
    return None


def load_smpl_2d_joints(seq_dir: Path, num_frames: int, image_size: int) -> np.ndarray | None:
    """
    Return (T, J, 3) normalized 2D joints in crop space if SMPL data exists.
    Falls back to None — caller uses MediaPipe landmarks only.
    """
    smpl_path = _find_smpl_json(seq_dir)
    if smpl_path is None:
        return None

    try:
        import torch
        from smplx import SMPLX
    except ImportError:
        return None

    device = torch.device("cpu")
    model_dir = Path(__file__).resolve().parent / "assets" / "smplx"
    if not (model_dir / "SMPLX_NEUTRAL.npz").exists() and not (model_dir / "smplx").exists():
        return None

    try:
        body = SMPLX(str(model_dir), num_betas=10, use_pca=False, flat_hand_mean=True)
    except Exception:
        return None

    # Minimal loader: if json list of per-frame params
    if smpl_path.suffix == ".json":
        params = json.loads(smpl_path.read_text())
        if not params:
            return None
        joints_list = []
        for i in range(min(num_frames, len(params))):
            frame_p = params[i]
            betas = torch.tensor(frame_p.get("betas", [0] * 10), dtype=torch.float32).unsqueeze(0)
            body_pose = torch.tensor(frame_p.get("body_pose", [0] * 63), dtype=torch.float32).unsqueeze(0)
            global_orient = torch.tensor(frame_p.get("global_orient", [0] * 3), dtype=torch.float32).unsqueeze(0)
            transl = torch.tensor(frame_p.get("transl", [0, 0, 0]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = body(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
            j3d = out.joints[0, :SMPL_BODY_JOINTS].numpy()
            # Orthographic projection (NeuMan cameras vary; placeholder scale to image)
            j2d = j3d[:, :2].copy()
            j2d[:, 0] = (j2d[:, 0] - j2d[:, 0].min()) / max(np.ptp(j2d[:, 0]), 1e-6) * (image_size - 1)
            j2d[:, 1] = (j2d[:, 1] - j2d[:, 1].min()) / max(np.ptp(j2d[:, 1]), 1e-6) * (image_size - 1)
            z = j3d[:, 2:3]
            joints_list.append(np.concatenate([j2d, z], axis=1).astype(np.float32))
        if not joints_list:
            return None
        return np.stack(joints_list)

    return None


def merge_smpl_into_npz(npz_path: Path, seq_dir: Path) -> bool:
    """Add smpl_joints key to processed.npz when official SMPL is available."""
    data = dict(np.load(npz_path))
    n = len(data["frames"])
    smpl_j = load_smpl_2d_joints(seq_dir, n, int(data.get("image_size", [384])[0]))
    if smpl_j is None:
        return False
    # Align length with frames (may differ if MP4 preprocess skipped frames)
    m = min(len(smpl_j), n)
    data["smpl_joints"] = smpl_j[:m]
    if m < n:
        pad = np.repeat(smpl_j[m - 1 : m], n - m, axis=0)
        data["smpl_joints"] = np.concatenate([smpl_j[:m], pad], axis=0)
    np.savez_compressed(npz_path, **data)
    return True
