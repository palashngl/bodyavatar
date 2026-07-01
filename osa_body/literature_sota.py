"""Published full-body / human avatar results (2025–2026 focus).

Numbers are copied from original papers — datasets and LPIPS scales differ.
Use `normalize_lpips()` before cross-paper numeric comparison.
"""

from __future__ import annotations

from typing import Any

# LPIPS reporting conventions vary by paper.
LPIPS_SCALE = {
    "standard": 1.0,
    "x100": 100.0,
    "x1000": 1000.0,
}


def normalize_lpips(value: float, scale: str) -> float:
    return value / LPIPS_SCALE[scale]


def normalize_ssim(value: float, scale: str = "standard") -> float:
    if scale == "x100":
        return value / 100.0
    return value


# BodyAvatar (ours) — same synthetic benchmark as scripts/body/compare_sota.py
BODYAVATAR_OURS: dict[str, Any] = {
    "method": "BodyAvatar (ours)",
    "year": 2026,
    "venue": "OSA repo",
    "dataset": "3 synthetic full-body subjects (147 train / 27 val)",
    "task": "self-reenactment (2D landmark-driven)",
    "input": "monocular video + MediaPipe pose landmarks",
    "representation": "KNN retrieval + Procrustes + neural residual",
    "psnr": 26.00,
    "ssim": 0.780,
    "lpips": 0.087,
    "lpips_scale": "standard",
    "comparable": True,
    "note": "Unified in-repo val split; not ZJU/NeuMan/WildAvatar.",
    "paper_url": None,
}

# ≥12 methods from 2025–2026 literature (reference benchmarks).
SOTA_LITERATURE_2025_2026: list[dict[str, Any]] = [
    {
        "method": "SFGS",
        "year": 2026,
        "venue": "arXiv",
        "dataset": "NeuMan (bike,citron,jogging,seattle)",
        "task": "monocular expressive avatar / novel view",
        "input": "monocular RGB + SMPL-X video",
        "representation": "SMPL-X + fine-grained 3DGS + HexPlane",
        "psnr": 35.34,
        "ssim": 0.985,
        "lpips": 0.009,
        "lpips_scale": "standard",
        "paper_url": "https://arxiv.org/html/2604.09324v1",
        "note": "Table 1 mean; foreground segmented.",
    },
    {
        "method": "HiAvatar",
        "year": 2026,
        "venue": "ICCV / ICCVM",
        "dataset": "ZJU-MoCap (6 subjects, novel view avg)",
        "task": "monocular avatar reconstruction",
        "input": "monocular video",
        "representation": "3DGS + spatial/temporal enhancement",
        "psnr": 31.71,
        "ssim": 0.984,
        "ssim_scale": "standard",
        "lpips": 29.39,
        "lpips_scale": "x1000",
        "paper_url": "https://iccvm.org/2026/files/papers/128.pdf",
        "note": "Avg PSNR/SSIM/LPIPS over subjects 377,386,387,392,393,394; LPIPS×1000.",
    },
    {
        "method": "Vid2Avatar-Pro",
        "year": 2025,
        "venue": "CVPR",
        "dataset": "NeuMan",
        "task": "interpolation synthesis",
        "input": "monocular in-the-wild video",
        "representation": "Universal prior + expressive 3D Gaussians",
        "psnr": 32.71,
        "ssim": 0.983,
        "lpips": 1.19,
        "lpips_scale": "x100",
        "paper_url": "https://arxiv.org/html/2503.01610v1",
        "note": "Table 1; LPIPS×100.",
    },
    {
        "method": "R³-Avatar",
        "year": 2025,
        "venue": "arXiv",
        "dataset": "ZJU-MoCap",
        "task": "novel view rendering",
        "input": "multi-view / video",
        "representation": "Temporal codebook + 3DGS",
        "psnr": 34.64,
        "ssim": 0.976,
        "lpips": 22.5,
        "lpips_scale": "x1000",
        "paper_url": "https://arxiv.org/html/2503.12751v1",
        "note": "Table 1 ZJU column; LPIPS×1000.",
    },
    {
        "method": "MPMAvatar",
        "year": 2025,
        "venue": "NeurIPS",
        "dataset": "ActorsHQ",
        "task": "physics-based animation + rendering",
        "input": "multi-view video",
        "representation": "MPM dynamics + 3DGS",
        "psnr": 32.0,
        "ssim": 0.963,
        "lpips": 0.033,
        "lpips_scale": "standard",
        "paper_url": "https://arxiv.org/html/2510.01619v1",
        "note": "Table 1 ActorsHQ appearance metrics.",
    },
    {
        "method": "EVA",
        "year": 2025,
        "venue": "SIGGRAPH",
        "dataset": "Multi-view studio capture (4 subjects, 2K)",
        "task": "novel view + novel pose/expression",
        "input": "multi-view video",
        "representation": "Disentangled mesh + 3D Gaussians (body/face/hands)",
        "psnr": 46.00,
        "ssim": 98.79,
        "ssim_scale": "x100",
        "lpips": 12.55,
        "lpips_scale": "x100",
        "paper_url": "https://arxiv.org/html/2505.15385",
        "note": "Table 1 novel pose/expr column; 2K resolution; SSIM/LPIPS×100 in paper.",
    },
    {
        "method": "LSA (Locality Sensitive Avatars)",
        "year": 2025,
        "venue": "ICLR",
        "dataset": "ZJU-MoCap (8 subjects, novel view avg)",
        "task": "monocular avatar",
        "input": "monocular video",
        "representation": "Locality-sensitive implicit + 3DGS",
        "psnr": 30.34,
        "ssim": 0.978,
        "lpips": 25.58,
        "lpips_scale": "x1000",
        "paper_url": "https://www.cs.ubc.ca/~lsigal/Publications/song2025iclr.pdf",
        "note": "Appendix Tab. A average; LPIPS×1000.",
    },
    {
        "method": "HumanRAM",
        "year": 2025,
        "venue": "SIGGRAPH",
        "dataset": "THuman2.1",
        "task": "feed-forward reconstruction (4 views)",
        "input": "sparse multi-view + SMPL-X",
        "representation": "Transformer LRM + SMPL-X neural texture",
        "psnr": 30.34,
        "ssim": 0.9535,
        "lpips": 0.0184,
        "lpips_scale": "standard",
        "paper_url": "https://arxiv.org/html/2506.03118v1",
        "note": "Table 1 reconstruction on unseen subjects.",
    },
    {
        "method": "HumanRAM",
        "year": 2025,
        "venue": "SIGGRAPH",
        "dataset": "ZJU-MoCap",
        "task": "feed-forward animation (4 views)",
        "input": "sparse multi-view + SMPL-X",
        "representation": "Transformer LRM + SMPL-X neural texture",
        "psnr": 23.40,
        "ssim": 0.9529,
        "lpips": 0.0252,
        "lpips_scale": "standard",
        "paper_url": "https://arxiv.org/html/2506.03118v1",
        "note": "Table 3 multi-view animation on unseen subjects.",
    },
    {
        "method": "HumanGS",
        "year": 2026,
        "venue": "arXiv",
        "dataset": "THuman2.1 → AvatarReX (4 views)",
        "task": "cross-dataset reconstruction + animation",
        "input": "sparse multi-view + SMPL-X",
        "representation": "Feed-forward canonical 3DGS on SMPL-X",
        "psnr": 25.24,
        "ssim": 0.96,
        "lpips": 0.04,
        "lpips_scale": "standard",
        "paper_url": "https://arxiv.org/html/2604.10259v1",
        "note": "Table 1; ~0.96s modeling vs hours for optimization methods.",
    },
    {
        "method": "HumanNOVA",
        "year": 2026,
        "venue": "arXiv",
        "dataset": "Custom 100k assets benchmark",
        "task": "single-image → 3D avatar",
        "input": "single RGB image",
        "representation": "Feed-forward generative 3D human",
        "psnr": 22.07,
        "ssim": 0.9344,
        "lpips": 45.18,
        "lpips_scale": "x1000",
        "paper_url": "https://arxiv.org/html/2606.02573v1",
        "note": "Table 1; 512×512 render; not video-to-avatar.",
    },
    {
        "method": "GauHuman (WildAvatar eval)",
        "year": 2025,
        "venue": "CVPR (WildAvatar benchmark)",
        "dataset": "WildAvatar (133 subjects, novel pose)",
        "task": "per-subject monocular avatar",
        "input": "in-the-wild monocular video",
        "representation": "3DGS + SMPL LBS",
        "psnr": 25.89,
        "ssim": 95.7,
        "ssim_scale": "x100",
        "lpips": 5.7,
        "lpips_scale": "x100",
        "paper_url": "https://arxiv.org/html/2407.02165v4",
        "note": "WildAvatar Tab. 4 with refined annotations; SSIM/LPIPS×100.",
    },
    {
        "method": "Vid2Avatar-Pro",
        "year": 2025,
        "venue": "CVPR",
        "dataset": "MonoPerfCap",
        "task": "extrapolation synthesis",
        "input": "monocular in-the-wild video",
        "representation": "Universal prior + expressive 3D Gaussians",
        "psnr": 31.97,
        "ssim": 0.981,
        "lpips": 1.37,
        "lpips_scale": "x100",
        "paper_url": "https://arxiv.org/html/2503.01610v1",
        "note": "Table 2 extrapolation; LPIPS×100.",
    },
]

# Legacy entries kept for backward compatibility with compare_sota.py
SOTA_LITERATURE_LEGACY: list[dict[str, Any]] = [
    {
        "method": "GauHuman (CVPR 2024)",
        "year": 2024,
        "dataset": "ZJU-MoCap / MonoCap",
        "typical_psnr": "28-32",
        "category": "3D Gaussian + SMPL",
    },
    {
        "method": "InstantAvatar (CVPR 2023)",
        "year": 2023,
        "dataset": "PeopleSnapshot / ZJU",
        "typical_psnr": "27-31",
        "category": "NeRF + parametric body",
    },
    {
        "method": "NeuralBody (CVPR 2021)",
        "year": 2021,
        "dataset": "ZJU-MoCap",
        "typical_psnr": "31+",
        "category": "Neural implicit + mesh",
    },
    {
        "method": "HumanNeRF (CVPR 2022)",
        "year": 2022,
        "dataset": "ZJU-MoCap",
        "typical_psnr": "30+",
        "category": "NeRF warping",
    },
]


def literature_row(entry: dict[str, Any]) -> dict[str, Any]:
    """Add normalized metrics for side-by-side display."""
    row = dict(entry)
    scale = entry.get("lpips_scale", "standard")
    if "lpips" in entry:
        row["lpips_normalized"] = round(normalize_lpips(entry["lpips"], scale), 4)
    ssim_scale = entry.get("ssim_scale", "standard")
    if "ssim" in entry:
        row["ssim_normalized"] = round(normalize_ssim(entry["ssim"], ssim_scale), 4)
    return row


def build_literature_comparison(body_metrics: dict[str, float] | None = None) -> dict[str, Any]:
    """Build structured comparison: ours + 2025/2026 papers."""
    ours = dict(BODYAVATAR_OURS)
    if body_metrics:
        ours["psnr"] = round(body_metrics.get("psnr", ours["psnr"]), 2)
        ours["ssim"] = round(body_metrics.get("ssim", ours["ssim"]), 4)
        ours["lpips"] = round(body_metrics.get("lpips", ours["lpips"]), 4)
        ours["lpips_normalized"] = ours["lpips"]

    papers = [literature_row(p) for p in SOTA_LITERATURE_2025_2026]
    by_psnr = sorted(papers, key=lambda p: p.get("psnr", 0), reverse=True)

    return {
        "ours": ours,
        "papers_2025_2026": papers,
        "ranked_by_reported_psnr": [p["method"] for p in by_psnr],
        "disclaimer": (
            "Literature rows use different datasets, resolutions, foreground masks, and LPIPS scales. "
            "Only BodyAvatar numbers share the same val protocol. "
            "Use lpips_normalized for rough cross-paper ordering, not strict ranking."
        ),
        "lpips_scale_legend": LPIPS_SCALE,
    }
