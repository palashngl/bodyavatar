"""Resolve data directories and default checkpoints across installs."""

from __future__ import annotations

from pathlib import Path


def default_processed_dir() -> Path:
    candidates = [
        Path("data/body/processed"),
        Path(__file__).resolve().parents[1] / "data/body/processed",
        Path("/home/server/Projects/OSA/data/body/processed"),
        Path("/home/server/Projects/bodyavatar/data/body/processed"),
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("*/processed.npz")):
            return p
    return Path("data/body/processed")


def default_bench_split() -> Path:
    processed = default_processed_dir()
    candidates = [
        processed.parent / "splits" / "bench_split.json",
        Path("data/body/splits/bench_split.json"),
        Path(__file__).resolve().parents[1] / "data/body/splits/bench_split.json",
        Path("/home/server/Projects/OSA/data/body/splits/bench_split.json"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return processed.parent / "splits" / "bench_split.json"


def default_pretrained_checkpoint() -> Path | None:
    """Best available generic v4 init (synthetic pretrain preferred)."""
    roots = [
        Path(__file__).resolve().parents[1],
        Path("/home/server/Projects/bodyavatar"),
        Path("/home/server/Projects/OSA"),
    ]
    rel_paths = [
        "runs/body_best_v4/synthetic_pretrain/best.pt",
        "runs/body_v4_synthetic/best.pt",
        "runs/body_neuman_v4/neuman_seattle/best.pt",
        "runs/body_neuman_v4_test/neuman_seattle/best.pt",
    ]
    for root in roots:
        for rel in rel_paths:
            p = root / rel
            if p.is_file():
                return p
    return None
