from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_clips(root: Path, clip_length: int = 8) -> list[dict]:
    clips = []
    for subject_dir in sorted(root.glob("*/processed.npz")):
        data = np.load(subject_dir)
        n = len(data["frames"])
        if n < clip_length:
            continue
        max_start = n - clip_length
        for start in range(0, max_start, clip_length // 2):
            clips.append(
                {
                    "subject": subject_dir.parent.name,
                    "path": str(subject_dir),
                    "start": int(start),
                    "clip_length": clip_length,
                }
            )
    return clips


def split_clips(
    clips: list[dict],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clips))
    rng.shuffle(idx)
    split_idx = int(len(idx) * train_ratio)
    train = [clips[i] for i in idx[:split_idx]] if split_idx > 0 else clips
    val = [clips[i] for i in idx[split_idx:]] if split_idx < len(idx) else clips[-max(1, len(clips) // 5):]
    return train, val


def split_clips_by_subject(
    clips: list[dict],
    val_subjects: list[str],
) -> tuple[list[dict], list[dict]]:
    """Hold out entire subject(s) for validation (recommended for real benchmarks)."""
    val_set = set(val_subjects)
    train = [c for c in clips if c["subject"] not in val_set]
    val = [c for c in clips if c["subject"] in val_set]
    if not val and clips:
        last = sorted({c["subject"] for c in clips})[-1]
        train = [c for c in clips if c["subject"] != last]
        val = [c for c in clips if c["subject"] == last]
    return train, val


def split_clips_temporal(
    clips: list[dict],
    train_ratio: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Split by clip start frame so val uses later motion (self-reenactment style)."""
    if not clips:
        return [], []
    by_subject: dict[str, list[dict]] = {}
    for c in clips:
        by_subject.setdefault(c["subject"], []).append(c)
    train, val = [], []
    for subj_clips in by_subject.values():
        subj_clips = sorted(subj_clips, key=lambda c: c["start"])
        split_idx = max(1, int(len(subj_clips) * train_ratio))
        if split_idx >= len(subj_clips):
            split_idx = len(subj_clips) - 1
        train.extend(subj_clips[:split_idx])
        val.extend(subj_clips[split_idx:])
    return train, val


def filter_clips(clips: list[dict], subjects: list[str] | None) -> list[dict]:
    if not subjects:
        return clips
    allow = set(subjects)
    return [c for c in clips if c["subject"] in allow]


def train_starts_by_subject(train_clips: list[dict]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for c in train_clips:
        out.setdefault(c["subject"], set()).add(c["start"])
    return out


def save_split(path: Path, train: list[dict], val: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": train, "val": val}, indent=2))


def load_split(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text())
    return data["train"], data["val"]
