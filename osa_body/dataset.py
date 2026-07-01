from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from osa.data.splits import build_clips, load_split, save_split, split_clips


class BodyVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        clip_length: int = 8,
        split: str = "train",
        split_file: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        split_path = Path(split_file or self.data_root.parent / "splits" / "bench_split.json")
        if split_path.exists():
            train, val = load_split(split_path)
        else:
            clips = build_clips(self.data_root)
            train, val = split_clips(clips)
            save_split(split_path, train, val)
        self.clips = train if split == "train" else val

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        meta = self.clips[index]
        data = np.load(meta["path"])
        start = meta["start"]
        end = start + meta["clip_length"]
        frames = data["frames"][start:end]
        landmarks = data["landmarks"][start:end]
        item = {
            "frames": torch.from_numpy(frames).permute(0, 3, 1, 2).float(),
            "landmarks": torch.from_numpy(landmarks).float(),
            "neutral_landmarks": torch.from_numpy(data["neutral_landmarks"]).float(),
            "subject": meta["subject"],
            "frame_indices": list(range(start, end)),
        }
        if "masks" in data:
            item["masks"] = torch.from_numpy(data["masks"][start:end]).float()
        return item


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        if key in ("subject",):
            out[key] = [b[key] for b in batch]
        elif key == "frame_indices":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
