#!/usr/bin/env python3
"""
Merge NeuMan official SMPL-X release into processed BodyAvatar NPZ files.

Download NeuMan from https://neuman.is.tue.mpg.de/ (registration required).
Extract so each sequence lives at:

  data/body/neuman_official/bike/
  data/body/neuman_official/citron/
  ...

Then run:

  python scripts/body/prepare_neuman_official.py --sequences bike citron jogging seattle
"""

from __future__ import annotations

import argparse
from pathlib import Path

from osa_body.neuman_smpl import merge_smpl_into_npz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-dir", default="data/body/neuman_official")
    p.add_argument("--processed-dir", default="data/body/processed")
    p.add_argument("--sequences", nargs="+", default=["bike", "citron", "jogging", "seattle"])
    args = p.parse_args()

    official = Path(args.official_dir)
    processed = Path(args.processed_dir)
    if not official.exists():
        raise SystemExit(
            f"Missing {official}. Download NeuMan official release and extract sequences there.\n"
            "See https://neuman.is.tue.mpg.de/"
        )

    merged = 0
    for seq in args.sequences:
        subject = f"neuman_{seq}"
        npz = processed / subject / "processed.npz"
        seq_dir = official / seq
        if not npz.exists():
            print(f"skip {subject}: no processed.npz (run prepare_neuman.py first)")
            continue
        if not seq_dir.exists():
            print(f"skip {seq}: no {seq_dir}")
            continue
        if merge_smpl_into_npz(npz, seq_dir):
            print(f"merged SMPL -> {npz}")
            merged += 1
        else:
            print(f"no SMPL found in {seq_dir} (install smplx + place SMPLX models in osa_body/assets/smplx/)")

    print(f"Done: {merged}/{len(args.sequences)} sequences updated")


if __name__ == "__main__":
    main()
