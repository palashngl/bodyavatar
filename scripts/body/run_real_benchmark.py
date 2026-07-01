#!/usr/bin/env python3
"""End-to-end: reprocess NeuMan, train BodyAvatar v2, compare vs baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", default=["bike", "citron", "jogging", "seattle"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-preprocess", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    seqs = " ".join(args.sequences)

    if not args.skip_preprocess:
        run(
            [
                py,
                str(root / "scripts/body/prepare_neuman.py"),
                "--sequences",
                *args.sequences,
                "--max-frames",
                "360",
            ]
        )

    run(
        [
            py,
            str(root / "scripts/body/run_neuman_benchmark.py"),
            "--sequences",
            *args.sequences,
            "--epochs",
            str(args.epochs),
            "--gpu",
            str(args.gpu),
            "--output",
            "runs/body_neuman_v2",
        ]
    )

    run(
        [
            py,
            str(root / "scripts/body/compare_neuman.py"),
            "--sequences",
            *args.sequences,
            "--checkpoint-dir",
            "runs/body_neuman_v2",
            "--split-dir",
            "runs/body_neuman_v2/splits",
            "--gpu",
            str(args.gpu),
        ]
    )


if __name__ == "__main__":
    main()
