#!/usr/bin/env python3
"""
End-to-end best-results pipeline:
  1) v3 synthetic pretrain
  2) v3 NeuMan finetune (all sequences)
  3) baseline comparison
  4) inference FPS benchmark on seattle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], env: dict | None = None) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def bench_fps(checkpoint: Path, processed: Path, split_file: Path, subject: str, gpu: int) -> dict:
    from osa_body.model_v3 import BodyAvatarModelV3
    from osa_body.retrieval_bank import BodyTrainBank

    device = torch.device(f"cuda:{gpu}")
    data = np.load(processed / subject / "processed.npz")
    n = min(64, len(data["frames"]))
    frames = torch.from_numpy(data["frames"][:n]).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    lmks = torch.from_numpy(data["landmarks"][:n]).float().unsqueeze(0).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = BodyAvatarModelV3(knn_k=5).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bank = BodyTrainBank(processed, split_file, warp="piecewise")
    bank.preload([subject])

    # warmup
    with torch.inference_mode():
        for _ in range(3):
            model(frames=frames, landmarks=lmks, retrieval_bank=bank, subjects=[subject], exclude_frame_indices=[set(range(n))])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(10):
            model(frames=frames, landmarks=lmks, retrieval_bank=bank, subjects=[subject], exclude_frame_indices=[set(range(n))])
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 10
    fps = n / elapsed
    return {"frames": n, "sec_per_clip": elapsed, "effective_fps": fps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--synthetic-epochs", type=int, default=40)
    p.add_argument("--neuman-epochs", type=int, default=25)
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--skip-neuman", action="store_true")
    p.add_argument("--output", default="runs/body_best")
    args = p.parse_args()

    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT), "TORCH_COMPILE_DISABLE": "1", f"CUDA_VISIBLE_DEVICES": str(args.gpu)}
    py = sys.executable
    out = Path(args.output)
    syn_ckpt = out / "synthetic_pretrain" / "best.pt"
    neuman_out = out / "neuman_v3"

    if not args.skip_synthetic:
        run([py, "scripts/body/train_v3.py", "--epochs", str(args.synthetic_epochs), "--output", str(out / "synthetic_pretrain"), "--gpu", "0"], env)

    init = str(syn_ckpt) if syn_ckpt.exists() else ""
    if not args.skip_neuman:
        cmd = [py, "scripts/body/run_neuman_v3_benchmark.py", "--epochs", str(args.neuman_epochs), "--output", str(neuman_out), "--gpu", "0"]
        if init:
            cmd += ["--init-checkpoint", init]
        run(cmd, env)

    run([py, "scripts/body/compare_neuman.py", "--checkpoint-dir", str(neuman_out), "--split-dir", str(neuman_out / "splits"), "--model-version", "v3", "--output", str(neuman_out / "comparison.json"), "--gpu", "0"], env)

    split = neuman_out / "splits/neuman_seattle.json"
    ckpt = neuman_out / "neuman_seattle/best.pt"
    fps_stats = {}
    if ckpt.exists() and split.exists():
        fps_stats = bench_fps(ckpt, Path("data/body/processed"), split, "neuman_seattle", args.gpu)

    summary = {
        "synthetic_ckpt": str(syn_ckpt),
        "neuman_dir": str(neuman_out),
        "comparison": str(neuman_out / "comparison.json"),
        "inference_fps_seattle": fps_stats,
    }
    if (neuman_out / "comparison.json").exists():
        summary["comparison_data"] = json.loads((neuman_out / "comparison.json").read_text())

    (out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Pipeline complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
