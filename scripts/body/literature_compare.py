#!/usr/bin/env python3
"""Print / save 2025–2026 literature comparison vs BodyAvatar (no GPU eval)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osa_body.literature_sota import BODYAVATAR_OURS, build_literature_comparison


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-metrics", default="runs/body_sota_comparison.json")
    p.add_argument("--output", default="runs/body_literature_comparison.json")
    args = p.parse_args()

    body = dict(BODYAVATAR_OURS)
    src = Path(args.body_metrics)
    if src.exists():
        prev = json.loads(src.read_text())
        m = prev.get("methods", {}).get("BodyAvatar")
        if m:
            body = {k: m[k] for k in ("psnr", "ssim", "lpips") if k in m}

    lit = build_literature_comparison(body)
    Path(args.output).write_text(json.dumps(lit, indent=2))

    print("=== BodyAvatar vs 2025–2026 literature (reference only) ===\n")
    print(f"{'Method':<28} {'Year':>4}  {'PSNR':>7}  {'SSIM':>7}  {'LPIPS*':>8}  Dataset / task")
    print("-" * 95)
    o = lit["ours"]
    print(
        f"{o['method']:<28} {o['year']:>4}  {o['psnr']:7.2f}  {o['ssim']:7.3f}  "
        f"{o['lpips']:8.4f}  {o['dataset']}"
    )
    for row in sorted(lit["papers_2025_2026"], key=lambda r: r.get("psnr", 0), reverse=True):
        lp = row.get("lpips_normalized", row.get("lpips"))
        print(
            f"{row['method']:<28} {row['year']:>4}  {row.get('psnr', 0):7.2f}  "
            f"{row.get('ssim_normalized', row.get('ssim', 0)):7.3f}  {lp:8.4f}  "
            f"{row.get('dataset', '')[:50]}"
        )
    print(f"\n* LPIPS* = normalized to standard scale.\n{lit['disclaimer']}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
