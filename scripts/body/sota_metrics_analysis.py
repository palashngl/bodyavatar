#!/usr/bin/env python3
"""Multi-metric SOTA analysis: in-repo baselines + literature reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from osa_body.literature_sota import (
    BODYAVATAR_OURS,
    SOTA_LITERATURE_2025_2026,
    build_literature_comparison,
    literature_row,
    normalize_lpips,
    normalize_ssim,
)


def combined(m: dict[str, float]) -> float:
    return m["psnr"] + 5.0 * m["ssim"] - 15.0 * m["lpips"]


def rank_methods(methods: dict[str, dict], key: str, higher_better: bool = True) -> list[str]:
    items = [(name, methods[name].get(key, 0)) for name in methods]
    return [n for n, _ in sorted(items, key=lambda x: x[1], reverse=higher_better)]


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def neuman_mean(comparison: dict) -> dict[str, float]:
    psnrs, ssims, lpipss = [], [], []
    for subj in comparison.values():
        m = subj["methods"]["BodyAvatar"]
        psnrs.append(m["psnr"])
        ssims.append(m["ssim"])
        lpipss.append(m["lpips"])
    n = len(psnrs)
    mean = {
        "psnr": sum(psnrs) / n,
        "ssim": sum(ssims) / n,
        "lpips": sum(lpipss) / n,
    }
    mean["combined"] = combined(mean)
    return mean


def literature_table(ours_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in ours_rows:
        row = literature_row(entry)
        row["combined"] = round(
            row["psnr"] + 5 * row.get("ssim_normalized", row.get("ssim", 0))
            - 15 * row.get("lpips_normalized", row.get("lpips", 0)),
            3,
        )
        row["paradigm"] = entry.get("representation", "")
        rows.append(row)
    return rows


def build_analysis(project_root: Path) -> dict[str, Any]:
    synthetic = load_json(project_root / "runs/body_sota_comparison.json")
    neuman_cmp = load_json(project_root / "runs/body_neuman_v2/comparison.json")
    neuman_bench = load_json(project_root / "runs/body_neuman_v2/neuman_benchmark.json")
    face_v4 = load_json(project_root / "runs/comparison_v4.json")

    # --- Tier 1: in-repo body synthetic ---
    syn_methods = synthetic["methods"] if synthetic else {}
    syn_rankings = {
        "psnr": rank_methods(syn_methods, "psnr"),
        "ssim": rank_methods(syn_methods, "ssim"),
        "lpips": rank_methods(syn_methods, "lpips", higher_better=False),
        "combined": rank_methods(syn_methods, "combined"),
    }
    syn_body = {k: syn_methods[k] for k in syn_methods if k == "BodyAvatar"}
    syn_body["BodyAvatar"]["rank"] = {
        metric: syn_rankings[metric].index("BodyAvatar") + 1 for metric in syn_rankings
    }

    # --- Tier 1b: in-repo real NeuMan ---
    neuman_per_subject = {}
    neuman_baseline_wins = 0
    if neuman_cmp:
        for subj, data in neuman_cmp.items():
            methods = data["methods"]
            ranks = {
                "psnr": rank_methods(methods, "psnr"),
                "ssim": rank_methods(methods, "ssim"),
                "lpips": rank_methods(methods, "lpips", higher_better=False),
                "combined": rank_methods(methods, "combined"),
            }
            neuman_per_subject[subj] = {
                "bodyavatar": methods["BodyAvatar"],
                "best_baseline": data["ranking"][1] if data["ranking"][0] == "BodyAvatar" else data["ranking"][0],
                "rank": {m: ranks[m].index("BodyAvatar") + 1 for m in ranks},
                "winner": data["best"],
            }
            if data["best"] != "BodyAvatar":
                neuman_baseline_wins += 1

    neuman_mean_metrics = neuman_mean(neuman_cmp) if neuman_cmp else {}

    # --- Tier 2: literature reference (3 BodyAvatar variants) ---
    body_synthetic = {
        **BODYAVATAR_OURS,
        "psnr": syn_methods.get("BodyAvatar", {}).get("psnr", 26.0),
        "ssim": syn_methods.get("BodyAvatar", {}).get("ssim", 0.78),
        "lpips": syn_methods.get("BodyAvatar", {}).get("lpips", 0.087),
        "benchmark": "synthetic_in_repo",
    }
    body_real = {
        "method": "BodyAvatar (ours, real NeuMan)",
        "year": 2026,
        "dataset": "NeuMan (bike,citron,jogging,seattle)",
        "task": "self-reenactment masked",
        "input": "monocular video + MediaPipe",
        "representation": "KNN piecewise + neural residual",
        "psnr": neuman_mean_metrics.get("psnr", 13.54),
        "ssim": neuman_mean_metrics.get("ssim", 0.886),
        "lpips": neuman_mean_metrics.get("lpips", 0.092),
        "lpips_scale": "standard",
        "benchmark": "real_neuman_in_repo",
    }

    ours_variants = literature_table([body_synthetic, body_real])
    lit_rows = literature_table(SOTA_LITERATURE_2025_2026)
    all_lit = ours_variants + lit_rows

    for metric, higher in [("psnr", True), ("ssim_normalized", True), ("lpips_normalized", False), ("combined", True)]:
        ranked = sorted(all_lit, key=lambda r: r.get(metric, 0), reverse=higher)
        for i, row in enumerate(ranked):
            row.setdefault("ranks", {})[metric] = i + 1

    # NeuMan-only literature subset
    neuman_lit = [r for r in lit_rows if "NeuMan" in r.get("dataset", "")]
    neuman_lit_with_ours = [body_real] + neuman_lit

    # --- Face OSA v4 ---
    face_metrics = None
    if face_v4 and "OSA_v4" in face_v4.get("methods", {}):
        face_metrics = face_v4["methods"]["OSA_v4"]
        face_metrics["combined"] = combined(face_metrics)

    # --- Metric deltas vs best baseline (synthetic) ---
    best_bl = "KNNBodyBlendWarp"
    deltas = {}
    if synthetic and "BodyAvatar" in syn_methods and best_bl in syn_methods:
        for k in ("psnr", "ssim", "lpips", "combined"):
            deltas[k] = syn_methods["BodyAvatar"][k] - syn_methods[best_bl][k]

    return {
        "generated_by": "scripts/body/sota_metrics_analysis.py",
        "metric_definitions": {
            "psnr": "Peak signal-to-noise ratio (dB, higher better)",
            "ssim": "Structural similarity [0,1], higher better",
            "lpips": "Learned perceptual distance [0,1], lower better",
            "combined": "psnr + 5*ssim - 15*lpips (in-repo ranking score)",
            "lpips_normalized": "LPIPS rescaled to standard AlexNet scale across papers",
        },
        "tier1_synthetic_in_repo": {
            "comparability": "apples-to-apples — same val split, same metrics",
            "methods": {k: {m: syn_methods[k][m] for m in ("psnr", "ssim", "lpips", "combined") if m in syn_methods[k]}
                        for k in syn_methods},
            "rankings": syn_rankings,
            "bodyavatar_rank": syn_body.get("BodyAvatar", {}).get("rank"),
            "bodyavatar_vs_best_baseline": {"baseline": best_bl, "delta": deltas},
            "winner_combined": syn_rankings["combined"][0] if syn_rankings["combined"] else None,
        },
        "tier1_real_neuman_in_repo": {
            "comparability": "apples-to-apples — masked metrics, per-subject train/val",
            "mean": neuman_mean_metrics,
            "per_subject": neuman_per_subject,
            "subjects_won_by_bodyavatar": sum(
                1 for s in neuman_per_subject.values() if s["winner"] == "BodyAvatar"
            ),
            "total_subjects": len(neuman_per_subject),
        },
        "tier2_literature_reference": {
            "comparability": "reference only — different datasets, tasks, LPIPS scales",
            "disclaimer": build_literature_comparison()["disclaimer"],
            "ours_variants": ours_variants,
            "all_methods_ranked_by_psnr": sorted(all_lit, key=lambda r: r["psnr"], reverse=True),
            "all_methods_ranked_by_combined": sorted(all_lit, key=lambda r: r["combined"], reverse=True),
            "neuman_dataset_subset": neuman_lit_with_ours,
        },
        "face_osa_v4_synthetic": {
            "comparability": "face-only 256x256 synthetic benchmark",
            "metrics": face_metrics,
        },
        "summary": {
            "wins_in_repo_synthetic_combined": syn_rankings["combined"][0] == "BodyAvatar" if syn_rankings.get("combined") else False,
            "wins_all_neuman_subjects": len(neuman_per_subject) > 0 and neuman_baseline_wins == 0,
            "bodyavatar_synthetic_psnr_rank": syn_body.get("BodyAvatar", {}).get("rank", {}).get("psnr"),
            "bodyavatar_synthetic_lpips_rank": syn_body.get("BodyAvatar", {}).get("rank", {}).get("lpips"),
            "literature_psnr_rank_among_all": next(
                (r["ranks"]["psnr"] for r in ours_variants if r["method"].startswith("BodyAvatar") and "real" not in r["method"].lower()),
                None,
            ),
            "gap_to_sfgs_neuman_psnr": round(
                next(r["psnr"] for r in lit_rows if r["method"] == "SFGS") - body_real["psnr"], 2
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", default="runs/sota_metrics_analysis.json")
    args = p.parse_args()
    root = Path(args.root)
    report = build_analysis(root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    s = report["summary"]
    t1 = report["tier1_synthetic_in_repo"]
    t1n = report["tier1_real_neuman_in_repo"]
    print("=== Multi-metric SOTA analysis ===\n")
    print("TIER 1 — In-repo synthetic (apples-to-apples)")
    print(f"  Combined rank #1: {t1['winner_combined']}")
    ba = t1["bodyavatar_rank"]
    if ba:
        print(f"  BodyAvatar ranks: PSNR #{ba['psnr']}  SSIM #{ba['ssim']}  LPIPS #{ba['lpips']}  Combined #{ba['combined']}")
    print(f"\nTIER 1 — Real NeuMan (apples-to-apples)")
    print(f"  BodyAvatar wins {t1n['subjects_won_by_bodyavatar']}/{t1n['total_subjects']} subjects")
    m = t1n["mean"]
    print(f"  Mean: PSNR={m.get('psnr',0):.2f}  SSIM={m.get('ssim',0):.3f}  LPIPS={m.get('lpips',0):.3f}  Combined={m.get('combined',0):.2f}")
    print(f"\nTIER 2 — Literature reference (NeuMan PSNR gap to SFGS: {s['gap_to_sfgs_neuman_psnr']} dB)")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
