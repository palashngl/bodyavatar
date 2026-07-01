# CVPR SOTA Roadmap

## Current status (BodyAvatar v2)

| Benchmark | PSNR | SSIM | LPIPS | vs in-repo baselines |
|-----------|------|------|-------|----------------------|
| Synthetic | 26.0 | 0.780 | 0.087 | **#1 combined** |
| NeuMan real | 13.5 | 0.886 | 0.092 | **4/4 subjects won** |

Published NeuMan SOTA (SFGS): ~35 PSNR with SMPL-X + 3DGS.

## v3 improvements (this repo)

- `BodyAvatarModelV3`: U-Net-lite refiner + multi-scale loss
- Lower retrieval anchor (0.06) for stronger neural correction
- `scripts/body/run_neuman_v3_benchmark.py`

## Path to CVPR numbers (~33+ PSNR NeuMan)

1. **Official NeuMan SMPL** — `prepare_neuman_official.py` + `smplx` models
2. **Retrieval 3DGS init** — `export_gs_init.py` → GauHuman/gsplat fine-tune
3. **Novel-view eval** — match SFGS / Vid2Avatar-Pro protocol exactly
4. **ZJU-MoCap** — second benchmark for generalization

## Commands

```bash
# v3 train on real NeuMan
python scripts/body/run_neuman_v3_benchmark.py --epochs 25 --gpu 0

# Merge official SMPL (after download from neuman.is.tue.mpg.de)
python scripts/body/prepare_neuman_official.py

# Export retrieval targets for 3DGS stage
python scripts/body/export_gs_init.py --subject neuman_bike
```
