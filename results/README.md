# Results

This directory holds all output from training and evaluation.

## Directory layout

```
results/
  figures/          Key figures exported from the notebook (committed)
  checkpoints/      Training checkpoints — gitignored (large .pt files)
  _smoke_run/       Smoke-test artefacts — gitignored
```

## Figures (committed)

Add exported PNG/PDF figures here from the notebook before submission.
Expected files:

| File | Description |
|------|-------------|
| `figures/degradation_progression.png` | Forward D(x,t) for inpainting / blur / SR |
| `figures/reconstruction_inpainting.png` | Degraded / Direct / Alg 2 / Original rows |
| `figures/sampler_comparison.png` | Alg 1 vs Alg 2 vs Alg 3 visual comparison |
| `figures/alg3_alpha_sweep.png` | FID vs alpha for Algorithm 3 (inpainting) |
| `figures/cold_generation.png` | Color seed → generated samples (§5.3) |
| `figures/mnist_experiments.png` | MNIST inpainting / blur reconstructions |

## Quantitative metrics

### CIFAR-10 Inpainting (T = 50, 60k steps)

| Method | FID ↓ | SSIM ↑ | RMSE ↓ |
|--------|------:|-------:|-------:|
| **Paper** — Degraded | 40.83 | 0.615 | 0.143 |
| **Paper** — Direct | 9.71 | 0.869 | 0.063 |
| **Paper** — Alg 2 sampled | 8.92 | 0.859 | 0.068 |
| **Ours** — Degraded | 70.73 | 0.579 | 0.302 |
| **Ours** — Direct | 15.21 | 0.877 | 0.071 |
| **Ours** — Alg 1 (naive) | 123.17 | 0.440 | 0.248 |
| **Ours** — Alg 2 sampled | 12.16 | 0.807 | 0.081 |
| **Ours** — Alg 3 (α=0.3) | **11.89** | **0.872** | **0.080** |

### CIFAR-10 Blur (T = 40, 30k steps)

| Method | FID ↓ | SSIM ↑ | RMSE ↓ |
|--------|------:|-------:|-------:|
| **Paper** — Degraded | 238.26 | 0.315 | 0.136 |
| **Paper** — Direct | 83.69 | 0.875 | 0.071 |
| **Paper** — Alg 2 sampled | 80.08 | 0.873 | 0.075 |
| **Ours** — Degraded | 252.71 | 0.410 | 0.125 |
| **Ours** — Direct | 99.18 | 0.789 | 0.062 |
| **Ours** — Alg 1 (naive) | 105.98 | 0.621 | 0.112 |
| **Ours** — Alg 2 sampled | 96.59 | 0.794 | 0.063 |
| **Ours** — Alg 3 (α=0.3) | 97.37 | 0.794 | 0.062 |

### Algorithm 3 — alpha sweep (inpainting FID)

| α | FID ↓ |
|---|------:|
| 0.1 | 12.44 |
| 0.3 | **11.89** |
| 0.5 | 12.15 |
| 0.7 | 12.45 |
| 1.0 | 13.62 |

Best at α = 0.3. α = 0 recovers Algorithm 2 (FID 12.16).

## Checkpoints

Checkpoints are written to `results/checkpoints/<run_name>/ckpt_XXXXXX.pt`
during training and are excluded from git. Progress grids are saved to
`results/checkpoints/<run_name>/samples/step_XXXXXX.png` at every
`sample_every` steps (default 2000).
