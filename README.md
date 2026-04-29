# CS4782 Final Project — Cold Diffusion

Re-implementation of *Cold Diffusion: Inverting Arbitrary Image Transforms
Without Noise* (Bansal et al., 2022, [arXiv:2208.09392](https://arxiv.org/abs/2208.09392))
on CIFAR-10. The paper shows that diffusion-style generative models do not
require Gaussian noise — any deterministic, smoothly-parameterized image
degradation can play that role, and a generalized iterative sampler
(Algorithm 2 in the paper) inverts it.

## 1. Introduction

This repo contains a from-scratch PyTorch implementation of Cold Diffusion's
training and sampling procedures. We reproduce the inpainting, blur, and
super-resolution forward processes; train a U-Net restoration network; and
demonstrate the paper's headline result that the improved sampler
(Algorithm 2) reconstructs cleanly where the naive sampler (Algorithm 1)
accumulates errors.

## 2. Chosen Result

Three results, all on CIFAR-10:

1. **Reconstruction quality** for the inpainting and blur tasks (paper
   Tables 1–2): FID / SSIM / RMSE of the degraded, direct, and Algorithm-2
   sampled images vs. the test set.
2. **Sampler comparison** (paper Figure 2 and Appendix A.7): same trained
   model, Algorithm 1 vs Algorithm 2 — Algorithm 1 visibly fails to invert
   smooth degradations.
3. **Cold generation** (paper §5.3): unconditional CIFAR-10 samples produced
   by running Algorithm 2 from random color blocks.

## 3. GitHub Contents

```
src/
  models/         U-Net (unet.py) and EMA wrapper (ema.py)
  degradations/   inpainting, blur, super-resolution as nn.Modules
  diffusion/      ColdDiffusion: q_sample, predict_x0, sample_naive (Alg.1), sample_improved (Alg.2)
  generation/     Random-color-fill inpainting + sample_generative
  evaluation/     FID/SSIM/RMSE via torchmetrics
  training/       Training loop (L1 loss, Adam 2e-5, EMA, gradient accumulation)
data/             CIFAR-10 lands here; see data/README.md
results/          Checkpoints + sample grids written here at training time
project.ipynb     End-to-end demo notebook (degradations → train → reconstruct → sample → metrics)
smoke_test.py     Quick CPU sanity check of every module
```

## 4. Re-implementation Details

- **Backbone:** 3-level U-Net (32→16→8) with sinusoidal time embeddings,
  GroupNorm residual blocks, and self-attention at the 8×8 bottleneck.
  ~6.7M parameters.
- **Degradations** (all per the paper's Appendix A):
  - *Inpainting*: Gaussian mask `1 - exp(-d²/2β²)` with β starting at 1
    and growing by 0.1 per step, T=50, randomized center per CIFAR image.
  - *Blur*: 11×11 Gaussian kernel applied recursively, σ_t = 0.01·t + 0.35,
    T=40.
  - *Super-resolution*: 2× avg-pool + nearest upsample, T=3 (32→16→8→4).
  - *Generative inpainting*: same mask, but the masked region is filled
    with a random per-image color so x_T is a near-solid color block.
- **Training:** L1 loss between `R(D(x_0,t),t)` and `x_0`, Adam @ 2e-5,
  gradient accumulation every 2 steps, EMA decay 0.995 updated every 10
  steps. Inpainting target: 60k gradient steps, batch 64.
- **Sampling:** Algorithm 2 from the paper.

Modifications vs. paper: trained on a smaller compute budget than the
paper's 60k–700k step regime; the U-Net is somewhat smaller than the
paper's (which is not fully specified) to fit single-GPU training.

## 5. Reproduction Steps

```bash
pip install -r requirements.txt

# Sanity check
python smoke_test.py

# Train on the inpainting task (paper recipe; expect 6–10h on a modern GPU)
python -m src.training.train --degradation inpainting --total-steps 60000

# Or the random-color-fill variant for unconditional generation
python -m src.training.train --degradation generative_inpainting --total-steps 60000

# End-to-end demo
jupyter lab project.ipynb
```

GPU is recommended (CIFAR-10 inpainting at 60k steps takes ~6h on a single
modern GPU). On CPU the smoke test runs in seconds; full training is
impractical. Python 3.10+ with a Python build that includes the `xz/lzma`
shared library (i.e. anything but a pyenv build missing xz at compile time).

## 6. Results / Insights

Expected qualitative output:

- **Reconstruction:** at T=50 inpainting the input is mostly black; the
  one-shot prediction recovers a blurry image; Algorithm 2 produces a
  sharper reconstruction whose FID is lower than the direct prediction.
- **Sampler comparison:** Algorithm 2 closely tracks the original,
  Algorithm 1 produces compounding artifacts within a few steps — exactly
  the paper's Figure 2.
- **Cold generation:** diverse 32×32 samples emerge from random color
  blocks, demonstrating that the iterative sampler can synthesize images
  from minimal initial information when the U-Net was trained with
  per-batch random color seeds.

Quantitative numbers depend on training budget. Paper Table 2 reports
CIFAR-10 inpainting FID ≈ 8.9 (sampled) at the full 60k-step recipe; we
report whatever our compute budget reaches in `results/`.

## 7. Conclusion

The paper's central claim — that Gaussian noise is not load-bearing for
diffusion-style generation, and any smoothly-parameterized deterministic
degradation works given the right reverse rule — is reproducible from a
small, self-contained codebase. The biggest delta is the choice of
sampler: Algorithm 2's first-order correction is the difference between
"works" and "compounding artifacts" for smooth `D`.

## 8. References

- Bansal, A. et al. *Cold Diffusion: Inverting Arbitrary Image Transforms
  Without Noise.* arXiv:2208.09392, 2022.
- Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.*
  NeurIPS 2020.
- Song, J., Meng, C., Ermon, S. *Denoising Diffusion Implicit Models.*
  ICLR 2021.
- Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.*
  Tech report, 2009 (CIFAR-10).
- Hendrycks, D., Dietterich, T. *Benchmarking Neural Network Robustness
  to Common Corruptions and Perturbations.* ICLR 2019 (snowification, used
  as reference but not implemented here).
- PyTorch, torchvision, torchmetrics.

## 9. Acknowledgements

Cornell CS 4782 (Spring 2026) final project. Thanks to the course staff
for project guidance. The reference Cold Diffusion implementation by the
original authors is at
[github.com/arpitbansal297/Cold-Diffusion-Models](https://github.com/arpitbansal297/Cold-Diffusion-Models);
we wrote our code from scratch using the paper as the spec.

Authors: **Johnson Wang** (jw2693), **Eric Weng** (ew522).
