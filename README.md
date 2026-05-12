# CS4782 Final Project — Cold Diffusion

## Authors: Eric Weng (ew522), Johnson Wang (jw2693)

Re-implementation of _Cold Diffusion: Inverting Arbitrary Image Transforms
Without Noise_ (Bansal et al., 2022) on CIFAR-10 and MNIST dataset. The paper shows that diffusion-style
generative models do not require Gaussian noise — any deterministic, smoothly-parameterized image
degradation can play that role, and a generalized iterative sampler
(Algorithm 2 in the paper) inverts it.

## 1. Introduction

This repository contains a from-scratch PyTorch implementation of Cold Diffusion's
training and sampling procedures. We reproduce the inpainting and blur
forward processes; train a U-Net restoration network; and demonstrate the
paper's headline result that the improved sampler (Algorithm 2) reconstructs
cleanly where the naive sampler (Algorithm 1) accumulates errors. This opens up a new way of thinking about diffusion models: Gaussian noise may not be a mandatory ingredient for diffusion-style training and sampling. Instead, the core mechanism may be the iterative reversal of a gradual degradation process. By showing that deterministic transformations like blur and masking can replace noise, Cold Diffusion provides support for new discoveries in generative modeling beyond the traditional Gaussian noising framework.

## 2. Chosen Result

We aim to reproduce two results as documented in _Section 4.1_ and _Section 4.2_ of the paper
where it performed experiments on the vision tasks of deblurring and inpainting. We will gradually
remove information from the clean image and using the trained restoration model and different sampling
algorithms to restore these images back to their original form. We strive to achieve similar
qualitative and quantitative results similar to the paper.

Two results, both on CIFAR-10 and MNIST dataset:

1. **Reconstruction quality** for the masking and blurring tasks (paper
      Tables 1–2): FID / SSIM / RMSE of the degraded, direct, and Algorithm-2
      sampled images vs. the test set.
2. **Sampler comparison** (paper Figure 2 and Appendix A.7): same trained
      model, Algorithm 1 vs Algorithm 2 — Algorithm 1 visibly fails to invert
      smooth degradations.

## 3. GitHub Contents

```
code/
  models/         U-Net (unet.py) and EMA wrapper (ema.py)
  degradations/   inpainting and blur as nn.Modules
  diffusion/      ColdDiffusion: q_sample, predict_x0, sample_naive (Alg.1), sample_improved (Alg.2)
  evaluation/     FID/SSIM/RMSE via torchmetrics
  training/       Training loop (L1 loss, Adam 2e-5, EMA, gradient accumulation)
data/             CIFAR-10 lands here; see data/README.md
results/          Quantitative Result Grids (FID, SSIM, RMSE) and Qualitative Results (Restored images)
project.ipynb     End-to-end demo notebook (degradations → train → reconstruct → sample → metrics)
```

## 4. Re-implementation Details

### Model

3-level U-Net (32→16→8) with sinusoidal time embeddings, GroupNorm residual
blocks, and self-attention at the 8×8 bottleneck. Same architecture is used
for inpainting and blur, on both CIFAR-10 (3-channel) and MNIST (1-channel,
padded to 32×32).

### Degradations (per the paper's Appendix A)

- _Inpainting_: Gaussian mask `1 - exp(-d²/2β²)` with β starting at 1 and
    growing by 0.1 per step, T=50, randomized mask center per image.
- _Blur_ (CIFAR): 11×11 Gaussian kernel applied recursively, σ_t = 0.01·t + 0.35,
    T=40.
- _Blur_ (MNIST): more aggressive 17×17 kernel with σ_t = 0.05·t + 0.5,
    T=40, so digits degrade to near-uniform blobs by t=T.

### Training

Shared recipe across all four runs: L1 loss between `R(D(x_0,t),t)` and
`x_0`, Adam @ 2e-5, batch 64 with gradient accumulation every 2 steps,
EMA decay 0.995 updated every 10 grad steps.

| Run               | Dataset   | Degradation          | Steps                   |
| ----------------- | --------- | -------------------- | ----------------------- |
| CIFAR inpainting  | CIFAR-10  | Gaussian mask, T=50  | 60k (paper recipe)      |
| CIFAR blur        | CIFAR-10  | Gaussian blur, T=40  | 30k (compute-budgeted)  |
| MNIST inpainting  | MNIST     | Gaussian mask, T=50  | 30k                     |
| MNIST blur        | MNIST     | severe blur, T=40    | 30k                     |

### Sampling

Three reverse rules, all sharing the same trained model:

- **Algorithm 1** (naive): `x_{t-1} = D(R(x_t, t), t-1)`. Compounds error
    because each step re-degrades a fresh prediction.
- **Algorithm 2** (paper §3.3, headline result): `x_{t-1} = x_t - D(x̂_0, t) + D(x̂_0, t-1)`.
    First-order correction that anchors the update to the current `x_t`.
- **Algorithm 3** (our proposed experiment): same as Alg 2 but with an EMA over
    `x̂_0` across sampler steps. Implemented in `code/diffusion/cold.py` as
    `ColdDiffusion.sample_ema()`.

### Modifications vs. paper

Smaller compute budget than the paper's 60k–700k step regime (especially
for blur); the U-Net is somewhat smaller than the paper's (which is not
fully specified) to fit single-GPU training.

## 5. Reproduction Steps

### End-to-end notebook (recommended)

`project.ipynb` runs the full pipeline — degradations, training, all three
samplers, and the FID/SSIM/RMSE evaluation — on both CIFAR-10 and MNIST.
Training cells auto-skip if a checkpoint is already present in
`results/checkpoints/<run_name>`, so you can re-run the notebook without retraining.

```bash
pip install -r requirements.txt
jupyter lab project.ipynb
```

### Compute

GPU is strongly recommended. CIFAR-10 inpainting at 60k steps takes ~6h on
a modern GPU; the lighter 30k runs (CIFAR blur, MNIST) finish in 1–2h. Full
training on CPU is impractical.

## 6. Results / Insights

Both figures below come from the **CIFAR-10 inpainting** run (T=50, 60k
steps, Gaussian mask with random per-image center) evaluated on the full
10k-image test set.

### Qualitative — sampler comparison

![CIFAR-10 inpainting reconstructions](IMG_7081.png)

Rows, top to bottom: **degraded input** (mask roughly blacks out 30–40% of
the image), **Algorithm 1 (naive)**, **Algorithm 2 (paper)**, **Algorithm
3 (EMA, α=0.3)**, **original**. Algorithm 1 visibly collapses — colorful
blob artifacts that compound across the T sampler steps because each step
re-degrades a fresh prediction. Algorithm 2 reconstructs the masked region
cleanly and matches the originals; Algorithm 3 looks essentially
indistinguishable from Algorithm 2 in this column-by-column comparison.

### Quantitative — FID / SSIM / RMSE on the test set

![CIFAR-10 inpainting metrics table](image.png)

Columns are **degraded | direct R(x_T, T) | Algorithm 1 (naive) | Algorithm 2 (sampled)**.

|          | Degraded  | Direct  | Alg 1 (naive)  | Alg 2 (sampled)  |
| -------- | --------- | ------- | -------------- | ---------------- |
| FID ↓    | 70.73     | 15.21   | **123.17**     | **12.16**        |
| SSIM ↑   | 0.579     | 0.877   | **0.440**      | 0.807            |
| RMSE ↓   | 0.302     | 0.071   | **0.248**      | **0.081**        |

Two things stand out, and together they make the paper's claim concrete:

1. **Algorithm 1 is worse than the degraded input on FID** (123.17 vs 70.73).
      Iterating the naive rule actively _destroys_ image quality — the
      compounding artifacts in the qualitative figure show up here as the
      worst FID/SSIM/RMSE in the table.
2. **Algorithm 2 wins on FID** (12.16, lower than even the direct
      prediction at 15.21). The one-shot direct prediction is pixel-faithful
      (highest SSIM=0.877) but blurry; Algorithm 2's anchored update trades a
      tiny amount of SSIM for substantially better perceptual quality which means
      we are producing more REALISTIC images.

The paper reports CIFAR-10 inpainting FID ≈ 8.9 at the same 60k-step
recipe; we land at 12.16 with a smaller U-Net and no architectural search.
The Algorithm 1 vs Algorithm 2 gap — the headline result we set out to
reproduce — is clearly present.

## 7. Conclusion

The paper's central claim — that Gaussian noise is not load-bearing for
diffusion-style generation, and any smoothly-parameterized deterministic
degradation works given the right reverse rule — is reproducible from a
small, self-contained codebase. We see this concretely in our CIFAR-10
inpainting numbers: the same trained restoration network produces a
near-useless reconstruction under Algorithm 1 (FID 123.17, worse than the
degraded input) and the paper's headline-quality reconstruction under
Algorithm 2 (FID 12.16). The model never changes — only the reverse rule
does. The first-order correction in Algorithm 2 (`x_t - D(x̂_0, t) + D(x̂_0, t-1)`)
is therefore not a tuning detail but the load-bearing piece of the
method: it is the difference between "works" and "compounding artifacts"
for smooth `D`.

A few lessons stood out during reproduction:

- **Direct prediction vs. iterative sampling is a trade-off.**
  One-shot `R(x_T, T)` had the highest SSIM (0.877) because
    it is pixel-faithful but blurry, while Algorithm 2 traded a small amount
    of SSIM for a much better FID (12.16 vs 15.21). The "best" sampler
    depends on which metric you optimize.
- **Mask state must be shared between the forward and reverse passes.**
    With per-image randomized mask centers, sampling falls apart unless the
    same mask state is threaded through `q_sample` and every `D`-call inside
    the sampler. This isn't called out in the paper but is essential for the
    numbers to come out right.
- **Compute budget matters more for blur than inpainting.** Inpainting
    converges to paper-comparable quality in 60k steps; blur takes ~700k in
    the paper, and our 30k-step blur run shows the Alg 1 vs Alg 2 trend but
    with noisier reconstructions.

## 8. References

- Bansal, A. et al. _Cold Diffusion: Inverting Arbitrary Image Transforms
    Without Noise._ arXiv:2208.09392, 2022.
- Ho, J., Jain, A., Abbeel, P. _Denoising Diffusion Probabilistic Models._
    NeurIPS 2020.
- Song, J., Meng, C., Ermon, S. _Denoising Diffusion Implicit Models._
    ICLR 2021.
- Krizhevsky, A. _Learning Multiple Layers of Features from Tiny Images._
    Tech report, 2009 (CIFAR-10).
- Hendrycks, D., Dietterich, T. _Benchmarking Neural Network Robustness
    to Common Corruptions and Perturbations._ ICLR 2019 (snowification, used
    as reference but not implemented here).
- PyTorch, torchvision, torchmetrics.

## 9. Acknowledgements

Cornell CS 4782 (Spring 2026) final project. Thanks to the course staff
for project guidance. The reference Cold Diffusion implementation by the
original authors is at
[github.com/arpitbansal297/Cold-Diffusion-Models](https://github.com/arpitbansal297/Cold-Diffusion-Models);
we wrote our code from scratch using the paper as the spec.
