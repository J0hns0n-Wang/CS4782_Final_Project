# Re-implementing Cold Diffusion on CIFAR-10
### CS 4782 Final Project Report
**Johnson Wang (jw2693) · Eric Weng (ew522) · Cornell University · May 12, 2026**

---

## 1. Introduction

Standard diffusion models iteratively add Gaussian noise and train a neural network to reverse it. *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise* (Bansal et al., NeurIPS 2022) challenges this design: Gaussian noise is not a necessary ingredient. Any smoothly-parameterized, deterministic image degradation can replace it, and the resulting model retains the ability to reconstruct and synthesize images. The key contribution is **Algorithm 2** (§3.3), an anchored reverse sampler that at step *s* computes x_{s-1} = x_s − D(x̂₀, s) + D(x̂₀, s−1), where x̂₀ = R_θ(x_s, s), stabilizing the trajectory for any choice of deterministic forward process D.

We re-implement Cold Diffusion from scratch in PyTorch on CIFAR-10, reproducing all three degradations from the paper (inpainting, Gaussian blur, super-resolution) using the published training recipe. We additionally extend the paper with **Algorithm 3**, an EMA-smoothed variant of Algorithm 2, and validate it on both CIFAR-10 and MNIST.

---

## 2. Chosen Result

We target two results: **Tables 1–2** (FID / SSIM / RMSE for degraded, direct, and sampled outputs on CIFAR-10 inpainting and blur) and **Figure 2** (the qualitative sampler comparison showing Algorithm 1 failing and Algorithm 2 succeeding). These jointly establish the paper's central claim: the anchored update in Algorithm 2 is essential for smooth degradations. For inpainting the paper reports a sampled FID of 8.92 vs. a degraded FID of 40.83, giving a clear quantitative target. We also provide Algorithm 1 metrics (not reported in the paper) to quantify the failure mode.

---

## 3. Methodology

**U-Net.** 3-level U-Net with base channels 64, multipliers (1, 2, 2), two residual blocks per level (GroupNorm + SiLU activations), and self-attention at the 8×8 bottleneck. Sinusoidal time embeddings projected through a 2-layer MLP are injected into each residual block. Total: ~12M parameters.

**Degradations** (following Appendix A of the paper):
- *Inpainting*: Cumulative Gaussian mask m(i,j) = 1 − exp(−d²/2β_t²), with β₁ = 1, β_{t+1} = β_t + 0.1, T = 50, center randomized per image on CIFAR-10.
- *Blur*: 11×11 Gaussian kernel applied recursively, σ_t = 0.01t + 0.35, T = 40.
- *Super-resolution*: 2× average-pool + nearest-neighbor upsample, T = 3.

**Training.** L1 loss E[‖R_θ(D(x₀, t), t) − x₀‖₁]; Adam at 2×10⁻⁵; batch 64 with 2-step gradient accumulation (effective batch 128); EMA decay 0.995 updated every 10 steps. Inpainting: 60k steps; blur: 30k steps (paper uses 700k); SR: 100k steps. Hardware: single NVIDIA T4 GPU on Google Colab.

**Algorithm 3 (our contribution).** Algorithm 2 discards each per-step x̂₀ prediction immediately. We maintain instead a running EMA: x̂₀⁽ˢ⁾ ← α · x̂₀⁽ˢ⁺¹⁾ + (1 − α) · R(x_s, s), then use the smoothed x̂₀⁽ˢ⁾ in the anchored update. Setting α = 0 recovers Algorithm 2 exactly. We sweep α ∈ {0.1, 0.3, 0.5, 0.7, 1.0} on the CIFAR-10 test set.

**Key implementation note.** For randomized inpainting, every D(·) call within one trajectory must share the same per-image mask center; otherwise the subtraction D(x̂₀, s) − D(x̂₀, s−1) uses mismatched masks and produces artifacts. We implement explicit *state-passing*: the center is drawn once per trajectory and locked for all subsequent D calls. This requirement is implicit in the paper and was the primary debugging challenge.

---

## 4. Results & Analysis

Table 1 shows FID / SSIM / RMSE for inpainting (60k steps) and blur (30k steps). All Algorithm 1 numbers are ours; the paper does not report them.

**Table 1.** Reconstruction metrics on CIFAR-10. ↓ = lower is better, ↑ = higher is better. Paper values from Tables 1–2. Algorithm 1 numbers are ours only.

| Task | Metric | Paper — Degraded | Ours — Degraded | Paper — Direct | Ours — Direct | Paper — Alg 1 | Ours — Alg 1 | Paper — Alg 2 | Ours — Alg 2 |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Inpainting (T=50) | FID ↓ | 40.83 | 70.73 | 9.71 | 15.21 | — | 123.17 | 8.92 | 12.16 |
| | SSIM ↑ | 0.615 | 0.579 | 0.869 | 0.877 | — | 0.440 | 0.859 | 0.807 |
| | RMSE ↓ | 0.143 | 0.302 | 0.063 | 0.071 | — | 0.248 | 0.068 | 0.081 |
| Blur (T=40) | FID ↓ | 238.26 | 252.71 | 83.69 | 99.18 | — | 105.98 | 80.08 | 96.59 |
| | SSIM ↑ | 0.315 | 0.410 | 0.875 | 0.789 | — | 0.621 | 0.873 | 0.794 |
| | RMSE ↓ | 0.136 | 0.125 | 0.071 | 0.062 | — | 0.112 | 0.075 | 0.063 |

**Algorithm 1 vs. 2.** The most striking result is how completely Algorithm 1 fails: inpainting FID jumps to 123.17 (vs. Algorithm 2's 12.16) and SSIM collapses to 0.440. Each naive step re-degrades a fresh prediction using a new random mask center, compounding errors and erasing spatial coherence. Algorithm 2's anchored update eliminates this entirely.

**Gap from paper.** Our inpainting results (FID 12.16 vs. paper's 8.92) are close; the gap is consistent with a slightly smaller U-Net — the paper does not fully specify its architecture. For blur the gap is larger (FID 96.59 vs. 80.08) because we trained only 30k of the paper's 700k steps; the Algorithm 1 vs. Algorithm 2 trend is still clearly visible at this budget.

**Algorithm 3.** Table 2 compares Algorithm 2 and Algorithm 3 at the best α = 0.3 (FID sweep: 12.44, **11.89**, 12.15, 12.45, 13.62 for α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}).

**Table 2.** Algorithm 2 vs. Algorithm 3 (α = 0.3) on CIFAR-10.

| Task | Method | FID ↓ | SSIM ↑ | RMSE ↓ |
|------|--------|------:|-------:|-------:|
| Inpainting (T=50) | Alg 2 (paper) | 12.16 | 0.807 | 0.081 |
| Inpainting (T=50) | **Alg 3 (ours)** | **11.89** | **0.872** | **0.080** |
| Blur (T=40) | Alg 2 (paper) | 96.59 | 0.794 | 0.063 |
| Blur (T=40) | Alg 3 (ours) | 97.37 | 0.794 | 0.062 |

On inpainting (nonlinear D) Algorithm 3 improves FID by 0.27 and SSIM by 0.065. On blur (linear D) Algorithm 3 essentially ties. This asymmetry precisely confirms our hypothesis: the EMA helps when per-step D calls are only approximately consistent (random mask centers make inpainting weakly nonlinear in the trajectory), but adds no signal when D is exactly linear and Algorithm 2 is already optimal per the paper's §3.3 analysis.

---

## 5. Reflections

The biggest practical challenge was the state-locking requirement for randomized inpainting. Without fixing the mask center for the entire trajectory, Algorithm 2's subtraction D(x̂₀, s) − D(x̂₀, s−1) uses inconsistent masks and the output is visually indistinguishable from Algorithm 1's failures — a subtle bug that took significant debugging to isolate. We solved it with an explicit `sample_state()` abstraction that draws the stochastic part of D once per trajectory and passes it to all subsequent calls.

The broader lesson is that the paper's theoretical analysis in §3.3 (Algorithm 2 is exact for linear D) is tight enough to guide empirical design. Algorithm 3's clean linear/nonlinear split confirms that EMA smoothing is useful specifically where the theory predicts imperfection. Given more compute we would: (1) train blur/SR to the paper's full step budgets; (2) anneal α during sampling rather than holding it fixed; (3) test on CelebA (256×256) where inpainting's spatial nonlinearity is stronger.

---

## References

[1] Bansal, A., et al. *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise.* NeurIPS 2022. arXiv:2208.09392.

[2] Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.

[3] Song, J., Meng, C., Ermon, S. *Denoising Diffusion Implicit Models.* ICLR 2021.

[4] Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.* Technical Report, 2009.

[5] Paszke, A., et al. PyTorch. torchvision, torchmetrics.
