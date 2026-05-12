"""Gaussian-blur degradation from Cold Diffusion §4.1 / Appendix A.1.

The forward process composes Gaussian convolutions: x_t = G_t * G_{t-1} * ... * G_1 * x_0.
For CIFAR-10 the paper uses an 11x11 kernel and std σ_t = 0.01*t + 0.35,
recursively applied for T = 40 steps. Equivalently, since composing Gaussians
adds variances, the t-step blur has σ_eff = sqrt(Σ σ_i^2). We use the direct
recursive form to match the paper's training distribution exactly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(size: int, sigma: float, device, dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


class GaussianBlur(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        T: int = 40,
        sigma_fn=lambda t: 0.01 * t + 0.35,
        channels: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.T = T
        self.sigma_fn = sigma_fn
        self.channels = channels
        # Pre-compute one kernel per step (depthwise)
        kernels = []
        for i in range(1, T + 1):
            k = _gaussian_kernel_2d(kernel_size, max(sigma_fn(i), 1e-6), device="cpu")
            kernels.append(k)
        # shape: (T, 1, k, k)
        self.register_buffer("kernels", torch.stack(kernels)[:, None], persistent=False)

    def _apply_step(self, x: torch.Tensor, i: int) -> torch.Tensor:
        # i is 1-indexed
        k = self.kernels[i - 1].expand(self.channels, 1, self.kernel_size, self.kernel_size)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, k, groups=self.channels)

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply recursive blur to a level given per-image by t.

        Different images in a batch may want different t. We loop over the
        max t and gate updates with a per-image active mask — same trick as
        the inpainting module.
        """
        max_t = int(t.max().item())
        x = x0
        if max_t == 0:
            return x
        for i in range(1, max_t + 1):
            x_blurred = self._apply_step(x, i)
            active = (t >= i).float().view(-1, 1, 1, 1)
            x = x_blurred * active + x * (1.0 - active)
        return x
