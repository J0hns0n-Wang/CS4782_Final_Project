"""Gaussian-mask inpainting degradation from Cold Diffusion §4.2.

For an n x n image, build a 2D Gaussian centered at (cx, cy) with variance β,
peak-normalized to 1, then flipped to 1 - mask so the center is 0. The
degradation at step t multiplies x0 by the product of masks for i=1..t.

β schedule (Appendix A.2): β_1 = 1, β_{i+1} = β_i + 0.1, so β_i = 1 + 0.1*(i-1).
Total steps T = 50. For CIFAR-10 the center is randomized per image; for
CelebA the paper keeps it centered. We expose `randomize_center` for either.

State plumbing: the random mask center is the only stochastic element of
this degradation. To run a coherent trajectory (Algorithm 2) every D(·,s)
call inside one trajectory MUST share the same center. `sample_state(x0)`
draws a fresh per-image center; `forward` and `cumulative_mask` accept it
explicitly. With state=None they fall back to drawing a fresh center per
call (correct for training, wrong for sampling).
"""

import torch
import torch.nn as nn


class GaussianMaskInpainting(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        T: int = 50,
        beta_start: float = 1.0,
        beta_step: float = 0.1,
        randomize_center: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.T = T
        self.beta_start = beta_start
        self.beta_step = beta_step
        self.randomize_center = randomize_center

        coords = torch.arange(image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("yy", yy, persistent=False)
        self.register_buffer("xx", xx, persistent=False)

    def sample_state(self, x0: torch.Tensor) -> dict | None:
        """Draw fresh per-image mask centers (or None if mask is centered)."""
        if not self.randomize_center:
            return None
        B = x0.shape[0]
        device = x0.device
        cx = torch.randint(0, self.image_size, (B,), device=device).float()
        cy = torch.randint(0, self.image_size, (B,), device=device).float()
        return {"cx": cx, "cy": cy}

    def _resolve_centers(self, B: int, device, cx, cy):
        if cx is not None and cy is not None:
            return cx, cy
        if self.randomize_center:
            cx = torch.randint(0, self.image_size, (B,), device=device).float()
            cy = torch.randint(0, self.image_size, (B,), device=device).float()
        else:
            cx = torch.full((B,), (self.image_size - 1) / 2.0, device=device)
            cy = torch.full((B,), (self.image_size - 1) / 2.0, device=device)
        return cx, cy

    def _single_mask(self, beta: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        dx = self.xx[None] - cx[:, None, None]
        dy = self.yy[None] - cy[:, None, None]
        g = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * beta[:, None, None]))
        return 1.0 - g

    def cumulative_mask(
        self,
        t: torch.Tensor,
        cx: torch.Tensor | None = None,
        cy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the cumulative mask M_t = ∏_{i=1..t} z_{β_i} per image."""
        device = t.device
        B = t.shape[0]
        mask = torch.ones(B, self.image_size, self.image_size, device=device)
        cx, cy = self._resolve_centers(B, device, cx, cy)

        max_t = int(t.max().item())
        for i in range(1, max_t + 1):
            beta_i = torch.full((B,), self.beta_start + self.beta_step * (i - 1), device=device)
            step_mask = self._single_mask(beta_i, cx, cy)
            active = (t >= i).float()[:, None, None]
            mask = mask * (step_mask * active + (1.0 - active))
        return mask.unsqueeze(1)

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        cx: torch.Tensor | None = None,
        cy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self.cumulative_mask(t, cx=cx, cy=cy)
        return x0 * mask
