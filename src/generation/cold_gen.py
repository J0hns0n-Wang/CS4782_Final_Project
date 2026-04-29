"""Cold generation via the inpainting transform (Cold Diffusion §5.3).

Standard inpainting masks the image to zero, which is fine for restoration
but doesn't allow for diverse *generation* — every x_T collapses to the
same all-black image. The paper's fix: replace the masked region with a
per-image random solid color c, so x_T = M_T*x0 + (1 - M_T)*c, and as
M_T -> 0 over the schedule x_T becomes essentially a solid color block.

Crucially, R must be **trained** with this degradation — at every step
the color seed c is freshly random — so it learns to grow plausible
content out of a color block. At sampling time we plug in chosen colors.
"""

import torch
import torch.nn as nn

from src.degradations.inpainting import GaussianMaskInpainting


class GenerativeInpainting(nn.Module):
    """Inpainting that fills the masked region with a per-image solid color.

    Behavior:
      * If a color override has been set with `set_color(c)`, use it.
      * Otherwise sample a fresh random color per call (training mode).

    During training the auto-sampled colors give the model many seeds; at
    sampling time we pin the color so the same `c` is used at every step.
    """

    def __init__(self, image_size: int = 32, T: int = 50):
        super().__init__()
        self.base = GaussianMaskInpainting(image_size=image_size, T=T, randomize_center=False)
        self.image_size = image_size
        self.T = T
        self._color_override: torch.Tensor | None = None

    def set_color(self, c: torch.Tensor | None) -> None:
        """c: (B, C, 1, 1) tensor in [0, 1], or None to clear the override."""
        self._color_override = c

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mask = self.base.cumulative_mask(t)
        if self._color_override is not None:
            c = self._color_override
        else:
            c = torch.rand(x0.shape[0], x0.shape[1], 1, 1, device=x0.device)
        return x0 * mask + (1.0 - mask) * c


@torch.no_grad()
def sample_generative(diffusion, n: int = 16, image_size: int = 32,
                      device: str = "cuda", colors: torch.Tensor | None = None) -> torch.Tensor:
    """Generate `n` images by running Algorithm 2 from random color blocks.

    Requires `diffusion.degradation` to be a `GenerativeInpainting`. Pin
    the color, build the fully-masked x_T (which is just the color block),
    then sample.
    """
    if not isinstance(diffusion.degradation, GenerativeInpainting):
        raise TypeError("Diffusion must use GenerativeInpainting for generation.")

    if colors is None:
        colors = torch.rand(n, 3, 1, 1, device=device)
    diffusion.degradation.set_color(colors)
    try:
        t_full = torch.full((n,), diffusion.T, device=device, dtype=torch.long)
        dummy = torch.zeros(n, 3, image_size, image_size, device=device)
        xT = diffusion.degradation(dummy, t_full)
        out = diffusion.sample_improved(xT)
    finally:
        diffusion.degradation.set_color(None)
    return out.clamp(0, 1)
