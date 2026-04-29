"""Super-resolution degradation from Cold Diffusion §4.3.

At each step the image is halved (avg-pool 2x2) then nearest-neighbor
upsampled back to the original size. For CIFAR-10 / MNIST the paper uses
T = 3 (32 -> 16 -> 8 -> 4); for CelebA T = 6 (down to 2x2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResolution(nn.Module):
    def __init__(self, image_size: int = 32, T: int = 3):
        super().__init__()
        self.image_size = image_size
        self.T = T
        self.resolutions = [image_size // (2 ** i) for i in range(T + 1)]
        if any(r < 1 for r in self.resolutions):
            raise ValueError(f"T={T} too large for image_size={image_size}")

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x0)
        for i, ti in enumerate(t.tolist()):
            xi = x0[i : i + 1]
            if ti > 0:
                low = F.avg_pool2d(xi, kernel_size=2 ** ti, stride=2 ** ti)
                xi = F.interpolate(low, size=self.image_size, mode="nearest")
            out[i] = xi[0]
        return out
