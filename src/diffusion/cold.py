"""Cold Diffusion: forward = arbitrary deterministic D, reverse = trained R.

Wraps a restoration U-Net R_θ and a degradation module D. Provides:

  q_sample(x0, t)            -> D(x0, t)        (forward)
  predict_x0(xt, t)           -> R_θ(xt, t)      (one-shot reconstruction)
  sample_naive(xT)            -> Algorithm 1     (paper Fig. 2 top)
  sample_improved(xT)         -> Algorithm 2     (paper Fig. 2 bottom; recommended)

Algorithm 2 implements the update
    x_{s-1} = x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1)
which (per §3.3) cancels the first-order Taylor error of D in R, so the
iteration produces the *correct* trajectory whenever D is approximately
linear in x even when R is an imperfect inverse.
"""

from typing import Callable

import torch
import torch.nn as nn


class ColdDiffusion(nn.Module):
    def __init__(self, restoration: nn.Module, degradation: nn.Module, T: int):
        super().__init__()
        self.restoration = restoration
        self.degradation = degradation
        self.T = T

    # -- forward / restoration ------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.degradation(x0, t)

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.restoration(xt, t.float())

    # -- training loss --------------------------------------------------------

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)
        xt = self.q_sample(x0, t)
        x0_pred = self.predict_x0(xt, t)
        return torch.mean(torch.abs(x0_pred - x0))

    # -- samplers -------------------------------------------------------------

    @torch.no_grad()
    def sample_naive(self, xT: torch.Tensor, t_start: int | None = None,
                     return_trajectory: bool = False) -> torch.Tensor:
        """Algorithm 1: x_{s-1} = D(R(x_s, s), s-1)."""
        t_start = t_start if t_start is not None else self.T
        x = xT
        traj = [x.clone()] if return_trajectory else None
        for s in range(t_start, 0, -1):
            t_batch = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            x0_pred = self.predict_x0(x, t_batch)
            if s - 1 == 0:
                x = x0_pred
            else:
                t_prev = torch.full_like(t_batch, s - 1)
                x = self.q_sample(x0_pred, t_prev)
            if return_trajectory:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x

    @torch.no_grad()
    def sample_improved(self, xT: torch.Tensor, t_start: int | None = None,
                        return_trajectory: bool = False) -> torch.Tensor:
        """Algorithm 2: x_{s-1} = x_s - D(R(x_s,s), s) + D(R(x_s,s), s-1)."""
        t_start = t_start if t_start is not None else self.T
        x = xT
        traj = [x.clone()] if return_trajectory else None
        for s in range(t_start, 0, -1):
            t_batch = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            x0_pred = self.predict_x0(x, t_batch)
            d_s = self.q_sample(x0_pred, t_batch)
            if s - 1 == 0:
                d_sm1 = x0_pred
            else:
                t_prev = torch.full_like(t_batch, s - 1)
                d_sm1 = self.q_sample(x0_pred, t_prev)
            x = x - d_s + d_sm1
            if return_trajectory:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x
