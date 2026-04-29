"""Cold Diffusion: forward = arbitrary deterministic D, reverse = trained R.

Wraps a restoration U-Net R_θ and a degradation module D. Provides:

  q_sample(x0, t, *, state)            -> D(x0, t)        (forward)
  predict_x0(xt, t)                    -> R_θ(xt, t)      (one-shot)
  sample_naive(xT, *, state)           -> Algorithm 1     (paper Fig. 2 top)
  sample_improved(xT, *, state)        -> Algorithm 2     (paper Fig. 2 bottom)

State plumbing
--------------
For randomized degradations like inpainting with `randomize_center=True`,
every call to `D` inside one Algorithm-2 trajectory must use the same
random draw, otherwise `D(x̂_0, s) - D(x̂_0, s-1)` doesn't share masks and
the iteration is incoherent. `state` is an opaque dict produced by the
degradation's `sample_state(x0)` method. Pass it into `q_sample` /
`sample_*` to lock the trajectory.

Deterministic degradations (blur, super-resolution) just don't define
`sample_state` — they ignore the kwarg, which is a no-op.
"""

import torch
import torch.nn as nn


class ColdDiffusion(nn.Module):
    def __init__(self, restoration: nn.Module, degradation: nn.Module, T: int):
        super().__init__()
        self.restoration = restoration
        self.degradation = degradation
        self.T = T

    # -- state ----------------------------------------------------------------

    def sample_state(self, x0: torch.Tensor) -> dict | None:
        """Draw fresh random state for the degradation, or None if deterministic."""
        if hasattr(self.degradation, "sample_state"):
            return self.degradation.sample_state(x0)
        return None

    # -- forward / restoration ------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, *, state: dict | None = None) -> torch.Tensor:
        if state is None:
            return self.degradation(x0, t)
        return self.degradation(x0, t, **state)

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.restoration(xt, t.float())

    # -- training loss --------------------------------------------------------

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)
        # One per-batch draw of state: every image gets a fresh per-image
        # state but the same state is used for whatever t the image lands
        # on. Matches the paper's setup (one mask center per image).
        state = self.sample_state(x0)
        xt = self.q_sample(x0, t, state=state)
        x0_pred = self.predict_x0(xt, t)
        return torch.mean(torch.abs(x0_pred - x0))

    # -- samplers -------------------------------------------------------------

    @torch.no_grad()
    def sample_naive(
        self,
        xT: torch.Tensor,
        t_start: int | None = None,
        return_trajectory: bool = False,
        state: dict | None = None,
    ) -> torch.Tensor:
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
                x = self.q_sample(x0_pred, t_prev, state=state)
            if return_trajectory:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x

    @torch.no_grad()
    def sample_improved(
        self,
        xT: torch.Tensor,
        t_start: int | None = None,
        return_trajectory: bool = False,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Algorithm 2: x_{s-1} = x_s - D(R(x_s,s), s) + D(R(x_s,s), s-1).

        For randomized degradations, pass the same `state` dict that was
        used to construct `xT`. With state=None each q_sample call samples
        fresh randomness, which is incorrect for randomized D.
        """
        t_start = t_start if t_start is not None else self.T
        x = xT
        traj = [x.clone()] if return_trajectory else None
        for s in range(t_start, 0, -1):
            t_batch = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            x0_pred = self.predict_x0(x, t_batch)
            d_s = self.q_sample(x0_pred, t_batch, state=state)
            if s - 1 == 0:
                d_sm1 = x0_pred
            else:
                t_prev = torch.full_like(t_batch, s - 1)
                d_sm1 = self.q_sample(x0_pred, t_prev, state=state)
            x = x - d_s + d_sm1
            if return_trajectory:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x
