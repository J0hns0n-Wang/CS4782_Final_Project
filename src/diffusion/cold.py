"""Cold Diffusion: forward = arbitrary deterministic D, reverse = trained R.

Wraps a restoration U-Net R_θ and a degradation module D. Provides:

  q_sample(x0, t, *, state)            -> D(x0, t)        (forward)
  predict_x0(xt, t)                    -> R_θ(xt, t)      (one-shot)
  sample_naive(xT, *, state)           -> Algorithm 1     (paper Fig. 2 top)
  sample_improved(xT, *, state)        -> Algorithm 2     (paper Fig. 2 bottom)
  sample_ema(xT, alpha, *, state)      -> Algorithm 3     (this project)

Algorithm 3 (sample_ema)
------------------------
Algorithm 2 throws away each x̂_0 prediction immediately. But across the T
sampler steps, R is predicting the same target (x_0) repeatedly with
different inputs -- those predictions can be averaged to reduce noise.
We maintain a running EMA of x̂_0 and use the smoothed estimate in the
D updates instead of the raw single-step prediction.

For perfectly linear D, Algorithm 2 is already exact and the EMA gives no
benefit. For nonlinear/imperfect D (e.g. inpainting with random mask
centers), the EMA reduces the per-step error and improves both visual
quality and FID.

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

    @torch.no_grad()
    def sample_ema(
        self,
        xT: torch.Tensor,
        alpha: float = 0.5,
        t_start: int | None = None,
        return_trajectory: bool = False,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Algorithm 3: Algorithm 2 with EMA-smoothed x̂_0 across sampler steps.

        At step s, instead of using the raw R(x_s, s) prediction, blend it
        with the running EMA:

            x̂_0_smooth = alpha * x̂_0_smooth_prev + (1 - alpha) * R(x_s, s)

        Then apply Algorithm 2's update with the smoothed estimate. alpha=0
        recovers Algorithm 2 exactly. alpha=1 freezes the first prediction.
        """
        t_start = t_start if t_start is not None else self.T
        x = xT
        x0_smooth = None
        traj = [x.clone()] if return_trajectory else None
        for s in range(t_start, 0, -1):
            t_batch = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            x0_pred = self.predict_x0(x, t_batch)
            x0_smooth = x0_pred if x0_smooth is None else alpha * x0_smooth + (1 - alpha) * x0_pred
            d_s = self.q_sample(x0_smooth, t_batch, state=state)
            if s - 1 == 0:
                d_sm1 = x0_smooth
            else:
                t_prev = torch.full_like(t_batch, s - 1)
                d_sm1 = self.q_sample(x0_smooth, t_prev, state=state)
            x = x - d_s + d_sm1
            if return_trajectory:
                traj.append(x.clone())
        return (x, traj) if return_trajectory else x
