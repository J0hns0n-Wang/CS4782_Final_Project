"""Reconstruction quality metrics: FID, SSIM, RMSE.

Mirrors the paper's Tables 1–3. We compare three image populations
against the held-out test set:

  - "Degraded": D(x0, T)
  - "Direct":   R(D(x0, T), T)              (one-shot reconstruction)
  - "Sampled":  Algorithm-2 trajectory      (iterative reconstruction)

Evaluation is **streaming**: we update running FID statistics and
weighted SSIM/RMSE per batch. Nothing is stacked at 299x299 across the
whole test set, so 10K-image evaluation fits in a few hundred MB of GPU
memory regardless of batch count. Same public API as before:

    results = evaluate_diffusion(diff, test_loader, device='cuda')
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


@dataclass
class MetricResult:
    fid: float
    ssim: float
    rmse: float


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).to(torch.uint8)


def _resize_for_inception(x: torch.Tensor) -> torch.Tensor:
    # Inception expects >= 299x299; for CIFAR we bilinearly upsample.
    if x.shape[-1] < 299:
        x = F.interpolate(x.float(), size=299, mode="bilinear", align_corners=False)
    return _to_uint8(x)


@torch.no_grad()
def evaluate_diffusion(diffusion, test_loader, device: str = "cuda",
                       max_batches: int | None = None) -> dict[str, MetricResult]:
    """Streaming degraded / direct / sampled metrics over the test set."""
    diffusion = diffusion.to(device).eval()
    T = diffusion.T

    keys = ["degraded", "direct", "sampled"]
    fid_mods = {k: FrechetInceptionDistance(normalize=False).to(device) for k in keys}
    ssim_mod = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ssim_sums = {k: 0.0 for k in keys}
    rmse_sums = {k: 0.0 for k in keys}
    n = 0

    is_cuda = (device == "cuda" or
               (isinstance(device, torch.device) and device.type == "cuda"))

    for i, (x0, _) in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        bs = x0.shape[0]
        t_full = torch.full((bs,), T, device=device, dtype=torch.long)
        state = diffusion.sample_state(x0)
        xT = diffusion.q_sample(x0, t_full, state=state)
        x_direct  = diffusion.predict_x0(xT, t_full).clamp(0, 1)
        x_sampled = diffusion.sample_improved(xT, state=state).clamp(0, 1)
        deg = xT.clamp(0, 1)

        # FID: feed real once per FID module (cheap), then each fake.
        real_resized = _resize_for_inception(x0)
        for k in keys:
            fid_mods[k].update(real_resized, real=True)
        fid_mods["degraded"].update(_resize_for_inception(deg),       real=False)
        fid_mods["direct"  ].update(_resize_for_inception(x_direct),  real=False)
        fid_mods["sampled" ].update(_resize_for_inception(x_sampled), real=False)

        # SSIM/RMSE: per-batch weighted average.
        for k, fake in (("degraded", deg), ("direct", x_direct), ("sampled", x_sampled)):
            ssim_sums[k] += ssim_mod(fake, x0).item() * bs
            rmse_sums[k] += torch.sqrt(F.mse_loss(fake, x0)).item() * bs
        n += bs

        del x0, xT, x_direct, x_sampled, deg, real_resized
        if is_cuda:
            torch.cuda.empty_cache()

    return {
        k: MetricResult(
            fid=fid_mods[k].compute().item(),
            ssim=ssim_sums[k] / n,
            rmse=rmse_sums[k] / n,
        ) for k in keys
    }


@torch.no_grad()
def evaluate_sampler(diffusion, sampler_fn, test_loader, device: str = "cuda",
                     max_batches: int | None = None) -> MetricResult:
    """Streaming FID/SSIM/RMSE for an arbitrary sampler.

    sampler_fn(xT, state) -> reconstruction in [0, 1]. Useful for
    comparing custom samplers (e.g., Algorithm 3) against the trained
    model on the same test loader.
    """
    diffusion = diffusion.to(device).eval()
    T = diffusion.T

    fid = FrechetInceptionDistance(normalize=False).to(device)
    ssim_mod = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_sum, rmse_sum, n = 0.0, 0.0, 0

    is_cuda = (device == "cuda" or
               (isinstance(device, torch.device) and device.type == "cuda"))

    for i, (x0, _) in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = x0.to(device, non_blocking=True)
        bs = x0.shape[0]
        t_full = torch.full((bs,), T, device=device, dtype=torch.long)
        state = diffusion.sample_state(x0)
        xT = diffusion.q_sample(x0, t_full, state=state)
        x_hat = sampler_fn(xT, state).clamp(0, 1)

        fid.update(_resize_for_inception(x0),    real=True)
        fid.update(_resize_for_inception(x_hat), real=False)
        ssim_sum += ssim_mod(x_hat, x0).item() * bs
        rmse_sum += torch.sqrt(F.mse_loss(x_hat, x0)).item() * bs
        n += bs

        del x0, xT, x_hat
        if is_cuda:
            torch.cuda.empty_cache()

    return MetricResult(fid=fid.compute().item(), ssim=ssim_sum / n, rmse=rmse_sum / n)


def compute_metrics(real: torch.Tensor, fake: torch.Tensor, device: str = "cuda") -> MetricResult:
    """Two-tensor metric on already-stacked real and fake.

    Kept for one-off use on small batches; for full-test-set evaluation
    use the streaming evaluate_diffusion() / evaluate_sampler() above.
    """
    fid = FrechetInceptionDistance(normalize=False).to(device)
    fid.update(_resize_for_inception(real), real=True)
    fid.update(_resize_for_inception(fake), real=False)
    fid_val = fid.compute().item()

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim(fake.clamp(0, 1), real.clamp(0, 1)).item()

    rmse_val = torch.sqrt(F.mse_loss(fake.clamp(0, 1), real.clamp(0, 1))).item()
    return MetricResult(fid=fid_val, ssim=ssim_val, rmse=rmse_val)
