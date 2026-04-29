"""Reconstruction quality metrics: FID, SSIM, RMSE.

These mirror the paper's Tables 1–3. We compare three image populations
against the held-out test set:

  - "Degraded": D(x0, T)
  - "Direct": R(D(x0, T), T)              (one-shot reconstruction)
  - "Sampled": Algorithm-2 trajectory     (iterative reconstruction)

FID uses the standard InceptionV3 features (torchmetrics). Inputs are in
[0, 1]; we convert to uint8 [0, 255] for FID.
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


def compute_metrics(real: torch.Tensor, fake: torch.Tensor, device: str = "cuda") -> MetricResult:
    """Both `real` and `fake` are batches of images in [0, 1] on `device`."""
    fid = FrechetInceptionDistance(normalize=False).to(device)
    fid.update(_resize_for_inception(real), real=True)
    fid.update(_resize_for_inception(fake), real=False)
    fid_val = fid.compute().item()

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim(fake.clamp(0, 1), real.clamp(0, 1)).item()

    rmse_val = torch.sqrt(F.mse_loss(fake.clamp(0, 1), real.clamp(0, 1))).item()
    return MetricResult(fid=fid_val, ssim=ssim_val, rmse=rmse_val)


@torch.no_grad()
def evaluate_diffusion(diffusion, test_loader, device: str = "cuda",
                       max_batches: int | None = None) -> dict[str, MetricResult]:
    """Run degraded / direct / sampled metrics over the test set."""
    diffusion = diffusion.to(device).eval()
    T = diffusion.T

    real_batches, deg_batches, dir_batches, samp_batches = [], [], [], []
    for i, (x0, _) in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = x0.to(device)
        t_full = torch.full((x0.shape[0],), T, device=device, dtype=torch.long)
        xT = diffusion.q_sample(x0, t_full)
        x_direct = diffusion.predict_x0(xT, t_full).clamp(0, 1)
        x_sampled = diffusion.sample_improved(xT).clamp(0, 1)

        real_batches.append(x0.cpu())
        deg_batches.append(xT.clamp(0, 1).cpu())
        dir_batches.append(x_direct.cpu())
        samp_batches.append(x_sampled.cpu())

    real = torch.cat(real_batches).to(device)
    deg = torch.cat(deg_batches).to(device)
    direct = torch.cat(dir_batches).to(device)
    sampled = torch.cat(samp_batches).to(device)

    return {
        "degraded": compute_metrics(real, deg, device),
        "direct": compute_metrics(real, direct, device),
        "sampled": compute_metrics(real, sampled, device),
    }
