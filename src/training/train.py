"""Cold Diffusion training loop.

Per the paper (Appendix A.1, A.2): L1 loss between R(D(x0,t),t) and x0,
Adam @ 2e-5, gradient accumulation every 2 steps, EMA decay 0.995 updated
every 10 gradient steps. The inpainting recipe runs 60k gradient steps with
batch 64; deblurring uses 700k. We expose `total_steps` so a class project
can dial things down to whatever fits a single GPU in a reasonable time.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.diffusion.cold import ColdDiffusion
from src.models.ema import EMA


def save_progress_grid(diffusion: ColdDiffusion, ema: EMA, x0_sample: torch.Tensor,
                       t_max: int, out_path: str) -> None:
    from torchvision.utils import save_image  # lazy: torchvision optional at module load

    diffusion_eval = ColdDiffusion(ema.ema_model, diffusion.degradation, diffusion.T)
    diffusion_eval.eval()

    t_full = torch.full((x0_sample.shape[0],), t_max, device=x0_sample.device, dtype=torch.long)
    xT = diffusion_eval.q_sample(x0_sample, t_full)

    with torch.no_grad():
        x0_direct = diffusion_eval.predict_x0(xT, t_full).clamp(0, 1)
        x0_sampled = diffusion_eval.sample_improved(xT).clamp(0, 1)

    grid = torch.cat([xT.clamp(0, 1), x0_direct, x0_sampled, x0_sample], dim=0)
    save_image(grid, out_path, nrow=x0_sample.shape[0])


def train(
    diffusion: ColdDiffusion,
    train_loader,
    *,
    total_steps: int = 60_000,
    accumulate_every: int = 2,
    lr: float = 2e-5,
    ema_decay: float = 0.995,
    ema_update_every: int = 10,
    log_every: int = 100,
    sample_every: int = 2_000,
    save_every: int = 5_000,
    run_dir: str = "results/checkpoints/run",
    device: str = "cuda",
    grad_clip: float | None = 1.0,
) -> EMA:
    diffusion = diffusion.to(device)
    diffusion.train()

    optim = Adam(diffusion.restoration.parameters(), lr=lr)
    ema = EMA(diffusion.restoration, decay=ema_decay, update_every=ema_update_every)
    ema.ema_model.to(device)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # A single fixed batch we visualize every `sample_every` steps.
    fixed_batch, _ = next(iter(train_loader))
    fixed_batch = fixed_batch[:8].to(device)

    # Live progress bar for notebook / terminal. tqdm.auto picks the right
    # frontend (ipywidgets in Jupyter, plain text otherwise).
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    except ImportError:
        pbar = None

    step = 0
    accum = 0
    optim.zero_grad()
    t_start = time.time()
    running = 0.0
    running_n = 0

    while step < total_steps:
        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)

            loss = diffusion.training_loss(x0) / accumulate_every
            loss.backward()
            accum += 1

            if accum == accumulate_every:
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(diffusion.restoration.parameters(), grad_clip)
                optim.step()
                optim.zero_grad()
                accum = 0
                step += 1
                ema.update(diffusion.restoration)

                running += loss.item() * accumulate_every
                running_n += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{running / max(running_n, 1):.4f}")

                if step % log_every == 0:
                    elapsed = time.time() - t_start
                    msg = (
                        f"step {step:>6d}/{total_steps}  "
                        f"loss {running / running_n:.4f}  "
                        f"({step / elapsed:.1f} it/s)"
                    )
                    if pbar is None:
                        print(msg, flush=True)
                    else:
                        pbar.write(msg)
                    running, running_n = 0.0, 0

                if step % sample_every == 0:
                    diffusion.eval()
                    try:
                        save_progress_grid(
                            diffusion, ema, fixed_batch, diffusion.T,
                            str(samples_dir / f"step_{step:06d}.png"),
                        )
                    except Exception as e:
                        print(f"  (skipped progress grid: {e})")
                    diffusion.train()

                if step % save_every == 0 or step == total_steps:
                    torch.save(
                        {
                            "step": step,
                            "model": diffusion.restoration.state_dict(),
                            "ema": ema.state_dict(),
                            "optim": optim.state_dict(),
                        },
                        run_dir / f"ckpt_{step:06d}.pt",
                    )

                if step >= total_steps:
                    break
    if pbar is not None:
        pbar.close()
    return ema


def _build_diffusion(degradation_name: str, image_size: int, device: str) -> ColdDiffusion:
    from src.degradations import GaussianMaskInpainting, GaussianBlur, SuperResolution
    from src.generation.cold_gen import GenerativeInpainting
    from src.models.unet import UNet

    if degradation_name == "inpainting":
        deg = GaussianMaskInpainting(image_size=image_size, T=50)
        T = 50
    elif degradation_name == "generative_inpainting":
        deg = GenerativeInpainting(image_size=image_size, T=50)
        T = 50
    elif degradation_name == "blur":
        deg = GaussianBlur(kernel_size=11, T=40, sigma_fn=lambda t: 0.01 * t + 0.35)
        T = 40
    elif degradation_name == "super_resolution":
        deg = SuperResolution(image_size=image_size, T=3)
        T = 3
    else:
        raise ValueError(degradation_name)

    unet = UNet(image_size=image_size)
    return ColdDiffusion(unet, deg, T=T).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--degradation",
        choices=["inpainting", "generative_inpainting", "blur", "super_resolution"],
        default="inpainting",
    )
    ap.add_argument("--total-steps", type=int, default=60_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.degradation}_cifar10"
    run_dir = os.path.join("results", "checkpoints", run_name)

    from src.training.dataset import cifar10_loaders

    train_loader, _ = cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )

    diffusion = _build_diffusion(args.degradation, image_size=32, device=device)
    train(
        diffusion,
        train_loader,
        total_steps=args.total_steps,
        lr=args.lr,
        run_dir=run_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
