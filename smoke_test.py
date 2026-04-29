"""Smoke tests: every module imports, every component produces correct shapes,
the training loop runs for a few steps without errors, and Algorithm 1/2 both
produce sensible outputs from random init.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from src.models.unet import UNet
from src.models.ema import EMA
from src.degradations import GaussianMaskInpainting, GaussianBlur, SuperResolution
from src.diffusion.cold import ColdDiffusion
from src.generation.cold_gen import GenerativeInpainting, sample_generative

DEVICE = "cpu"  # smoke test on CPU
torch.manual_seed(0)

B, C, H = 4, 3, 32
x0 = torch.rand(B, C, H, H, device=DEVICE)


def test_unet():
    net = UNet(image_size=H).to(DEVICE)
    t = torch.tensor([0, 10, 25, 49], device=DEVICE)
    out = net(x0, t)
    assert out.shape == x0.shape, f"unet out shape {out.shape}"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  unet ok ({n_params/1e6:.1f}M params)")


def test_degradations():
    for name, deg, T in [
        ("inpainting", GaussianMaskInpainting(image_size=H, T=50), 50),
        ("blur", GaussianBlur(kernel_size=11, T=40), 40),
        ("super_resolution", SuperResolution(image_size=H, T=3), 3),
        ("generative_inpainting", GenerativeInpainting(image_size=H, T=50), 50),
    ]:
        deg = deg.to(DEVICE)
        for tval in [0, T // 2, T]:
            t = torch.full((B,), tval, dtype=torch.long, device=DEVICE)
            xt = deg(x0, t)
            assert xt.shape == x0.shape, f"{name} t={tval} shape {xt.shape}"
            assert torch.isfinite(xt).all(), f"{name} t={tval} non-finite"
            if tval == 0:
                # D(x, 0) should be x for inpainting and blur (and pixelate).
                # Generative inpainting also matches at t=0 since cumulative_mask is 1.
                err = (xt - x0).abs().max().item()
                assert err < 1e-5, f"{name} D(x,0) != x (err {err})"
        print(f"  {name} ok")


def test_cold_diffusion_loss_and_samplers():
    deg = GaussianMaskInpainting(image_size=H, T=50)
    net = UNet(image_size=H)
    diff = ColdDiffusion(net, deg, T=50).to(DEVICE)

    # Forward / loss
    loss = diff.training_loss(x0)
    assert torch.isfinite(loss), "loss not finite"
    loss.backward()
    grads = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    assert grads > 0, "no gradients"
    print(f"  loss {loss.item():.3f}, grads_sum {grads:.1f}")

    diff.eval()
    with torch.no_grad():
        t_full = torch.full((B,), 50, dtype=torch.long, device=DEVICE)
        xT = diff.q_sample(x0, t_full)
        out_naive = diff.sample_naive(xT)
        out_imp = diff.sample_improved(xT)
        assert out_naive.shape == x0.shape
        assert out_imp.shape == x0.shape
        assert torch.isfinite(out_naive).all() and torch.isfinite(out_imp).all()
    print("  Algorithm 1 & 2 ran successfully")


def test_ema():
    net = UNet(image_size=H)
    ema = EMA(net, decay=0.5, update_every=1)
    # Modify net params, ensure EMA tracks them
    for p in net.parameters():
        p.data.add_(1.0)
    ema.update(net)
    diff = sum((a - b).abs().sum().item() for a, b in zip(ema.ema_model.parameters(), net.parameters()))
    assert diff > 0, "EMA didn't update"
    print(f"  ema ok (param diff after one update: {diff:.1f})")


def test_train_one_step():
    from torch.utils.data import TensorDataset, DataLoader

    deg = GaussianMaskInpainting(image_size=H, T=50)
    net = UNet(image_size=H)
    diff = ColdDiffusion(net, deg, T=50)

    # Tiny synthetic loader
    fake = TensorDataset(torch.rand(16, 3, H, H), torch.zeros(16, dtype=torch.long))
    loader = DataLoader(fake, batch_size=8)

    from src.training.train import train

    ema = train(
        diff, loader,
        total_steps=2,
        accumulate_every=1,
        lr=1e-4,
        log_every=1,
        sample_every=10_000,  # don't bother saving images in the smoke test
        save_every=10_000,
        run_dir="results/_smoke_run",
        device=DEVICE,
    )
    assert ema is not None
    print("  train loop ran 2 steps")


def test_state_consistency():
    """Two q_sample calls with the same state must use the same mask."""
    deg = GaussianMaskInpainting(image_size=H, T=50, randomize_center=True)
    net = UNet(image_size=H)
    diff = ColdDiffusion(net, deg, T=50)

    state = diff.sample_state(x0)
    t = torch.full((B,), 25, dtype=torch.long)
    a = diff.q_sample(x0, t, state=state)
    b = diff.q_sample(x0, t, state=state)
    assert torch.allclose(a, b), "same-state q_sample produced different outputs"

    # Without state, two calls should disagree (centers re-sampled).
    c = diff.q_sample(x0, t)
    d = diff.q_sample(x0, t)
    assert not torch.allclose(c, d), "stateless q_sample should re-randomize"
    print("  state consistency ok (with state: identical, without state: differ)")


def test_generation():
    deg = GenerativeInpainting(image_size=H, T=50)
    net = UNet(image_size=H)
    diff = ColdDiffusion(net, deg, T=50).to(DEVICE).eval()
    out = sample_generative(diff, n=2, image_size=H, device=DEVICE)
    assert out.shape == (2, 3, H, H), f"gen shape {out.shape}"
    assert torch.isfinite(out).all()
    print("  cold generation ok")


if __name__ == "__main__":
    print("== U-Net ==");                test_unet()
    print("== Degradations ==");          test_degradations()
    print("== ColdDiffusion ==");         test_cold_diffusion_loss_and_samplers()
    print("== EMA ==");                   test_ema()
    print("== Train loop ==");            test_train_one_step()
    print("== State consistency =="); test_state_consistency()
    print("== Cold generation ==");       test_generation()
    print("\nAll smoke tests passed.")
