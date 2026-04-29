"""Exponential moving average of model weights.

The Cold Diffusion paper maintains an EMA of the U-Net with decay 0.995,
updated every 10 gradient steps. The EMA copy is what gets used at sampling
time — it produces visibly cleaner results than the raw trained weights.
"""

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995, update_every: int = 10):
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        if self.step % self.update_every != 0:
            return
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return {"step": self.step, "model": self.ema_model.state_dict()}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.ema_model.load_state_dict(state["model"])
