"""Time-conditional U-Net for 32x32 images (CIFAR-10).

Architecture follows the standard DDPM/cold-diffusion recipe: GroupNorm
residual blocks, sinusoidal time embedding projected through a small MLP,
self-attention at the lowest spatial resolution, three down/up stages.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8, heads: int = 4):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * (c // self.heads) ** -0.5, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """3-level U-Net suitable for 32x32 images.

    base_ch=64, ch_mults=(1,2,2) gives ~12M params, fast to train on CIFAR-10.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        ch_mults=(1, 2, 2),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        attn_resolutions=(8,),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_emb_dim_in = base_ch

        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        chs = [base_ch]
        cur_ch = base_ch
        cur_res = image_size
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                blocks = [ResBlock(cur_ch, out_ch, time_dim, dropout=dropout)]
                cur_ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(SelfAttention(cur_ch))
                self.down_blocks.append(nn.ModuleList(blocks))
                chs.append(cur_ch)
            if i != len(ch_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Down(cur_ch)]))
                chs.append(cur_ch)
                cur_res //= 2

        # Mid
        self.mid1 = ResBlock(cur_ch, cur_ch, time_dim, dropout=dropout)
        self.mid_attn = SelfAttention(cur_ch)
        self.mid2 = ResBlock(cur_ch, cur_ch, time_dim, dropout=dropout)

        # Up path (mirror of down, with skip concat)
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                blocks = [ResBlock(cur_ch + skip_ch, out_ch, time_dim, dropout=dropout)]
                cur_ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(SelfAttention(cur_ch))
                self.up_blocks.append(nn.ModuleList(blocks))
            if i != 0:
                self.up_blocks.append(nn.ModuleList([Up(cur_ch)]))
                cur_res *= 2

        self.out_norm = nn.GroupNorm(8, cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim_in))

        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for block in self.up_blocks:
            first = block[0]
            if isinstance(first, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                for layer in block:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                for layer in block:
                    h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))
