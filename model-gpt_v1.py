# -*- coding: utf-8 -*-
"""
Modular UNet for radar-fusion weight learning
Input : (B, 25, H, W)  where 25 = 5 radars * 5 vars
Output: weights (B, K, H, W) and fused rainfall (B, 1, H, W)

Default channel layout per radar (repeat K times):
    [R_1h, file_cnt_1h, rain_cnt_1h, hgt_rel_ground, dist]
So for radar k (0-based), its base index = k*5.

You can change layout by editing RadarChannelSpec below.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Channel spec (clean separation of "what is where")
# -------------------------
@dataclass(frozen=True)
class RadarChannelSpec:
    k_radars: int = 5
    vars_per_radar: int = 5

    # offsets within each radar block
    off_rain: int = 0
    off_filecnt: int = 1
    off_raincnt: int = 2
    off_hgt: int = 3
    off_dist: int = 4

    @property
    def in_channels(self) -> int:
        return self.k_radars * self.vars_per_radar

    def idx(self, radar_k: int, offset: int) -> int:
        return radar_k * self.vars_per_radar + offset

    def rain_indices(self) -> torch.Tensor:
        # shape (K,)
        return torch.tensor([self.idx(k, self.off_rain) for k in range(self.k_radars)], dtype=torch.long)

    def filecnt_indices(self) -> torch.Tensor:
        # shape (K,)
        return torch.tensor([self.idx(k, self.off_filecnt) for k in range(self.k_radars)], dtype=torch.long)


# -------------------------
# 2) Building blocks
# -------------------------
def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.conv = _conv3x3(in_ch, out_ch)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            # 8 groups is a safe default; adjust if out_ch small
            g = 8 if out_ch >= 8 else 1
            self.norm = nn.GroupNorm(g, out_ch)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Light residual block: (ConvNormAct -> ConvNorm) + skip"""
    def __init__(self, ch: int, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.c1 = ConvNormAct(ch, ch, norm=norm, act=act)
        self.c2 = _conv3x3(ch, ch)
        if norm == "bn":
            self.n2 = nn.BatchNorm2d(ch)
        elif norm == "gn":
            g = 8 if ch >= 8 else 1
            self.n2 = nn.GroupNorm(g, ch)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.n2(self.c2(y))
        return self.act(x + y)


class DownBlock(nn.Module):
    """Downsample + a few residual blocks"""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 1, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.proj = ConvNormAct(in_ch, out_ch, norm=norm, act=act)
        self.res = nn.Sequential(*[ResidualBlock(out_ch, norm=norm, act=act) for _ in range(n_res)])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.res(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsample + concat skip + residual blocks"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 1, norm: str = "bn", act: str = "silu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = ConvNormAct(in_ch + skip_ch, out_ch, norm=norm, act=act)
        self.res = nn.Sequential(*[ResidualBlock(out_ch, norm=norm, act=act) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # In case of odd sizes (shouldn't happen with 256), align by center crop
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.res(x)
        return x


# -------------------------
# 3) UNet backbone (pure feature extractor)
# -------------------------
class UNetBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 32,
        depth: int = 4,
        n_res: int = 1,
        norm: str = "bn",
        act: str = "silu",
    ):
        """
        depth=4 with 256x256 -> bottleneck at 16x16
        """
        super().__init__()
        assert depth >= 2, "depth should be >=2"
        self.stem = ConvNormAct(in_ch, base_ch, norm=norm, act=act)

        down_blocks = []
        ch = base_ch
        skips_ch = [base_ch]
        for _ in range(depth - 1):
            down_blocks.append(DownBlock(ch, ch * 2, n_res=n_res, norm=norm, act=act))
            ch *= 2
            skips_ch.append(ch)
        self.down = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(
            ConvNormAct(ch, ch, norm=norm, act=act),
            ResidualBlock(ch, norm=norm, act=act),
        )

        up_blocks = []
        # we will go back depth-1 times
        for i in range(depth - 1):
            skip_ch = skips_ch[-(i + 2)]  # reverse, skip channels excluding current ch
            up_blocks.append(UpBlock(ch, skip_ch, ch // 2, n_res=n_res, norm=norm, act=act))
            ch //= 2
        self.up = nn.ModuleList(up_blocks)

        self.out_ch = ch  # final feature channels == base_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = [x]
        for blk in self.down:
            x, s = blk(x)
            skips.append(s)
        x = self.bottleneck(x)
        # pop last skip (deepest) already used implicitly; we want to pair from second last
        for blk in self.up:
            skip = skips.pop(-2)  # take the corresponding skip
            x = blk(x, skip)
        return x


# -------------------------
# 4) Weight head + fusion wrapper
# -------------------------
class WeightHead(nn.Module):
    def __init__(self, feat_ch: int, k_radars: int):
        super().__init__()
        self.conv = nn.Conv2d(feat_ch, k_radars, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)  # logits


class RadarFusionWeightNet(nn.Module):
    """
    End-to-end differentiable:
        X -> logits -> weights(softmax) -> R_hat = sum(w_k * R_k)
    """
    def __init__(
        self,
        spec: RadarChannelSpec = RadarChannelSpec(),
        base_ch: int = 32,
        depth: int = 4,
        n_res: int = 1,
        norm: str = "bn",
        act: str = "silu",
        filecnt_gate: bool = True,
        gate_threshold: float = 0.0,   # file_cnt <= threshold -> treat as unavailable
        gate_logits_penalty: float = 1e4,  # large number to suppress logits
    ):
        super().__init__()
        self.spec = spec
        self.filecnt_gate = filecnt_gate
        self.gate_threshold = gate_threshold
        self.gate_logits_penalty = gate_logits_penalty

        self.backbone = UNetBackbone(
            in_ch=spec.in_channels,
            base_ch=base_ch,
            depth=depth,
            n_res=n_res,
            norm=norm,
            act=act,
        )
        self.head = WeightHead(self.backbone.out_ch, spec.k_radars)

        # register indices as buffers for device-safe usage
        self.register_buffer("rain_idx", spec.rain_indices(), persistent=False)      # (K,)
        self.register_buffer("filecnt_idx", spec.filecnt_indices(), persistent=False)  # (K,)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: (B, K, H, W)
            r_hat  : (B, 1, H, W)
            logits : (B, K, H, W)  (useful for debugging)
        """
        assert x.dim() == 4, f"Expect (B,C,H,W), got {x.shape}"
        B, C, H, W = x.shape
        assert C == self.spec.in_channels, f"Expect C={self.spec.in_channels}, got {C}"

        feat = self.backbone(x)
        logits = self.head(feat)  # (B,K,H,W)

        # Optional gating: if a radar has file_count==0 in a pixel/patch-hour, suppress its logits
        if self.filecnt_gate:
            filecnt = torch.index_select(x, dim=1, index=self.filecnt_idx)  # (B,K,H,W)
            unavailable = (filecnt <= self.gate_threshold).to(logits.dtype)  # 1 means unavailable
            logits = logits - unavailable * self.gate_logits_penalty

        weights = F.softmax(logits, dim=1)  # along radar dimension

        # Fuse using the rain channels only
        r_stack = torch.index_select(x, dim=1, index=self.rain_idx)  # (B,K,H,W)
        r_hat = torch.sum(weights * r_stack, dim=1, keepdim=True)    # (B,1,H,W)
        return weights, r_hat, logits


# -------------------------
# 5) (Optional) a helper loss for sparse gauges (you can use or ignore)
# -------------------------
def masked_gauge_loss(
    r_hat: torch.Tensor,
    gauge_mask: torch.Tensor,
    gauge_target: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    r_hat       : (B,1,H,W) predicted hourly rainfall
    gauge_mask  : (B,1,H,W) 1 at gauge pixels (AND valid obs at that hour), else 0
    gauge_target: (B,1,H,W) target hourly rainfall at gauge pixels, elsewhere 0 (or anything)
    Returns mean loss over valid gauge pixels per batch.
    """
    assert r_hat.shape == gauge_mask.shape == gauge_target.shape
    if loss_type == "mse":
        per = (r_hat - gauge_target) ** 2
    elif loss_type == "mae":
        per = torch.abs(r_hat - gauge_target)
    elif loss_type == "huber":
        per = F.huber_loss(r_hat, gauge_target, delta=huber_delta, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    per = per * gauge_mask
    denom = torch.sum(gauge_mask) + eps
    return torch.sum(per) / denom


# -------------------------
# 6) quick sanity check
# -------------------------
if __name__ == "__main__":
    spec = RadarChannelSpec(k_radars=5, vars_per_radar=5)
    model = RadarFusionWeightNet(spec=spec, base_ch=32, depth=4, n_res=1, norm="bn", act="silu")

    x = torch.randn(2, 25, 256, 256)
    weights, r_hat, logits = model(x)
    print("weights:", weights.shape, "r_hat:", r_hat.shape, "logits:", logits.shape)
    print("weights sum (should be ~1):", weights.sum(dim=1).mean().item())
