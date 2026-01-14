from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 2) Building blocks
# -------------------------
def _conv3x3(in_ch: int, out_ch: int, bias: bool) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)

class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "nonorm", act: str = "relu"):
        super().__init__()
        if norm == 'nonorm':
            self.conv = _conv3x3(in_ch, out_ch, bias=True)
        else:
            self.conv = _conv3x3(in_ch, out_ch, bias=False)
        
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            # 8 groups is a safe default; adjust if out_ch small
            g = 8 if out_ch >= 8 else 1
            self.norm = nn.GroupNorm(g, out_ch)
        elif norm == "nonorm":
            self.norm = nn.Identity()
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
    def __init__(self, ch: int, norm: str = "norm", act: str = "relu",):
        super().__init__()
        self.c1 = ConvNormAct(ch, ch, norm=norm, act=act)
        if norm == 'nonorm':
            self.c2 = _conv3x3(ch, ch, bias=True)
        else:
            self.c2 = _conv3x3(ch, ch, bias=False)
        
        if norm == "bn":
            self.n2 = nn.BatchNorm2d(ch)
        elif norm == "gn":
            g = 8 if ch >= 8 else 1
            self.n2 = nn.GroupNorm(g, ch)
        elif norm == "nonorm":
            self.n2 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x) # 卷积+激活
        y = self.n2(self.c2(y)) # 卷积
        return self.act(x + y) # 跳跃连接后+激活

 
class DownBlock(nn.Module):
    """Downsample + a few residual blocks"""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 1, norm: str = "norm", act: str = "silu"):
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 1, norm: str = "norm", act: str = "silu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = ConvNormAct(in_ch + skip_ch, out_ch, norm=norm, act=act)
        self.res = nn.Sequential(*[ResidualBlock(out_ch, norm=norm, act=act) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # In case of odd sizes (shouldn't happen with 256), align by center crop
        # if x.shape[-2:] != skip.shape[-2:]:
        #     dh = skip.shape[-2] - x.shape[-2]
        #     dw = skip.shape[-1] - x.shape[-1]
        #     x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
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
        norm: str = "norm",
        act: str = "relu",
    ):
        """
        depth=4 with 256x256 -> bottleneck at 16x16
        """
        super().__init__()
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
            skip_ch = skips_ch[-(i + 1)]  # reverse, skip channels excluding current ch
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
            skip = skips.pop()  # take the corresponding skip
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
        in_ch: int = 25,
        base_ch: int = 32,
        depth: int = 4,
        n_res: int = 1,
        norm: str = "nonorm",
        act: str = "relu",
        filecnt_mask: bool = True,
        filecnt_threshold: float = 0.8,   # file_cnt <= threshold -> treat as unavailable
        filecnt_logits_penalty: float = 1e9,  # large number to suppress logits
    ):
        super().__init__()
        self.filecnt_mask = filecnt_mask
        self.filecnt_threshold = filecnt_threshold
        self.filecnt_logits_penalty = filecnt_logits_penalty
        if self.filecnt_mask:
            print(f"filecnt masking is ENABLED with threshold={filecnt_threshold} and penalty={filecnt_logits_penalty:.1e}")

        self.backbone = UNetBackbone(
            in_ch=in_ch,
            base_ch=base_ch,
            depth=depth,
            n_res=n_res,
            norm=norm,
            act=act,
        )
        self.head = WeightHead(self.backbone.out_ch, k_radars=5)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: (B, K, H, W)
            r_hat  : (B, 1, H, W)
            logits : (B, K, H, W)  (useful for debugging)
        """
        B, C, H, W = x.shape

        feat = self.backbone(x)
        logits = self.head(feat)  # (B,K,H,W)

        # Optional gating: if a radar has file_count==0 in a pixel/patch-hour, suppress its logits
        if self.filecnt_mask:
            filecnt = x[:,[1,6,11,16,21]]  # (B,K,H,W)
            unavailable = (filecnt <= self.filecnt_threshold).to(logits.dtype)  # 1 means unavailable
            logits = logits - unavailable * self.filecnt_logits_penalty

        weights = F.softmax(logits, dim=1)  # along radar dimension

        # Fuse using the rain channels only
        r_stack = x[:, [0, 5, 10, 15, 20], :, :] # (B,K,H,W)
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
    model = RadarFusionWeightNet(base_ch=32, depth=4, n_res=1, norm="nonorm", act="relu",)

    x = torch.randn(2, 25, 256, 256)
    weights, r_hat, logits = model(x)
    print("weights:", weights.shape, "r_hat:", r_hat.shape, "logits:", logits.shape)
    print("weights sum (should be ~1):", weights.sum(dim=1).mean().item())
