from torch.utils.data import ConcatDataset
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset

# ========================================
# Module: 自定义损失函数
# ========================================


class MaskedMSELoss(nn.Module):
    """
    稀疏监督 MSE：只在 gauge_mask==1 的像素上计算 MSE。
    - gauge_grid: 只有雨量计像素有值，其余为0
    - gauge_mask: 雨量计像素=1，其余=0（并可叠加“该小时有效观测”）
    """
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        r_hat: torch.Tensor,
        gauge_grid: torch.Tensor,
        gauge_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 形状统一到 (B,1,H,W)
        if r_hat.dim() == 3:
            r_hat = r_hat.unsqueeze(1)
        if gauge_grid.dim() == 3:
            gauge_grid = gauge_grid.unsqueeze(1)
        if gauge_mask.dim() == 3:
            gauge_mask = gauge_mask.unsqueeze(1)

        # 类型/设备对齐
        gauge_grid = gauge_grid.to(dtype=r_hat.dtype, device=r_hat.device)
        gauge_mask = gauge_mask.to(dtype=r_hat.dtype, device=r_hat.device)

        # 逐像素平方误差
        se = (r_hat - gauge_grid) ** 2

        # mask：非雨量计像素误差置0
        se = se * gauge_mask

        if self.reduction == "sum":
            return se.sum()

        # mean：只对有效像素求平均（关键）
        denom = gauge_mask.sum().clamp_min(self.eps)
        return se.sum() / denom
    
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W) 或 (B,H,W) 都行
    if x.dim() == 3:
        x = x.unsqueeze(1)
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dx + dy


class MaskedMSELoss_cor(nn.Module):
    """
    总损失 = MaskedMSE(r_hat) + lam_c * E[(c-1)^2] + lam_tv * TV(c)

    用法：
      loss = loss_fn(r_hat, gauge_grid, gauge_mask, c=c)

    约定：
      - r_hat: (B,1,H,W) 或 (B,H,W)
      - gauge_grid: (B,1,H,W)/(B,H,W)
      - gauge_mask: (B,1,H,W)/(B,H,W)
      - c: (B,1,H,W)/(B,H,W) 订正场
    """
    def __init__(
        self,
        lam_c: float = 1e-3,
        lam_tv: float = 0.0,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        self.base = MaskedMSELoss(eps=eps, reduction=reduction)
        self.lam_c = lam_c
        self.lam_tv = lam_tv

    def forward(
        self,
        r_hat: torch.Tensor,
        gauge_grid: torch.Tensor,
        gauge_mask: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        loss_g = self.base(r_hat, gauge_grid, gauge_mask)

        if c.dim() == 3:
            c_ = c.unsqueeze(1)
        else:
            c_ = c

        loss_c = ((c_ - 1.0) ** 2).mean()
        loss = loss_g + self.lam_c * loss_c

        if self.lam_tv > 0:
            loss = loss + self.lam_tv * tv_loss(c_)

        return loss


class WeightedMaskedMSELoss(nn.Module):
    """
    分段加权的稀疏监督 MSE loss
    - r_hat:   (B,1,H,W) or (B,H,W)
    - gauge:   (B,1,H,W) or (B,H,W)，已归一化到 [0,1]
    - mask:    (B,1,H,W) or (B,H,W)，雨量计有效像素=1
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

        # 分段定义
        edge = np.arange(0.0, 1.01, 0.1)      # [0, 0.1, ..., 1.0]
        weight = edge * 10.0                  # [0, 1, 2, ..., 10]

        # 注册为 buffer，自动跟随 device
        self.register_buffer(
            "edge", torch.tensor(edge, dtype=torch.float32)
        )
        self.register_buffer(
            "weight", torch.tensor(weight, dtype=torch.float32)
        )

    def forward(self, r_hat, gauge, mask):
        # shape 对齐
        if r_hat.dim() == 3:
            r_hat = r_hat.unsqueeze(1)
        if gauge.dim() == 3:
            gauge = gauge.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 只在 mask=1 的位置计算
        valid = mask > 0
        if valid.sum() == 0:
            return torch.tensor(
                0.0, device=r_hat.device, requires_grad=True
            )

        # --- 分段权重 ---
        # torch.bucketize: 找 gauge 属于哪个区间
        # 返回 index ∈ [0, len(edge)-1]
        idx = torch.bucketize(gauge, self.edge, right=True) - 1
        idx = idx.clamp(min=0, max=len(self.weight) - 1)

        w = self.weight[idx]                  # (B,1,H,W)

        # --- 加权 MSE ---
        loss = w * (r_hat - gauge) ** 2
        loss = loss[valid]

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


# ========================================
# Module: 数据集
# ========================================

class RadarPatchDataset(Dataset):
    """
    每个分片文件：
      - dataset-xxx.npy: (N, C, 256, 256)  建议 C=27: [25输入, 1 gauge_grid, 1 gauge_mask]
      - index_dataset-xxx.csv: N行索引（可用于筛选、记录时间/patch位置等）
    """
    def __init__(self, npy_path: str, index_csv: str, require_gauge: bool = True):
        self.npy_path = npy_path
        self.index_csv = index_csv
        self.idx = pd.read_csv(index_csv)
        self.arr = np.load(npy_path, mmap_mode="r")  # 内存映射，省内存


        # 简单一致性检查
        assert len(self.idx) == self.arr.shape[0], f"index与npy样本数不一致: {len(self.idx)} vs {self.arr.shape[0]}"
        assert self.arr.ndim == 4 and self.arr.shape[1:] == (26, 256, 256) # 5个雷达，每个5通道，加2个gauge_grid

        self.require_gauge = require_gauge
        if self.require_gauge:
            # 只保留 mask 有站点的样本（假设最后一层是 gauge_mask）
            mask_sum = self.arr[:, -1].reshape(self.arr.shape[0], -1).sum(axis=1)
            self.keep = np.where(mask_sum > 0)[0]
        else:
            self.keep = np.arange(self.arr.shape[0])

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        j = self.keep[i]
        patch = self.arr[j]  # (C,256,256)

        # 约定：前25通道是输入，倒数2/1是 gauge_grid / gauge_mask
        x = patch[:25].astype(np.float32)
        gauge_grid = patch[25].astype(np.float32)      # (256,256)
        # gauge_mask = patch[26].astype(np.float32)      # (256,256)

        return (
            torch.from_numpy(x),                       # (25,256,256)
            torch.from_numpy(gauge_grid),              # (256,256)
            # torch.from_numpy(gauge_mask),              # (256,256)
        )
    

def build_concat_dataset(dataset_dir="dataset_files", require_gauge=True, task='2019no09'):
    """
    task 可选：
      - '2019': 只用2019年的数据
      - '1718': 只用2017和2018年的数据
      - '2019no09': 用2019年的数据，但不含09月
      - 其他: 全部数据
    """
    files = sorted(os.listdir(dataset_dir))
    if task == '2019':
        npys = [f for f in files if f.startswith("dataset-") and f.endswith(".npy") and '2019' in f] # 2019年的数据
    elif task == '1718':
        npys = [f for f in files if f.startswith("dataset-") and f.endswith(".npy") and '2019' not in f] # 非2019年的数据
    elif task == '2019no09':
        npys = [f for f in files if f.startswith("dataset-") and f.endswith(".npy") and '201909' not in f and '2019' in f] # 2019, 但不含09月
    else:
        npys = [f for f in files if f.startswith("dataset-") and f.endswith(".npy")]
    datasets = []
    for npy in npys:
        tag = npy.replace("dataset-", "").replace(".npy", "")
        csv = f"index_dataset-{tag}.csv"
        npy_path = os.path.join(dataset_dir, npy)
        csv_path = os.path.join(dataset_dir, csv)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        datasets.append(RadarPatchDataset(npy_path, csv_path, require_gauge=require_gauge))
    return ConcatDataset(datasets)

# ========================================
# Module: 训练和测试
# ========================================


def train_one_epoch(model, loader, optimizer, loss_func, device):
    model.train()
    total_loss = 0.0
    n = 0
    ls_pred = []
    ls_true = []
    for x, gauge_grid in loader:
        x = x.to(device)                                  # (B,25,256,256)
        gauge_grid = gauge_grid.to(device)                # (B,256,256)
        # gauge_mask = gauge_mask.to(device)                # (B,256,256)
        gauge_mask = (gauge_grid > 0).to(dtype=torch.float32)  # 动态生成 mask
        x[:,[0,5,10,15,20]] /= 100.0 # 简单归一化雨量通道, 最小值为0，最大值约为100
        gauge_grid /= 100.0


        optimizer.zero_grad(set_to_none=True)

        has_invalid = ~torch.isfinite(x).all() or ~torch.isfinite(gauge_grid).all()
        if has_invalid:
            print("Warning: 输入数据包含无效值，跳过该批次")
            continue
        weights, r_hat, logits, c = model(x)                 # r_hat: (B,1,256,256)
        # r_hat = r_hat.squeeze(1)                          # -> (B,256,256)
        ls_pred.append(r_hat[:,0][gauge_mask==1].detach().cpu().numpy())
        ls_true.append(gauge_grid[gauge_mask==1].detach().cpu().numpy())

        loss = loss_func(r_hat, gauge_grid, gauge_mask, c=c)   # 你的loss若要求B1HW就别squeeze
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    ls_pred = np.concatenate(ls_pred, axis=0)
    ls_true = np.concatenate(ls_true, axis=0)
    return total_loss / max(n, 1), ls_true, ls_pred

@torch.no_grad()
def eval_one_epoch(model, loader, loss_func, device):
    model.eval()
    total_loss = 0.0
    n = 0
    ls_pred = []
    ls_true = []
    for x, gauge_grid in loader:
        x = x.to(device)
        gauge_grid = gauge_grid.to(device)
        # gauge_mask = gauge_mask.to(device)
        gauge_mask = (gauge_grid > 0).to(dtype=torch.float32)
        x[:,[0,5,10,15,20]] /= 100.0 # 简单归一化雨量通道, 最小值为0，最大值约为100
        gauge_grid /= 100.0

        weights, r_hat, logits, c = model(x)
        # r_hat = r_hat.squeeze(1)
        ls_pred.append(r_hat[:,0][gauge_mask==1].detach().cpu().numpy())
        ls_true.append(gauge_grid[gauge_mask==1].detach().cpu().numpy())

        loss = loss_func(r_hat, gauge_grid, gauge_mask, c=c)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    ls_pred = np.concatenate(ls_pred, axis=0)
    ls_true = np.concatenate(ls_true, axis=0)
    return total_loss / max(n, 1), ls_true, ls_pred

