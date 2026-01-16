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
    

def build_concat_dataset(dataset_dir="dataset_files", require_gauge=True):
    files = sorted(os.listdir(dataset_dir))
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
    for x, gauge_grid in loader:
        x = x.to(device)                                  # (B,25,256,256)
        gauge_grid = gauge_grid.to(device)                # (B,256,256)
        # gauge_mask = gauge_mask.to(device)                # (B,256,256)
        gauge_mask = (gauge_grid > 0).to(dtype=torch.float32)  # 动态生成 mask
        x[:,[0,5,10,15,20]] /= 200.0 # 简单归一化雨量通道, 最小值为0，最大值约为200
        gauge_grid /= 200.0


        optimizer.zero_grad(set_to_none=True)

        has_invalid = ~torch.isfinite(x).all() or ~torch.isfinite(gauge_grid).all()
        if has_invalid:
            print("Warning: 输入数据包含无效值，跳过该批次")
            continue
        weights, r_hat, logits = model(x)                 # r_hat: (B,1,256,256)
        # r_hat = r_hat.squeeze(1)                          # -> (B,256,256)

        loss = loss_func(r_hat, gauge_grid, gauge_mask)   # 你的loss若要求B1HW就别squeeze
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_func, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, gauge_grid in loader:
        x = x.to(device)
        gauge_grid = gauge_grid.to(device)
        # gauge_mask = gauge_mask.to(device)
        gauge_mask = (gauge_grid > 0).to(dtype=torch.float32)
        x[:,[0,5,10,15,20]] /= 200.0 # 简单归一化雨量通道, 最小值为0，最大值约为200
        gauge_grid /= 200.0

        weights, r_hat, logits = model(x)
        # r_hat = r_hat.squeeze(1)

        loss = loss_func(r_hat, gauge_grid, gauge_mask)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

