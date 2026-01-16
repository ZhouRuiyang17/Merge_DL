import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MODELPATH = f'./models/{datetime.datetime.now().strftime("%Y%m%d_%H%M")}/'
os.makedirs(MODELPATH, exist_ok=True)

# ========================================
# Module: 加载模型
# ========================================
from model import RadarFusionWeightNet
model = RadarFusionWeightNet(base_ch=32, depth=4, n_res=1, norm="nonorm", act="relu",).to(device)

# ========================================
# Module: 优化器和损失函数
# ========================================
opmizer = torch.optim.Adam(model.parameters(), lr=1e-4)
from DLtools import MaskedMSELoss
loss_func = MaskedMSELoss()

# ========================================
# Module: 加载数据集
# ========================================
from DLtools import build_concat_dataset, train_one_epoch, eval_one_epoch

t1 = datetime.datetime.now()
dataset = build_concat_dataset("dataset_files", require_gauge=True)
print(f"数据集样本数: {len(dataset)}")
print(f"加载数据集用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

# 简单随机划分（你后面要 LOEO 就换成按事件分片）
n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
t1 = datetime.datetime.now()
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
print(f"划分数据集用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

t1 = datetime.datetime.now()
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
print(f"创建数据加载器用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

best = 1e18
ls_tr_loss = []
ls_va_loss = []
for epoch in range(1, 51):
    tr = train_one_epoch(model, train_loader, opmizer, loss_func, device)
    va = eval_one_epoch(model, val_loader, loss_func, device)
    print(f"Epoch {epoch:03d} | train={tr:.6f} | val={va:.6f}")
    ls_tr_loss.append(tr)
    ls_va_loss.append(va)
    if epoch % 5 == 0:
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va}, os.path.join(MODELPATH, f"epoch{epoch:03d}.pt"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.clear()
        ax.plot(ls_tr_loss, label="train")
        ax.plot(ls_va_loss, label="val")
        ax.legend()
        fig.savefig(os.path.join(MODELPATH, "loss_curve.png"))
        plt.close()

    # 保存最好模型
    if va < best:
        best = va
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va}, os.path.join(MODELPATH, "best.pt"))
