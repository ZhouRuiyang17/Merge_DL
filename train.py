import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

fontsize = 12
plt.rcParams['font.size'] = 10        # 默认字体大小
plt.rcParams['xtick.labelsize'] = 10  # X 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # Y 轴刻度字体大小
plt.rcParams['axes.labelsize'] = 12   # 轴标签大小
plt.rcParams['axes.titlesize'] = 12   # 标题大小
plt.rcParams['legend.fontsize'] = 10  # 图例大小


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ========================================
# Module: 加载模型
# ========================================
from model import RadarFusionWeightNet, RadarFusionWeightNet2Head
MODELPATH = f'./models/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(MODELPATH, exist_ok=True)
import logging
logging.basicConfig(
    filename=f'{MODELPATH}/main.log',                  # 日志文件名
    level=logging.INFO,                        # 记录 INFO 及以上级别的日志
    format='%(asctime)s---%(message)s',        # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'                # 时间格式
)
model = RadarFusionWeightNet(base_ch=32, depth=4, n_res=1, norm="nonorm", act="relu",).to(device)
logging.info('Model: RadarFusionWeightNet, base=32, depth=4, n_res=1, norm="nonorm", act="relu"')
# model = RadarFusionWeightNet2Head(base_ch=32, depth=4, n_res=1, norm="nonorm", act="relu",
#                                   c_min=0.5, c_max=2.0, c_head_hidden=16).to(device)
# logging.info('Model: RadarFusionWeightNet2Head, base=32, depth=4, n_res=1, norm="nonorm", act="relu", c_min=0.5, c_max=2.0, c_head_hidden=16')

# ========================================
# Module: 优化器和损失函数
# ========================================
opmizer = torch.optim.Adam(model.parameters(), lr=1e-4)
logging.info('Optimizer: Adam, lr=1e-4')
from DLtools import MaskedMSELoss, WeightedMaskedMSELoss, MaskedMSELoss_cor
loss_func = MaskedMSELoss()
logging.info('Loss: MSE Loss')
# loss_func = MaskedMSELoss_cor()
# logging.info('Loss: MSE Loss + correction loss + tv loss')
# loss_func = WeightedMaskedMSELoss(reduction="mean").to(device)
# logging.info(loss_func.edge)
# logging.info(loss_func.weight)

# ========================================
# Module: 加载数据集
# ========================================
from DLtools import build_concat_dataset, train_one_epoch, eval_one_epoch

t1 = datetime.datetime.now()
dataset = build_concat_dataset("dataset_files", require_gauge=True, task='1718')
logging.info("使用数据集: 2017和2018年的数据. 但是降雨的最大值从200改为100.")
logging.info(f"数据集样本数: {len(dataset)}")
logging.info(f"加载数据集用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

# 简单随机划分（你后面要 LOEO 就换成按事件分片）
n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
t1 = datetime.datetime.now()
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
logging.info(f"划分数据集用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

t1 = datetime.datetime.now()
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
logging.info(f"创建数据加载器用时: {(datetime.datetime.now()-t1).total_seconds():.2f} 秒")

best = 1e18
ls_tr_loss = []
ls_va_loss = []
for epoch in range(1, 201):
    tr, ls_true_tr, ls_pred_tr = train_one_epoch(model, train_loader, opmizer, loss_func, device)
    va, ls_true_va, ls_pred_va = eval_one_epoch(model, val_loader, loss_func, device)
    print(f"Epoch {epoch:03d} | train={tr:.6f} | val={va:.6f}")
    ls_tr_loss.append(tr)
    ls_va_loss.append(va)
    if epoch % 5 == 0:
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va}, os.path.join(MODELPATH, f"epoch{epoch:03d}.pt"))
        fig, ax = plt.subplots(1,3,figsize=(9, 3))
        ax[0].plot(ls_tr_loss, label="train")
        ax[0].plot(ls_va_loss, label="val")
        ax[0].legend()
        ax[1].scatter(ls_true_tr, ls_pred_tr, s=1, alpha=0.1)
        ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].grid(); ax[1].plot([0,1],[0,1],'r--')
        ax[2].scatter(ls_true_va, ls_pred_va, s=1, alpha=0.1)
        ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1); ax[2].grid(); ax[2].plot([0,1],[0,1],'r--')
        fig.savefig(os.path.join(MODELPATH, "loss_curve.png"), dpi = 300)
        plt.close()

    # 保存最好模型
    if va < best:
        best = va
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va}, os.path.join(MODELPATH, "best.pt"))
