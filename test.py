import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
import datetime
import logging

# ========================================
# Module: 用得到的模块
# ========================================
def load_model_from_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt.get("epoch", None), ckpt.get("val_loss", None)
import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def infer_full_1024_sliding(
    model,
    x_full,                      # (25, 1024, 1024) torch or numpy
    window: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    return_numpy: bool = True,
    entropy_normalized: bool = True,   # True: H/log(K) in [0,1]
    eps: float = 1e-12,
):
    """
    滑窗推理整场：
      - 输入: x_full (25, H, W)  (H=W=1024 但也可泛化到任意>=window)
      - 输出:
          r_full      (H, W)            融合后降雨
          w_full      (5, H, W)         权重场
          logits_full (5, H, W)         logits
          max_id      (H, W)            每像素最大权重的雷达id [0..4]
          entropy     (H, W)            权重熵 (可选归一化)
    说明：
      - 使用“重叠区域累加 + count平均”的拼接策略，边缘自然覆盖，无需 center-crop。
      - 假设你的 model.forward(x) 返回 (weights, r_hat, logits)，其中 weights 已 softmax。
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- input to torch (C,H,W) float32 on device
    if isinstance(x_full, np.ndarray):
        x_full = torch.from_numpy(x_full)
    assert torch.is_tensor(x_full), "x_full 必须是 torch.Tensor 或 np.ndarray"
    assert x_full.dim() == 3 and x_full.size(0) == 25, f"期望 (25,H,W)，但得到 {tuple(x_full.shape)}"
    x_full = x_full.to(device=device, dtype=torch.float32)

    C, H, W = x_full.shape
    assert H >= window and W >= window, "H/W 必须 >= window"

    # ---- prepare accumulation buffers
    K = 5
    r_acc = torch.zeros((H, W), device=device, dtype=torch.float32)
    r_cnt = torch.zeros((H, W), device=device, dtype=torch.float32)

    w_acc = torch.zeros((K, H, W), device=device, dtype=torch.float32)
    w_cnt = torch.zeros((H, W), device=device, dtype=torch.float32)

    logit_acc = torch.zeros((K, H, W), device=device, dtype=torch.float32)
    logit_cnt = torch.zeros((H, W), device=device, dtype=torch.float32)

    c_acc = torch.zeros((H, W), device=device, dtype=torch.float32)
    c_cnt = torch.zeros((H, W), device=device, dtype=torch.float32)

    # ---- sliding positions (确保最后一块覆盖边界)
    ys = list(range(0, H - window + 1, stride))
    xs = list(range(0, W - window + 1, stride))
    if ys[-1] != H - window:
        ys.append(H - window)
    if xs[-1] != W - window:
        xs.append(W - window)

    # ---- batch infer
    patches = []
    coords = []

    def _flush_batch():
        if not patches:
            return
        x_batch = torch.stack(patches, dim=0)  # (B,25,window,window)
        weights, r_hat, logits, c = model(x_batch)  # weights/logits: (B,5,win,win), r_hat: (B,1,win,win) or (B,win,win)
        c = c[:, 0]  # (B,)

        if r_hat.dim() == 4:
            r_hat_ = r_hat[:, 0]  # (B,win,win)
        else:
            r_hat_ = r_hat

        B = x_batch.size(0)
        for b in range(B):
            y0, x0 = coords[b]
            y1, x1 = y0 + window, x0 + window

            # r_hat
            r_acc[y0:y1, x0:x1] += r_hat_[b]
            r_cnt[y0:y1, x0:x1] += 1.0

            # weights
            w_acc[:, y0:y1, x0:x1] += weights[b]
            w_cnt[y0:y1, x0:x1] += 1.0

            # logits
            logit_acc[:, y0:y1, x0:x1] += logits[b]
            logit_cnt[y0:y1, x0:x1] += 1.0

            # correct coefficient c
            c_acc[y0:y1, x0:x1] += c[b]
            c_cnt[y0:y1, x0:x1] += 1.0

        patches.clear()
        coords.clear()

    for y0 in ys:
        for x0 in xs:
            patch = x_full[:, y0:y0 + window, x0:x0 + window]
            patches.append(patch)
            coords.append((y0, x0))
            if len(patches) >= batch_size:
                _flush_batch()
    _flush_batch()

    # ---- average overlaps
    r_full = r_acc / r_cnt.clamp_min(1.0)
    w_full = w_acc / w_cnt.clamp_min(1.0)
    logits_full = logit_acc / logit_cnt.clamp_min(1.0)
    c_full = c_acc / c_cnt.clamp_min(1.0)

    # ---- derived: max weight id & entropy
    max_id = torch.argmax(w_full, dim=0).to(torch.int16)  # (H,W)

    # Shannon entropy: -sum w log w
    w_safe = w_full.clamp_min(eps)
    entropy = -(w_safe * torch.log(w_safe)).sum(dim=0)  # (H,W), natural log
    if entropy_normalized:
        entropy = entropy / math.log(K)  # [0,1] 理论范围

    if return_numpy:
        return (
            r_full.detach().cpu().numpy(),              # (H,W)
            w_full.detach().cpu().numpy(),              # (5,H,W)
            logits_full.detach().cpu().numpy(),          # (5,H,W)
            max_id.detach().cpu().numpy(),               # (H,W)
            entropy.detach().cpu().numpy(),              # (H,W)
            c_full.detach().cpu().numpy(),              # (H,W)
        )
    else:
        return r_full, w_full, logits_full, max_id, entropy, c_full


def run(model, x_full_25hw, device="cuda"):
    device = torch.device(device)
    

    x_full = torch.as_tensor(x_full_25hw, dtype=torch.float32)  # (25,1024,1024)
    # 建议：推理前做和训练一致的归一化/clip（你现在是线性到[0,1]）
    x_full = x_full.to(device)
    x_full[[0,5,10,15,20],:,:] = x_full[[0,5,10,15,20],:,:] / 100.0

    r_full, w_full, logits_full, max_id, entropy, c_full = infer_full_1024_sliding(
        model, x_full,  window=256, stride=128, batch_size=8
    )
    r_full = r_full * 100.0  # 恢复到实际雨量范围

    return r_full, w_full, logits_full, max_id, entropy, c_full

def save_full_scene_npz(
    out_path,
    R_k,          # (5,H,W) numpy
    W_k,          # (5,H,W) numpy
    logits_k,    # (5,H,W) numpy
    R_hat,        # (H,W) numpy
    max_id,
    entropy,
    c_full,        # (H,W) numpy
    filecnt_k=None,
    rainfreq_k=None,
    meta=None,
):
    """
    out_path: xxx.npz
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    save_dict = {
        "R_k": R_k,
        "W_k": W_k,
        "logits_k": logits_k,
        "R_hat": R_hat,
        "entropy": entropy,
        "max_id": max_id,
        "correct_coeff": c_full,
    }

    if filecnt_k is not None:
        save_dict["filecnt_k"] = filecnt_k
    if rainfreq_k is not None:
        save_dict["rainfreq_k"] = rainfreq_k
    if meta is not None:
        save_dict["meta"] = meta

    np.savez_compressed(out_path, **save_dict)

def load_gridinfo(radarnames):
    dict_gridinfo = {radarname: np.load(f'gridinfo/Merge_DL-0.001deg/gridinfo_{radarname}.npz') for radarname in radarnames}
    return dict_gridinfo

def polar_to_cartesian(data, gridinfo, radarname, timestamp):
    acc = data['acc']
    num_file = data['num_file']
    num_rain = data['num_rain']
    ratio_file = num_file / 10 # 按照6min扫描间隔, 每小时应该有10个文件
    ratio_rain = num_rain / num_file # 在有文件的地方, 多少比例有雨
    mask_invalid = ~(np.isfinite(acc) & np.isfinite(ratio_file) & np.isfinite(ratio_rain))
    if mask_invalid.any():
        print(f"Warning: 发现无效值 at {radarname} {timestamp}")
    acc[mask_invalid] = 0.0
    ratio_file[mask_invalid] = 0.0
    ratio_rain[mask_invalid] = 0.0

    grid_agl = gridinfo['grid_AGL']/1000  # m to km
    grid_sr = gridinfo['grid_sr']/1000  # m to km
    L = 100  # 100km
    H = 2  # 2km
    grid_sr_wt = np.exp(-grid_sr**2/L**2) 
    grid_agl_wt = np.exp(-grid_agl**2/H**2)
    loc = ~np.isfinite(grid_sr_wt) | ~np.isfinite(grid_agl_wt)  # 无效点: 任意点为NaN或Inf
    grid_sr_wt[loc] = 0.0
    grid_agl_wt[loc] = 0.0

    grid_az = gridinfo['grid_az']
    grid_gt = gridinfo['grid_gt']
    loc = ~np.isfinite(grid_az) | ~np.isfinite(grid_gt)  # 无效点: 任意点为NaN或Inf
    idaz = grid_az[~loc].astype(int)
    idgt = grid_gt[~loc].astype(int)
    acc_grid = np.zeros_like(grid_az, dtype=np.float32)
    acc_grid[~loc] = acc[idaz, idgt]
    ratio_file_grid = np.zeros_like(grid_az, dtype=np.float32)
    ratio_file_grid[~loc] = ratio_file[idaz, idgt]
    ratio_rain_grid = np.zeros_like(grid_az, dtype=np.float32)
    ratio_rain_grid[~loc] = ratio_rain[idaz, idgt]
    """ from utils.information import area_Merge_DL
    MOSAIC_AREA = area_Merge_DL()[0]
    fig, ax = plt.subplots(3, 3, figsize=(15,15), subplot_kw={'projection': ccrs.PlateCarree()})
    ax = ax.flatten()
    et.RADAR(acc,'acc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, ax=ax[0], area=MOSAIC_AREA)
    et.RADAR(ratio_file,'cc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, ax=ax[1], area=MOSAIC_AREA)
    et.RADAR(ratio_rain,'cc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, ax=ax[2], area=MOSAIC_AREA)
    grid_lon = gridinfo['grid_lon']
    grid_lat = gridinfo['grid_lat']
    cmap, norm, _, _ = et.colorbar('acc')
    ax[3].pcolormesh(grid_lon, grid_lat, acc_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cmap, norm, _, _ = et.colorbar('cc')
    ax[4].pcolormesh(grid_lon, grid_lat, ratio_file_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax[5].pcolormesh(grid_lon, grid_lat, ratio_rain_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax[6].pcolormesh(grid_lon, grid_lat, grid_sr_wt, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax[7].pcolormesh(grid_lon, grid_lat, grid_agl_wt, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    for axi in ax:
        axi.set_extent(MOSAIC_AREA, crs=ccrs.PlateCarree())
    fig.savefig(f'./dataset/check_files-ACC1H_grid-{radarname}-{timestamp.strftime("%Y%m%d%H%M")}.png', dpi=300, bbox_inches='tight')
    input() """
    return np.stack([acc_grid, ratio_file_grid, ratio_rain_grid, grid_sr_wt, grid_agl_wt], axis=0)  # (5,H,W)

def prepare_inputs(timestamp, df_row, dict_gridinfo, radarnames):
    ls_radar_inputs = []
    for i, radarname in enumerate(radarnames):
        data = np.load(df_row[radarname])
        radar_input = polar_to_cartesian(data, dict_gridinfo[radarname], radarname, timestamp)
        ls_radar_inputs.append(radar_input)
    x_input = np.concatenate(ls_radar_inputs, axis=0)  # (5*num_radars,H,W)
    return x_input

# ========================================
# Module: 测试
# ========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from model import RadarFusionWeightNet, RadarFusionWeightNet2Head
    model = RadarFusionWeightNet2Head(base_ch=32, depth=4, n_res=1, norm="nonorm", act="relu",
                                    c_min=0.5, c_max=2.0, c_head_hidden=16).to(device)
    MODELPATH = 'models/ver2'
    model, best_ep, best_va = load_model_from_ckpt(model, f'{MODELPATH}/best.pt', device)
    import logging
    logging.basicConfig(
        filename=f'{MODELPATH}/test.log',                  # 日志文件名
        level=logging.INFO,                        # 记录 INFO 及以上级别的日志
        format='%(asctime)s---%(message)s',        # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'                # 时间格式
    )
    logging.info(f"Loaded model from {MODELPATH}/best.pt | best epoch={best_ep} | best val_loss={best_va}")

    df = pd.read_csv('dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True).iloc[:, :-1]
    radarnames = df.columns.tolist()
    dict_gridinfo = load_gridinfo(radarnames)
    for timestamp in df.index:
        x_input = prepare_inputs(timestamp, df.loc[timestamp], dict_gridinfo, radarnames)
        r_full, w_full, logits_full, max_id, entropy, c_full = run(model, x_input, device=device)

        date = timestamp.strftime('%Y%m%d')
        out_path = f'/data/zry/BJradar_processed/radarsys-out/{date}/ACC1H-hsr/Merge_DL_ver2/Merge_DL_{timestamp.strftime("%Y%m%d%H%M")}.npz'
        save_full_scene_npz(
            out_path,
            R_k = x_input[[0,5,10,15,20],:,:],   # (5,H,W) numpy
            W_k = w_full,                        # (5,H,W) numpy
            logits_k = logits_full,              # (5,H,W) numpy
            R_hat = r_full,                      # (H,W) numpy
            max_id = max_id,
            entropy = entropy,
            c_full = c_full,                      # (H,W) numpy
        )
        logging.info(f"Saved output to {out_path} for timestamp {timestamp}")


if __name__ == "__main__":
    main()