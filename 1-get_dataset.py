import numpy as np
import pandas as pd
import os, gc, csv
import datetime
import matplotlib.pyplot as plt
import utils.mytools as mt  # 自定义工具
import utils.eval_tools as et  # 评估工具
from utils.information import *
import cartopy.crs as ccrs
import parameters as params
WINDOW_SIZE = params.WINDOW_SIZE

fontsize = 12
plt.rcParams['font.size'] = 10        # 默认字体大小
plt.rcParams['xtick.labelsize'] = 10  # X 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # Y 轴刻度字体大小
plt.rcParams['axes.labelsize'] = 12   # 轴标签大小
plt.rcParams['axes.titlesize'] = 12   # 标题大小
plt.rcParams['legend.fontsize'] = 10  # 图例大小

def generate_dataset_and_index():
    """
    对于某一时间timestamp, 某一地点idlon和idlat, 取出WINDOW_SIZE的区域, 统计该区域内有多少个雨量站有数据num_gauge
    数据集这么组织: radar1, radar2, ..., radarN, gauge
    最后一层是label, 其余全为inputs
    每个radar有3个动态信息(acc, 文件覆盖率和降雨覆盖率)和2个静态信息(高度和距离)
    """

    ### 读取数据目录
    df = pd.read_csv('dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True)
    # df = df.loc['2019-7-22 15:00':'2019-7-22 22:00']

    ### 加载静态信息
    dict_gridinfo = {radarname: np.load(f'gridinfo/Merge_DL-0.001deg/gridinfo_{radarname}.npz') for radarname in df.columns[:-1]}

    rows = []
    dataset = []
    for timestamp in df.index:
        all_griddata = []
        ### 处理每个雷达和雨量站
        print('Processing timestamp:', timestamp)
        DATE = timestamp.strftime('%Y%m%d')
        for col in df.columns:
            if 'gauge' not in col:
                radarname = col
                data = np.load(df.loc[timestamp, radarname])
                acc = data['acc']
                num_file = data['num_file']
                num_rain = data['num_rain']
                ratio_file = num_file / 10 # 按照6min扫描间隔, 每小时应该有10个文件
                ratio_rain = num_rain / num_file # 在有文件的地方, 多少比例有雨

                gridinfo = dict_gridinfo[radarname]
                grid_agl = gridinfo['grid_AGL']/1000  # m to km
                grid_sr = gridinfo['grid_sr']/1000  # m to km
                L = 100  # 100km
                H = 2  # 2km
                grid_sr_wt = np.exp(-grid_sr**2/L**2) 
                grid_agl_wt = np.exp(-grid_agl**2/H**2)
                loc = np.isnan(grid_sr_wt) | np.isinf(grid_sr_wt) | np.isnan(grid_agl_wt) | np.isinf(grid_agl_wt) # 无效点: 任意点为NaN或Inf
                grid_sr_wt[loc] = 0.0
                grid_agl_wt[loc] = 0.0

                grid_az = gridinfo['grid_az']
                grid_gt = gridinfo['grid_gt']
                loc = np.isnan(grid_az) | np.isinf(grid_az) | np.isnan(grid_gt) | np.isinf(grid_gt) # 无效点: 任意点为NaN或Inf
                acc_grid = np.zeros_like(grid_az)*1.0
                acc_grid[~loc] = acc[grid_az[~loc].astype(int),grid_gt[~loc].astype(int)]
                ratio_file_grid = np.zeros_like(grid_az)*1.0
                ratio_file_grid[~loc] = ratio_file[grid_az[~loc].astype(int), grid_gt[~loc].astype(int)]
                ratio_rain_grid = np.zeros_like(grid_az)*1.0
                ratio_rain_grid[~loc] = ratio_rain[grid_az[~loc].astype(int), grid_gt[~loc].astype(int)]

                """ fig, ax = plt.subplots(3, 3, figsize=(15,15), subplot_kw={'projection': ccrs.PlateCarree()})
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
                fig.savefig(f'./dataset/check_files-ACC1H_grid-{radarname}-{timestamp.strftime("%Y%m%d%H%M")}.png', dpi=300, bbox_inches='tight') """
                a_radar = [acc_grid, ratio_file_grid, ratio_rain_grid, grid_sr_wt, grid_agl_wt]
                a_radar = np.stack(a_radar, axis=0)  # (5, H, W)
                all_griddata.append(a_radar)
            elif 'gauge' in col:
                gauge = np.load(df.loc[timestamp, col])['grid_data'].squeeze()  # (H, W)
                all_griddata.append(gauge[np.newaxis, ...])  # (1, H, W)
        all_griddata = np.concatenate(all_griddata, axis=0)  # (N_radars*5+1, H, W)
        ### 对该时间的所有雷达和雨量站数据, 进行切片
        for idlon in range(0, 1024, int(WINDOW_SIZE/2)):
            for idlat in range(0, 1024, int(WINDOW_SIZE/2)):
                subdata = all_griddata[:, idlat:idlat+WINDOW_SIZE, idlon:idlon+WINDOW_SIZE]  # (N_radars*5+1, WINDOW_SIZE, WINDOW_SIZE)
                if subdata.shape[1] < WINDOW_SIZE or subdata.shape[2] < WINDOW_SIZE:
                    continue
                gauge_sub = subdata[-1, ...]
                loc = gauge_sub > 0
                num_gauge = np.sum(loc)
                rows.append({
                    'timestamp': timestamp,
                    'idlon': idlon,
                    'idlat': idlat,
                    'num_gauge': num_gauge
                })
                dataset.append(subdata)
    index_dataset = pd.DataFrame(rows)
    index_dataset.to_csv(f'./dataset/index_dataset-W{WINDOW_SIZE}-hsr-2019.csv', index=False)
    dataset = np.stack(dataset, axis=0)  # (N_samples, N_radars*5+1, WINDOW_SIZE, WINDOW_SIZE)
    np.savez_compressed(f'./dataset/dataset-W{WINDOW_SIZE}-hsr-2019.npz', dataset=dataset)


def generate_dataset_and_index_memmap(df, dict_gridinfo, WINDOW_SIZE, out_dir, tag="2019"):
    os.makedirs(out_dir, exist_ok=True)

    stride = WINDOW_SIZE // 2
    # 只取完整窗口：id 从 0 到 1024-WINDOW_SIZE
    n_lon = ((1024 - WINDOW_SIZE) // stride) + 1
    n_lat = ((1024 - WINDOW_SIZE) // stride) + 1
    n_patch_per_time = n_lon * n_lat
    n_time = len(df.index)

    # 通道数：每个雷达 5 个 + 1 个 gauge；雷达数= df.columns 中非 gauge 的列数
    radar_cols = [c for c in df.columns if 'gauge' not in c]
    gauge_cols = [c for c in df.columns if 'gauge' in c]
    if len(gauge_cols) != 1:
        raise ValueError(f"期望只有1个gauge列，但得到 {len(gauge_cols)} 个：{gauge_cols}")
    C = len(radar_cols) * 5 + 1

    Ns = n_time * n_patch_per_time

    # 1) dataset 直接写成 .npy memmap：后续 np.load(..., mmap_mode='r') 就能按需读
    x_path = os.path.join(out_dir, f"dataset-W{WINDOW_SIZE}-hsr-{tag}.npy")
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(Ns, C, WINDOW_SIZE, WINDOW_SIZE)
    )

    # 2) index 边写边落盘
    idx_path = os.path.join(out_dir, f"index_dataset-W{WINDOW_SIZE}-hsr-{tag}.csv")
    with open(idx_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["timestamp", "idlon", "idlat", "num_gauge"])
        writer.writeheader()

        k = 0
        for timestamp in df.index:
            print("Processing timestamp:", timestamp)

            # 先把这一时刻所有雷达/gauge拼成 (C,1024,1024)，
            all_griddata = []

            # ---- 雷达部分 ----
            for radarname in radar_cols:
                data = np.load(df.loc[timestamp, radarname])  # 这里是 npz
                acc = data["acc"]
                num_file = data["num_file"]
                num_rain = data["num_rain"]

                ratio_file = num_file / 10.0
                ratio_rain = num_rain / np.maximum(num_file, 1)  # 防止除0

                gridinfo = dict_gridinfo[radarname]
                grid_agl = (gridinfo["grid_AGL"] / 1000.0).astype(np.float32)  # km
                grid_sr  = (gridinfo["grid_sr"]  / 1000.0).astype(np.float32)  # km
                L, H = 100.0, 2.0
                grid_sr_wt  = np.exp(-(grid_sr ** 2)  / (L ** 2)).astype(np.float32)
                grid_agl_wt = np.exp(-(grid_agl ** 2) / (H ** 2)).astype(np.float32)

                # 无效点置0
                loc_w = ~np.isfinite(grid_sr_wt) | ~np.isfinite(grid_agl_wt)
                grid_sr_wt[loc_w] = 0.0
                grid_agl_wt[loc_w] = 0.0

                grid_az = gridinfo["grid_az"]
                grid_gt = gridinfo["grid_gt"]
                loc = ~np.isfinite(grid_az) | ~np.isfinite(grid_gt)

                # 预分配输出格点（float32），避免 zeros_like * 1.0 这种隐式 dtype 问题
                acc_grid = np.zeros(grid_az.shape, dtype=np.float32)
                ratio_file_grid = np.zeros(grid_az.shape, dtype=np.float32)
                ratio_rain_grid = np.zeros(grid_az.shape, dtype=np.float32)

                iaz = grid_az[~loc].astype(np.int32)
                igt = grid_gt[~loc].astype(np.int32)

                acc_grid[~loc] = acc[iaz, igt]
                ratio_file_grid[~loc] = ratio_file[iaz, igt]
                ratio_rain_grid[~loc] = ratio_rain[iaz, igt]

                a_radar = np.stack(
                    [acc_grid, ratio_file_grid, ratio_rain_grid, grid_sr_wt, grid_agl_wt],
                    axis=0
                )  # (5,1024,1024) float32
                all_griddata.append(a_radar)

                # 释放本雷达临时变量引用（保险）
                del data, acc, num_file, num_rain, ratio_file, ratio_rain
                del acc_grid, ratio_file_grid, ratio_rain_grid, a_radar

            # ---- gauge 部分 ----
            gauge_name = gauge_cols[0]
            gauge = np.load(df.loc[timestamp, gauge_name])["grid_data"].squeeze().astype(np.float32)  # (1024,1024)
            all_griddata.append(gauge[np.newaxis, ...])  # (1,1024,1024)

            # 拼接成 (C,1024,1024)；这块是你循环内最大的内存块，但它不随 Ns 累积
            all_griddata = np.concatenate(all_griddata, axis=0).astype(np.float32, copy=False)

            # ---- 切片：边写 index，边写 X[k] ----
            for idlon in range(0, 1024 - WINDOW_SIZE + 1, stride):
                for idlat in range(0, 1024 - WINDOW_SIZE + 1, stride):
                    subdata = all_griddata[:, idlat:idlat+WINDOW_SIZE, idlon:idlon+WINDOW_SIZE]  # view

                    gauge_sub = subdata[-1]
                    num_gauge = int(np.sum(gauge_sub > 0))

                    writer.writerow({
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "idlon": idlon,
                        "idlat": idlat,
                        "num_gauge": num_gauge
                    })

                    # 写入 memmap（落盘），不进内存队列；
                    X[k] = subdata.astype(np.float32, copy=False)
                    k += 1

            # 清理这一时刻的大块数组
            del all_griddata, gauge
            gc.collect()

    # flush 到磁盘
    X.flush()
    print(f"Done. Saved X to: {x_path}")
    print(f"Saved index to: {idx_path}")
    return x_path, idx_path




# generate_dataset_and_index()
# 用法：
df = pd.read_csv('dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True)
dict_gridinfo = {radarname: np.load(f'gridinfo/Merge_DL-0.001deg/gridinfo_{radarname}.npz') for radarname in df.columns[:-1]}
x_path, idx_path = generate_dataset_and_index_memmap(df, dict_gridinfo, WINDOW_SIZE, out_dir="./dataset", tag="2019")
