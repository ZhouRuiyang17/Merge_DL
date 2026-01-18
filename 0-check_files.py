import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
# import radarsys as rds  # 雷达处理
import utils.mytools as mt  # 自定义工具
import utils.eval_tools as et  # 评估工具
from utils.information import *
import cartopy.crs as ccrs
from parameters import *

fontsize = 12
plt.rcParams['font.size'] = 10        # 默认字体大小
plt.rcParams['xtick.labelsize'] = 10  # X 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # Y 轴刻度字体大小
plt.rcParams['axes.labelsize'] = 12   # 轴标签大小
plt.rcParams['axes.titlesize'] = 12   # 标题大小
plt.rcParams['legend.fontsize'] = 10  # 图例大小

def check_QC():
    oldfile = '/data/zry/BJradar_processed/radarsys-out/20180716/BJXFS/BJXFS.20180716.003600_HSR.npz'
    newfile = '/data/zry/BJradar_processed/radarsys-out/20180716/QPE-hsr/BJXFS/BJXFS_20180716003600.npz'

    olddata = np.load(oldfile)
    newdata = np.load(newfile)

    oldref = olddata['data'][-1].squeeze()
    newref = newdata['rr_ra_rz'].squeeze()

    print('Old data shape:', oldref.shape)
    print('New data shape:', newref.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    et.RADAR(oldref, 'rr', *BJXFS, eles=[0.5]).ppi(0, ax=ax[0])
    et.RADAR(newref, 'rr', *BJXFS, eles=[0.5]).ppi(0, ax=ax[1])
    fig.savefig('./dataset/check_files-rr-20180716003600-BJXFS.png', dpi=300)

def check_ACC(ls_fp, timestamp):

    gauge = pd.read_csv('/data/zry/beijing/gauge/gauge_all.csv', index_col=0, parse_dates=True).loc[timestamp]
    gaugeinfo = pd.read_csv('/data/zry/siteinfo/gauge_info.csv', index_col=0)
    lons = gaugeinfo.loc[gauge.index, 'lon'].values
    lats = gaugeinfo.loc[gauge.index, 'lat'].values
    lonll, latll = 116.5, 40.0
    AREA = [lonll, lonll+WINDOW_SIZE*RESOLUTION, latll, latll+WINDOW_SIZE*RESOLUTION]

    grid_gauge = np.load(ls_fp[-1])['grid_data'].squeeze()
    loc = grid_gauge > 0

    fig, ax = plt.subplots(2,5, figsize=(40, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    ax = ax.flatten()
    for i, fp in enumerate(ls_fp[:-1]):
        data = np.load(fp)
        acc = data['acc'].squeeze()
        radarname = os.path.basename(fp).split('_')[0]
        et.RADAR(acc, 'acc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, area=MOSAIC_AREA, ax=ax[i])#, scatters=[lons, lats, gauge.values])
        et.RADAR(acc, 'acc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, area=AREA, ax=ax[i+5])#, scatters=[lons, lats, gauge.values])
        # et.load_shapefile(ax=ax[i])
        ax[i].set_xticks(np.arange(115.5, 117.6, 0.5))
        ax[i].set_yticks(np.arange(39.5, 41.2, 0.5))
        rec = plt.Rectangle((lonll, latll), WINDOW_SIZE*RESOLUTION, WINDOW_SIZE*RESOLUTION,
                        linewidth=1, edgecolor='r', facecolor='none')        
        ax[i].add_patch(rec)
        rec = plt.Rectangle((LONLL, LATLL), 1024*RESOLUTION, 1024*RESOLUTION,
                        linewidth=1, edgecolor='r', facecolor='none')        
        ax[i].add_patch(rec)
        cmap, norm, _, _ = et.colorbar('acc')
        ax[i].scatter(GRID_LON[loc], GRID_LAT[loc], s=20, c=grid_gauge[loc], marker='o', cmap=cmap, norm=norm, edgecolors='k', linewidths=0.2)
        ax[i+5].scatter(GRID_LON[loc], GRID_LAT[loc], s=100, c=grid_gauge[loc], marker='o', cmap=cmap, norm=norm, edgecolors='k', linewidths=0.2)

    fig.savefig(f'./dataset/check_files-ACC1H-{timestamp.strftime("%Y%m%d%H%M")}-{WINDOW_SIZE}.png', dpi=300, bbox_inches='tight')

def collect_files():
    rootdir = '/data/zry/BJradar_processed/'
    columns = ['BJXFS', 'BJXCP', 'BJXSY', 'BJXTZ', 'Z9010']
    # df = pd.DataFrame(columns=columns)
    # for root, dirs, files in os.walk(rootdir):
    #     if not ('2019' in root and 'QPE-hsr' in root):
    #         continue
    #     for file in files:
    #         if file.endswith('.npz'):
    #             filepath = os.path.join(root, file)
    #             vec = file.split('_')
    #             radarname = vec[0]
    #             timestamp_str = vec[1][:12]
    #             timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M')
    #             if radarname in columns:
    #                 df.loc[timestamp, radarname] = filepath
    # df = df.sort_index()
    # df = df.fillna('nodata')
    # df.to_csv('./dataset/filelist-QPE-hsr-2019.csv')
    df = pd.read_csv('./dataset/filelist-QPE-hsr-2019.csv', index_col=0, parse_dates=True)
    df = df.fillna('nodata')

    df_acc = pd.DataFrame(columns=columns)
    time_index = pd.date_range(
                    start="2019-07-01 00:00",
                    end="2019-09-30 23:00",
                    freq="H"
                )
    for timestamp in time_index:
        for radarname in columns:
            if 'BJX' in radarname:
                n_gt = 1000
            else:
                n_gt = 460

            ### prepare filepath
            date_str = timestamp.strftime('%Y%m%d')
            dirname = f'/data/zry/BJradar_processed/{date_str}/ACC1H-hsr/{radarname}/'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            timestamp_end = timestamp + pd.Timedelta(hours=1)
            filename = f'{radarname}_{timestamp_end.strftime("%Y%m%d%H%M")}.npz'
            fp_new = os.path.join(dirname, filename)
            info = '过去1h的累计降水量(mm), 数据为hsr数据, QPE方法为rarz\n'
            info += 'acc: 累计降水量\n'
            info += 'num_file: 有效文件个数, 因为有时候缺测, 不是每个小时都有10个观测(6min间隔)\n'
            info += 'num_rain: 每个格点降水>0的次数\n'
            # print(info)
            if os.path.exists(fp_new):
                print(f'File exists: {fp_new}, skip.')
                df_acc.loc[timestamp_end, radarname] = fp_new
                continue

            ### get filepaths of past 1 hour
            ls_fp = df.loc[timestamp:timestamp+pd.Timedelta(minutes=59), radarname].to_list()
            ### accumulate
            init = True
            num_file = 0
            for fp in ls_fp:
                ### 如果文件不存在, 则视为无降水
                if fp != 'nodata':
                    num_file += 1
                    data = np.load(fp)
                    if 'rr_ra_rz' in data:
                        rr = data['rr_ra_rz'].squeeze()
                    elif 'rr_ref' in data:
                        rr = data['rr_ref'].squeeze()
                else:
                    rr = np.zeros((360,n_gt))*1.0  # nodata as zero rainfall
                
                if init:
                    acc = rr * 6/60  # mm/h to mm/6min
                    num_rain = (rr > 0).astype(np.int32) # 统计有效降水格点数
                    init = False
                else:
                    acc += rr * 6/60  # mm/h to mm/6min
                    num_rain += (rr > 0).astype(np.int32)
            
            num_file = np.full_like(num_rain, num_file)
            np.savez_compressed(fp_new, info=info, acc=acc, num_rain=num_rain, num_file=num_file)
            df_acc.loc[timestamp_end, radarname] = fp_new
    df_acc = df_acc.sort_index()

    for timestamp in df_acc.index:
        timestamp_str = timestamp.strftime('%Y%m%d%H%M')
        date = timestamp.strftime('%Y%m%d')
        fp = f'/data/zry/BJradar_processed/{date}/ACC1H-hsr/Merge_DL-0.001deg/gauge_{timestamp_str}.npz'
        df_acc.loc[timestamp, 'gauge'] = fp
    df_acc.to_csv('./dataset/filelist-ACC1H-hsr-2019.csv')

if __name__ == '__main__':
    check_QC()
    # collect_files()
    # df = pd.read_csv('./dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True)
    # for hour in range(15, 22):
    #     timestamp = pd.to_datetime(f'2019-07-22 {hour}:00')
    #     ls_fp = df.loc[timestamp].to_list()
    #     check_ACC(ls_fp, timestamp)