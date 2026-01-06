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
    oldfile = '/data/zry/BJradar_processed/20190728/BJXSY/BJXSY.20190728.190000_HSR.npz'
    newfile = '/data/zry/BJradar_processed/20190728/QC/BJXSY/BJXSY_20190728190000.npz'

    olddata = np.load(oldfile)
    newdata = np.load(newfile)

    oldref = olddata['data'][0].squeeze()
    newref = newdata['ref'].squeeze()

    print('Old data shape:', oldref.shape)
    print('New data shape:', newref.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    et.RADAR(oldref, 'ref', *BJXSY, eles=[0.5]).ppi(0, ax=ax[0])
    et.RADAR(newref, 'ref', *BJXSY).ppi(1, ax=ax[1])
    fig.savefig('./dataset/check_files-ref-201907281900.png', dpi=300)

def check_ACC():
    df = pd.read_csv('./dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True)
    timestamp = pd.to_datetime('2019-07-22 16:00')
    ls_fp = df.loc[timestamp].to_list()

    gauge = pd.read_csv('/data/zry/beijing/gauge/gauge_all.csv', index_col=0, parse_dates=True).loc[timestamp]
    gaugeinfo = pd.read_csv('/data/zry/siteinfo/gauge_info.csv', index_col=0)
    lons = gaugeinfo.loc[gauge.index, 'lon'].values
    lats = gaugeinfo.loc[gauge.index, 'lat'].values
    lonll, latll = 116.5, 40.0
    AREA = [lonll, lonll+WINDOW_SIZE*RESOLUTION, latll, latll+WINDOW_SIZE*RESOLUTION]


    fig, ax = plt.subplots(2,4, figsize=(32, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    ax = ax.flatten()
    for i, fp in enumerate(ls_fp):
        data = np.load(fp)
        acc = data['count_vali'].squeeze()
        radarname = os.path.basename(fp).split('_')[0]
        et.RADAR(acc/10, 'cc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, area=MOSAIC_AREA, ax=ax[i])#, scatters=[lons, lats, gauge.values])
        et.RADAR(acc/10, 'cc', *BJ_RADAR_DICT[radarname], eles=[0.5]).ppi_wgs(0, area=AREA, ax=ax[i+4])#, scatters=[lons, lats, gauge.values])
        # et.load_shapefile(ax=ax[i])
        ax[i].set_xticks(np.arange(115.5, 117.6, 0.5))
        ax[i].set_yticks(np.arange(39.5, 41.2, 0.5))
        rec = plt.Rectangle((lonll, latll), WINDOW_SIZE*RESOLUTION, WINDOW_SIZE*RESOLUTION,
                        linewidth=1, edgecolor='r', facecolor='none')        
        ax[i].add_patch(rec)
        rec = plt.Rectangle((LONLL, LATLL), 1024*RESOLUTION, 1024*RESOLUTION,
                        linewidth=1, edgecolor='r', facecolor='none')        
        ax[i].add_patch(rec)

    fig.savefig(f'./dataset/check_files-count-201907221600-{WINDOW_SIZE}.png', dpi=300, bbox_inches='tight')

def collect_files():
    rootdir = '/data/zry/BJradar_processed/'
    columns = ['BJXFS', 'BJXCP', 'BJXSY', 'BJXTZ']
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
            ### prepare filepath
            date_str = timestamp.strftime('%Y%m%d')
            dirname = f'/data/zry/BJradar_processed/{date_str}/ACC1H-hsr/{radarname}/'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            timestamp_end = timestamp + pd.Timedelta(hours=1)
            filename = f'{radarname}_{timestamp_end.strftime("%Y%m%d%H%M")}.npz'
            fp_new = os.path.join(dirname, filename)
            info = '过去1h的累计降水量(mm), 数据为hsr数据, QPE方法为rarz\nacc: 累计降水量\ncount_vali: 有效文件个数, 因为有时候缺测, 不是每个小时都有10个观测(6min间隔)'
            # print(info)
            # if os.path.exists(fp_new):
            #     print(f'File exists: {fp_new}, skip.')
            #     continue

            ### get filepaths of past 1 hour
            ls_fp = df.loc[timestamp:timestamp+pd.Timedelta(minutes=59), radarname].to_list()
            if len(ls_fp) < 5:
                continue

            ### accumulate
            count = 0
            for fp in ls_fp:
                ### 如果文件不存在, 则视为无降水
                if fp != 'nodata':
                    data = np.load(fp)
                    rr = data['rr_ra_rz'].squeeze()
                else:
                    rr = np.zeros((360,1000))*1.0  # nodata as zero rainfall
                
                if count == 0:
                    acc = rr * 6/60  # mm/h to mm/6min
                    count_vali = (rr > 0).astype(np.int32) # 统计有效降水格点数
                else:
                    acc += rr * 6/60  # mm/h to mm/6min
                    count_vali += (rr > 0).astype(np.int32)
                count += 1
            
            np.savez_compressed(fp_new, info=info, acc=acc, count_vali=count_vali)
            df_acc.loc[timestamp_end, radarname] = fp_new
    df_acc = df_acc.sort_index()
    df_acc.to_csv('./dataset/filelist-ACC1H-hsr-2019.csv')

if __name__ == '__main__':
    collect_files()
    check_ACC()