import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import utils.mytools as mt  # 自定义工具
import utils.eval_tools as et  # 评估工具
from utils.information import BJ_AREA_SMALL, area_Merge_DL
import cartopy.crs as ccrs
from typing import List

MOSAIC_AREA, grid_lon, grid_lat = area_Merge_DL(True)
fontsize = 12
plt.rcParams['font.size'] = 10        # 默认字体大小
plt.rcParams['xtick.labelsize'] = 10  # X 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # Y 轴刻度字体大小
plt.rcParams['axes.labelsize'] = 12   # 轴标签大小
plt.rcParams['axes.titlesize'] = 12   # 标题大小
plt.rcParams['legend.fontsize'] = 10  # 图例大小

def plot_distribution(ls_griddata, ls_titles, ax):

    cmap,norm, _,_ = et.colorbar('acc')
    for i, axi in enumerate(ax):
        axi.set_extent(MOSAIC_AREA, crs=ccrs.PlateCarree())
        pm = axi.pcolormesh(grid_lon, grid_lat, ls_griddata[i], cmap=cmap, norm=norm)
        axi.set_title(ls_titles[i])
        # et.load_shapefile(ax=axi)




def plot_scatter(gauge: pd.DataFrame, ls_radar: List[pd.DataFrame], axs: List[plt.Axes], ls_titles: List[str]):
    ### df to arr
    gauge_arr = gauge.values.flatten()
    ls_radar_arr = [radar.values.flatten() for radar in ls_radar]

    ### arr to hit: only consider gauge>=threshold and radar>=threshold
    mask = gauge_arr >= 0.1
    for i in range(len(ls_radar_arr)):
        mask = mask & (ls_radar_arr[i] >= 0.1)
    gauge_hit = gauge_arr[mask]
    ls_radar_hit = [ls_radar_arr[i][mask] for i in range(len(ls_radar_arr))]

    ### plot
    for i in range(len(ls_radar)):
        et.plot_hist2d(gauge_hit, ls_radar_hit[i], bins=[np.linspace(0, 100, 101), np.linspace(0, 100, 101)],ax=axs[i],
                       showmet=1, drawline=1)
        axs[i].set_title(ls_titles[i])
        
    ### calculate csi, pod, far
    df = pd.DataFrame(columns=['RMB','RMSE','CORR','CSI-1mm', 'POD-1mm', 'FAR-1mm', 'CSI-10mm', 'POD-10mm', 'FAR-10mm'])
    for i in range(len(ls_radar)):
        metircs = et.get_metrics(gauge_arr, ls_radar_arr[i])
        metrics_1mm = et.get_metrics_hit(gauge_arr, ls_radar_arr[i], threshold=1)
        metrics_10mm = et.get_metrics_hit(gauge_arr, ls_radar_arr[i], threshold=10)
        df.loc[ls_titles[i]] = [metircs['RMB'], metircs['RMSE'], metircs['CORR'],
                                    metrics_1mm['CSI'], metrics_1mm['POD'], metrics_1mm['FAR'],
                                    metrics_10mm['CSI'], metrics_10mm['POD'], metrics_10mm['FAR']]
    return df


if __name__ == "__main__":
    gauge = pd.read_csv('/data/zry/beijing/gauge/gauge_all.csv', index_col=0, parse_dates=True).loc['2019']
    gauge[gauge>1000] = np.nan
    gauge = gauge.fillna(0)

    dict_radar = {'model': 'ver1-1718_100',
                  'max mosaic': 'max',
                   'BJXFS': 'BJXFS',
                   'BJXCP': 'BJXCP',
                   'BJXSY': 'BJXSY',
                   'BJXTZ': 'BJXTZ',
                   'Z9010': 'Z9010',
                   }
    ls_radar = [pd.read_csv(f'./results/acc-{filename}.csv', index_col=0, parse_dates=True).loc['2019'] for filename in dict_radar.values()]

    gauge = gauge.reindex(index=ls_radar[0].index, columns=ls_radar[0].columns)

    fig, ax = plt.subplots(1, len(ls_radar), figsize=(5*len(ls_radar), 5))
    df = plot_scatter(gauge, ls_radar, ax, list(dict_radar.keys()))
    fig.savefig(f'./results/scatter.png', dpi=300, bbox_inches='tight')
    print(df)
    df.to_csv(f'./results/metrics.csv')

    filelist = pd.read_csv('./results/filelist-ver1-1718_100.csv', index_col=0, parse_dates=True)
    timestamp = pd.to_datetime('201907221400', format='%Y%m%d%H%M')
    fp = filelist.loc[timestamp, 'filepath']
    files = np.load(fp)
    griddata_DL = files.get('R_hat').squeeze()
    griddata_k = files.get('R_k').squeeze()
    griddata_max = np.nanmax(griddata_k, axis=0)
    ls_griddata = [griddata_DL] + [griddata_max] + [griddata_k[i] for i in range(griddata_k.shape[0])]
    fig, ax = plt.subplots(1, len(ls_griddata), figsize=(5*len(ls_griddata), 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax = ax.flatten()
    plot_distribution(ls_griddata, list(dict_radar.keys()), ax)
    fig.savefig(f'./results/distribution-{timestamp.strftime("%Y%m%d%H%M")}.png', dpi=300, bbox_inches='tight')
