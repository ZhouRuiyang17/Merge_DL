import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import utils.mytools as mt  # 自定义工具
import utils.eval_tools as et  # 评估工具
from utils.information import *
import utils.point2radar as p2r
import radarsys as rds

MOSAIC_AREA, grid_lon, grid_lat = area_Merge_DL(True)
siteinfo_path = '/data/zry/siteinfo/gauge_info.csv'

""" df_fp = pd.DataFrame(columns=['filepath'])
PATH = '/data/zry/BJradar_processed/radarsys-out/'
for root, dirs, files in os.walk(PATH):
    for file in files:
        if file.endswith('.npz') and 'Merge_trad' in file and '2019' in root and 'Merge_DL_ver2' not in root:
            timestamp = file.split('_')[-1].split('.')[0]
            timestamp = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M')
            df_fp.loc[timestamp] = [os.path.join(root, file)]
df_fp = df_fp.sort_index()
df_fp.to_csv('./results/filelist-trad.csv') """
df_fp = pd.read_csv('./results/filelist-trad.csv', index_col=0, parse_dates=True)
ls_fp = df_fp['filepath'].tolist()


def reader(fp, mode=12):
    filename = os.path.basename(fp)
    timestamp_str = filename.split('_')[-1].split('.')[0]
    timestamp = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M')
    files = np.load(fp)
    if mode == -1:
        griddata = files['R_hat'].squeeze()
    elif 0 <= mode <= 4:
        griddata = files['R_k'][mode].squeeze()
    elif mode == 5: # 最大值融合
        griddata_k = files['R_k'].squeeze()
        griddata = np.nanmax(griddata_k, axis=0)
    elif mode == -2:
        correct_coeff = files['correct_coeff'].squeeze()
        griddata = correct_coeff
    elif mode == 11:
        griddata = files['R_max'].squeeze()
    elif mode == 12:
        griddata = files['R_avg'].squeeze()
    elif mode == 13:
        griddata = files['R_wt'].squeeze()
    return timestamp, griddata

out = p2r.lookup(ls_fp, grid_lon, grid_lat, siteinfo_path, reader)
out = out.sort_index()
out.to_csv('./results/acc-R_avg.csv')