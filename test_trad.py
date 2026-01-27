import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import radarsys as rds  # 雷达处理
import utils.mytools as mt  # 自定义工具
import utils.eval_tools as et  # 评估工具
from utils.information import *
import radarsys as rds


def load_gridinfo(radarnames):
    dict_gridinfo = {radarname: np.load(f'gridinfo/Merge_DL-0.001deg/gridinfo_{radarname}.npz') for radarname in radarnames}
    
    grid_azs = np.stack([dict_gridinfo[radarname]['grid_az'] for radarname in radarnames], axis=0)
    grid_gts = np.stack([dict_gridinfo[radarname]['grid_gt'] for radarname in radarnames], axis=0)
    grid_srs = np.stack([dict_gridinfo[radarname]['grid_sr'] for radarname in radarnames], axis=0)
    grid_AGLs = np.stack([dict_gridinfo[radarname]['grid_AGL'] for radarname in radarnames], axis=0)

    return grid_azs, grid_gts, grid_srs, grid_AGLs

def main():
    df = pd.read_csv('dataset/filelist-ACC1H-hsr-2019.csv', index_col=0, parse_dates=True).iloc[:, :-1]
    radarnames = df.columns.tolist()
    grid_azs, grid_gts, grid_srs, grid_AGLs = load_gridinfo(radarnames)
    for timestamp in df.index:
        date_str = timestamp.strftime("%Y%m%d")
        SAVEPATH = f'/data/zry/BJradar_processed/radarsys-out/{date_str}/ACC1H-hsr/Merge_DL_trad'
        os.makedirs(SAVEPATH, exist_ok=True)

        ls_fp = df.loc[timestamp].tolist()
        data2d_list = []
        for fp in ls_fp:
            if os.path.exists(fp):
                files = np.load(fp)
                data2d = files['acc']
            else:
                data2d = np.full((360, 230), 0.0) if 'Z9010' in fp else np.full((360, 1000), 0.0)
            data2d_list.append(data2d)

        data_mosaic_max = rds.mosaic.mosaic_by_max_value(data2d_list, grid_azs, grid_gts)[0]
        data_mosaic_avg = rds.mosaic.mosaic_by_weight(data2d_list, grid_azs, grid_gts,)[0]
        data_mosaic_wt = rds.mosaic.mosaic_by_weight(data2d_list, grid_azs, grid_gts,
                                                        grid_srs, grid_AGLs)[0]
        np.savez_compressed(f'{SAVEPATH}/Merge_trad_{timestamp.strftime("%Y%m%d%H%M")}.npz',
                            R_max=data_mosaic_max,
                            R_avg=data_mosaic_avg,
                            R_wt=data_mosaic_wt)
        print(f'Saved: {SAVEPATH}/Merge_trad_{timestamp.strftime("%Y%m%d%H%M")}.npz')

        
main()