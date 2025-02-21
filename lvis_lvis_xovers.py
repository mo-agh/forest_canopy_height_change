DESCRIPTION='''
Return LVIS_LVIS crossovers
University of Maryland
Mohammad Aghdami Nia
'''

import os
import glob
import sys
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
import h5py
import pyproj
import pandas as pd
import multiprocessing
from contextlib import contextmanager
import geopandas as gpd
from shapely.geometry import Polygon
import traceback


def get_tile_id(longitude, latitude, tilesize=6):
    ease2_origin = -17367530.445161499083042, 7314540.830638599582016
    ease2_nbins = int(34704 / tilesize), int(14616 / tilesize)
    ease2_binsize = 1000.895023349556141*tilesize, 1000.895023349562052*tilesize
    
    transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:6933', always_xy=True)
    x,y = transformer.transform(longitude, latitude)

    xidx = int( (x - ease2_origin[0]) / ease2_binsize[0]) + 1
    yidx = int( (ease2_origin[1] - y) / ease2_binsize[1]) + 1
    return xidx, yidx
    

def read_LVIS_tile(lvis_tile_x, lvis_tile_y):
    lvis_tiles_path = '/gpfs/data1/vclgp/data/iss_gedi/umd/lvis'
    tile_path = f'{lvis_tiles_path}/X{lvis_tile_x:05d}/Y{lvis_tile_y:05d}/*.h5'
    LVIS_paths = []
    for path in glob.glob(tile_path):
        if 'LVISF' not in path:
            LVIS_paths.append(path)
    return LVIS_paths


def query_LVIS_shots(fname):
    fields = []
    def print_h5_structure(name, obj):
        fields.append(name)
    with h5py.File(fname, 'r') as h5_file:
        h5_file.visititems(print_h5_structure)
    
    # fields = ['GLAT', 'GLON', 'LFID', 'RH98', 'RXWAVE', 'SHOTNUMBER', 'Z0', 
    #                'Z1023', 'ZG', 'ZT', 'INCIDENTANGLE', 'CHANNEL_ZG', 'SENSITIVITY', 'CHANNEL_ZT'] 

    h5 = h5py.File(fname,'r')
    LVIS_df = {}
    for field in fields:
        data = h5[field][()]
        if data.ndim>1:
            LVIS_df[field] = [item for item in data.astype('float')]
        else:
            LVIS_df[field] = data
            
    LVIS_df = pd.DataFrame(LVIS_df)   
    h5.close()
    return LVIS_df


def filter_LVIS_shots(LVIS_df):
    LVIS_df = LVIS_df[LVIS_df['INCIDENTANGLE']<6]
    LVIS_df = LVIS_df[LVIS_df['CHANNEL_ZG']==1]
    if 'SENSITIVITY' in LVIS_df.columns:
        LVIS_df = LVIS_df[LVIS_df['SENSITIVITY']>0.95]
    if 'CHANNEL_ZT' in LVIS_df.columns:
        LVIS_df = LVIS_df[LVIS_df['CHANNEL_ZT']==1]
    return LVIS_df


def create_tree(df1, df2, buffer_size=2):
    lons1 = df1['GLON_1']
    lats1 = df1['GLAT_1']
    lons2 = df2['GLON_2']
    lats2 = df2['GLAT_2']
    transformer = pyproj.Transformer.from_crs(pyproj.CRS("epsg:4326"), pyproj.CRS('epsg:6933'), always_xy=True)
    x1, y1 = transformer.transform(lons1, lats1)
    x2, y2 = transformer.transform(lons2, lats2)
    tree = KDTree(np.vstack((x1, y1)).T, leaf_size=2)
    dist, idx_1 = tree.query(np.vstack((x2, y2)).T, k=1)
    idx = list(idx_1)
    for i in range(len(idx)):
        if dist[i]>buffer_size:
            idx[i] = []         
    return idx


def find_crossovers(idx, df1, df2):
    df1_inds = []
    df2_inds = []
    for ind_2, ind_1 in enumerate(idx):
        if ind_1:
            df1_inds.append(ind_1)
            df2_inds.append(ind_2)    
    if len(df1_inds)>0:
        df1_inds = np.concatenate(df1_inds).tolist()
        return df1.iloc[df1_inds].reset_index(drop=True), df2.iloc[df2_inds].reset_index(drop=True)
    else:
        return [], []


def run_code_for_tile(lvis_tile_x, lvis_tile_y, campaigns, buffer_size=2):
    LVIS_paths = read_LVIS_tile(lvis_tile_x, lvis_tile_y)
    # lvis_tile_lons, lvis_tile_lats = get_tile_bbox(lvis_tile_x, lvis_tile_y)

    if len(LVIS_paths)==0:
        # print('## No LVIS classic files found in the tile')
        return []
    elif all(campaigns[0] not in s for s in LVIS_paths):
        # print(f'## No {campaigns[0]} file found in the tile')
        return []
    elif all(campaigns[1] not in s for s in LVIS_paths):
        # print(f'## No {campaigns[1]} file found in the tile')
        return []

    lvis_fn1 = LVIS_paths[np.where(np.array([campaigns[0] in s for s in LVIS_paths]))[0][0]]
    lvis_fn2 = LVIS_paths[np.where(np.array([campaigns[1] in s for s in LVIS_paths]))[0][0]]
    
    df1 = query_LVIS_shots(lvis_fn1)
    df1 = filter_LVIS_shots(df1)
    df1 = df1.rename(columns=lambda x: x + '_1')
    # print(f'## {len(df1)} shots found in file 1')
    df2 = query_LVIS_shots(lvis_fn2)
    df2 = filter_LVIS_shots(df2)
    df2 = df2.rename(columns=lambda x: x + '_2')
    # print(f'## {len(df2)} shots found in file 2')

    if len(df1)==0 or len(df2)==0:
        # print('## dataframes are empty')
        return []

    idx = create_tree(df1, df2, buffer_size=buffer_size)
    # print(f'## match dataframes')
    df1_match, df2_match = find_crossovers(idx, df1, df2)

    if len(df1_match)>0:
        # print(f'## concatenate dataframes')
        df_concat = pd.concat([df1_match, df2_match], axis=1)
        # print('## Done')
        return df_concat
    else:
        # print('## Done')
        return []


# @contextmanager
# def suppress_output():
#     # Redirect stdout and stderr to devnull
#     old_stdout = sys.stdout
#     old_stderr = sys.stderr
#     sys.stdout = open(os.devnull, 'w')
#     sys.stderr = open(os.devnull, 'w')
#     try:
#         yield
#     finally:
#         # Restore stdout and stderr
#         sys.stdout = old_stdout
#         sys.stderr = old_stderr



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-o', '--output', metavar='STR', type=str, help='output directory')
    argparser.add_argument('-l', '--lvis', metavar='STR', type=str, nargs='+', help='lvis campaign names')
    argparser.add_argument('-c', '--coord', metavar='FLOAT', type=float, nargs='+', help='coordinates: lon_min, lon_max, lat_min, lat_max')
    argparser.add_argument('-b', '--buffer', metavar='FLOAT', type=float, default=2.0, help='buffer size (meters) [default = %(default).2f]')
    argparser.add_argument('-n', '--cores', metavar='INT', type=int, default=10, help='number of cpu cores to use [default = %(default)d]')
    args = argparser.parse_args()

    lon_min, lon_max, lat_min, lat_max = args.coord[0], args.coord[1], args.coord[2], args.coord[3]
    
    tile_x_min, tile_y_min = get_tile_id(lon_min, lat_max, tilesize=6)
    tile_x_max, tile_y_max = get_tile_id(lon_max, lat_min, tilesize=6)
    tiles_x, tiles_y = np.meshgrid(np.arange(tile_x_min, tile_x_max), np.arange(tile_y_min, tile_y_max))
    tiles_xy = list(zip(tiles_x.ravel(), tiles_y.ravel()))

    campaigns = [args.lvis[0], args.lvis[1]]
    
    def parallel_run(tile_xy):
        tile_x = tile_xy[0]
        tile_y = tile_xy[1]
        # with suppress_output():
        df = run_code_for_tile(tile_x, tile_y, campaigns, buffer_size=args.buffer)
        if len(df)>0:
            df.to_parquet(f'{args.output}/X{tile_x:05d}Y{tile_y:05d}.parquet')
            # print(f'## X{tile_x:05d}Y{tile_y:05d}.parquet written to disk')
            return len(df)
        else:
            return 0

    all_lens = []
    pool = multiprocessing.Pool(args.cores)
    for f_len in tqdm(pool.imap_unordered(parallel_run, tiles_xy), total=len(tiles_xy), position=0, desc='## LVIS tiles: '):
        all_lens.append(f_len)
    pool.close()
    
    print(f'## {np.sum(all_lens)} crossovers found {np.sum(np.array(all_lens)>0)} tiles')
    print('## Done')