import starfile
import pandas as pd
import numpy as np
import os
import sys
import pickle as pkl
#from io import StringIO

file_name = sys.argv[1]
df = starfile.read(file_name)
file_name1 = sys.argv[2]
df1 = starfile.read(file_name1)
mics = df1['rlnMicrographName'].unique()
print("length of these two starfiles: ", len(df), len(df1))
print(f"exclude {file_name} from {file_name1}")
coords = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
df_exclude = pd.DataFrame()
# auto-detect original angpix from config.pkl: apix / downfrac
try:
    config = pkl.load(open('config.pkl', 'rb'))
    angpix = config['model_args']['Apix'] * config['dataset_args']['downfrac']
    print(f"auto-detected original angpix from config.pkl: {angpix:.4f} A")
except FileNotFoundError:
    angpix = 3.37
    print(f"config.pkl not found, using default angpix: {angpix} A")
#print(df['rlnMicrographName'].unique(), df1['rlnMicrographName'].unique())
#sys.exit()
#mics = [sys.argv[3]]#['TS_026', 'TS_027', 'TS_028', 'TS_029', 'TS_030', 'TS_034', 'TS_037', 'TS_041', 'TS_043', 'TS_045']
print(file_name1, 'has', len(mics), 'micrographs')
total = 0
for mic in mics:
    #df_ = df[df['rlnMicrographName'] == (mic+'.tomostar')].reset_index()
    df_ = df[df['rlnMicrographName'] == (mic)].reset_index()
    positions = df_[coords]
    #df1_ = df1[df1['rlnMicrographName'] == (mic+'.tomostar')].reset_index()
    df1_ = df1[df1['rlnMicrographName'] == (mic)].reset_index()
    if df_.shape[0] == 0:
        df_exclude = pd.concat([df_exclude, df1_], axis=0)
        continue
    positions1 = df1_[coords]
    dist2 = np.sum((positions.to_numpy()[:, None, :3] - positions1.to_numpy()[None, :, :3]) ** 2, axis=-1)
    min_idx = np.argmin(dist2, axis=1)
    found = np.sqrt(dist2[range(df_.shape[0]), min_idx]) < 136/angpix
    #print(min_idx)
    print(df1_.index, dist2.shape, found.shape)
    df1_exclude = df1_[~df1_.index.isin(min_idx[found])].reset_index()
    total += found.sum()
    #print(df1_.shape, df_.shape, df1_, df1_exclude)
    #print(df1_exclude.shape, df1_.shape)
    print(mic, found.sum(), len(found), 'keep', df1_exclude.shape[0]/df1_.shape[0], 'hit', found.sum()/len(found))
    df_exclude = pd.concat([df_exclude, df1_exclude], axis=0)
print(total, len(df))
df_exclude = df_exclude.drop(columns=['level_0', 'index'])
#print(df_exclude.head())
starfile.write(df_exclude, file_name1[:-5] + '_exclude.star', overwrite=True)
sys.exit(0)

