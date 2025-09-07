import starfile
import pandas as pd
import numpy as np
import os
factor = 0.125 #0.25
import sys
#from io import StringIO

file_name = sys.argv[1]
#df = starfile.read('./match_particles/'+file_name+'/'+file_name+'_are_particles.star')
df = starfile.read(file_name + '.star')
#df['rlnTomoName'] = file_name
mic = sys.argv[2]
#df.rename(columns={'ptmCoordinateX': 'rlnCoordinateX'}, inplace=True)
#df.rename(columns={'ptmCoordinateY': 'rlnCoordinateY'}, inplace=True)
#df.rename(columns={'ptmCoordinateZ': 'rlnCoordinateZ'}, inplace=True)
#dfg = df.groupby('rlnImageName')
#dfg = dfg.first().reset_index()
#print(len(dfg))
#df = df.sort_values(by=['rlnMicrographName', 'rlnImageName'])
#starfile.write(df[:6000], file_name+'_sort.star')
#df = df[df['rlnMicrographName'] == f'metadata/{mic}_Imod/{mic}_st.mrc']
df = df[df['rlnMicrographName'].str.contains(mic)] #for retrieving template matching
#df = df[df['rlnMicrographName'] == (mic+".tomostar")] #for exporting m
#df = df[df['rlnTomoMdocName'] == mic]#[::2]#for ground truth
print(len(df), mic)
#change ctf name
df['rlnMicrographName'] = f"{mic}.tomostar"
coords = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
#if mic == 'TS_026':
#    print(mic)
#    df['rlnCoordinateZ'] = df['rlnCoordinateZ'] - 1000
#df['rlnCoordinateZ'] = 2048 - df['rlnCoordinateZ'] #for ground truth
#df['rlnOriginZ'] = -df['rlnOriginZ']
starfile.write(df[['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                   ]], mic+'_norm_respic.star')
sys.exit(0)

threshold = int(sys.argv[3])/factor
max_z = 800

#df_new = pd.DataFrame()
#print(df['rlnCoordinateZ'].max(), threshold, max_z - threshold)
#for i in range(250,350):
#    i = '00{:03d}'.format(i)
#    df_mic = df[df['rlnMicrographName'] == f'{i}.tomostar'].sort_values(by='rlnImageName')[:750]
#    ##keep particles within a z range
#    ##print(len(df_mic))
#    #df_mic = df_mic[df_mic['rlnCoordinateZ'] > threshold]
#    ##print(len(df_mic))
#    #df_mic = df_mic[df_mic['rlnCoordinateZ'] < max_z - threshold].sort_values(by='rlnImageName')[:800]
#    ##[:threshold]
#    df_new = pd.concat([df_new, df_mic], ignore_index=True)
##print(len(df_new))
##df_new['rlnRondomSubset'] = df.index % 2
#starfile.write(df_new, file_name+'_subset.star')
#sys.exit(0)
#df = df_new
#df = df[df['rlnMicrographName'] == f'{mic}_Imod/{mic}_st.mrc']
#pd.set_option('display.max_colwidth', None)
#print(df['rlnImageName'])
#df = df.groupby('rlnImageName')
#df = df.first().reset_index()
#print(df['rlnImageName'])
#print(len(df))
df['rlnCoordinateX'] /= (480 - 1)
df['rlnCoordinateY'] /= (464 - 1)
df['rlnCoordinateZ'] /= (250 - 1)
starfile.write(df, file_name+'_norm.star')
sys.exit(0)

#df['rlnCoordinateX'] *= 480
#df['rlnCoordinateY'] *= 464
#df['rlnCoordinateZ'] *= 250
#df['rlnCoordinateZ'] = 250 - df['rlnCoordinateZ']

#df['rlnCoordinateZ'] -= 100
#df['rlnCoordinateZ'] -= 200
#df['rlnCoordinateZ'] -= 250

df_out = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']]
file_prefix = file_name.split('/')[-1]
print(file_prefix)

os.makedirs('./visu/', exist_ok=True)
df_out[coords].to_csv('./visu/'+file_prefix+'.txt', index=False, header=False, sep=' ')
df_coords = df_out[coords]

ref_coords = np.loadtxt(sys.argv[4], delimiter=' ')
#ref_coords[:, 2] -= 250

#print(df_coords)
#print(ref_coords)
dist2 = np.sum((df_coords.to_numpy()[:, None, :3] - ref_coords[None, :, :3]) ** 2, axis=-1)
min_idx = np.argmin(dist2, axis=1)
found = np.sqrt(dist2[range(df_coords.shape[0]), min_idx]) < 10
#print(min_idx, np.sqrt(dist2[range(df_coords.shape[0]), min_idx]) < 10)
#i = 0
#for i in range(len(found)//10):
#    print(found[i*10:(i+1)*10])
#print(found[(i+1)*10:])

print(found.sum(), found.sum()/len(found))
sys.exit()
#names = ['27', '28', '29', '30', '34', '37', '41', '43', '45']
#for i in range(len(names)):
#    str_i = file_prefix + names[i]
#    #df1 = starfile.read('./match_particles/'+str_i+'/'+str_i + '_are_particles.star')
#    df1 = starfile.read(str_i + '.star')
#    #df1['rlnTomoName'] = str_i
#    #df1.rename(columns={'ptmCoordinateX': 'rlnCoordinateX'}, inplace=True)
#    #df1.rename(columns={'ptmCoordinateY': 'rlnCoordinateY'}, inplace=True)
#    #df1.rename(columns={'ptmCoordinateZ': 'rlnCoordinateZ'}, inplace=True)
#    df1['rlnCoordinateX'] *=factor
#    df1['rlnCoordinateY'] *=factor
#    df1['rlnCoordinateZ'] *=factor
#    #df1['rlnCoordinateZ'] -= 100
#    #df1['rlnCoordinateZ'] -= 200
#    df1[coords].to_csv(str_i+'.coords', index=False, header=False, sep='\t')
#    df_out = pd.concat([df_out, df1[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']]], ignore_index=True)

#starfile.write(df_out, 'fas_particles.star')
