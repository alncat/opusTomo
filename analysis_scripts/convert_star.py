import starfile
import pandas as pd
import numpy as np
import os
import sys
#from io import StringIO

file_name = sys.argv[1]
assert file_name.endswith('.star')
df = starfile.read(file_name)
origins = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
origins_new = ['rlnOriginX', 'rlnOriginY', 'rlnOriginZ']
if origins[0] in df.columns:
    angpix = float(sys.argv[2])
    df[origins_new] = df[origins]/angpix
    df.drop(origins, axis=1, inplace=True)
df['rlnRandomSubset'] = (df.index) % 2 + 1
print(f'write out {len(df)} particles with even/odd split to {file_name[:-5]}30')
starfile.write(df, file_name[:-5]+'30.star')
sys.exit(0)
