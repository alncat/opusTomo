import starfile
import pandas as pd
import numpy as np
import os
import sys
#from io import StringIO

file_name = sys.argv[1]
assert file_name.endswith('.star')
df = starfile.read(file_name)
mic = sys.argv[2]
df = df[df['rlnMicrographName'].str.contains(mic)] #for retrieving template matching
#df = df[df['rlnMicrographName'] == (mic+".tomostar")] #for exporting m
print(len(df), mic)
df['rlnMicrographName'] = f"{mic}.tomostar"
coords = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
print(f'export the starfile for {mic} to {mic}_norm.star')
starfile.write(df[['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                   ]], mic+'_norm.star')
sys.exit(0)
