import starfile
import pandas as pd
import numpy as np
import os
import sys

file_name = sys.argv[1]
assert file_name.endswith('.star')
file_name = file_name[:-5]
df = starfile.read(file_name + '.star')
#sort by source name
df = df.sort_values('wrpSourceName')
df['rlnRandomSubset'] = df['wrpRandomSubset']
df['rlnMicrographName'] = df['wrpSourceName']#f"{n_mic}.tomostar"

angpix = float(sys.argv[2])
df['rlnPixelSize'] = angpix
df['rlnVoltage'] = 300
df['rlnSphericalAberration'] = 2.7
df['rlnDetectorPixelSize'] = angpix
df['rlnMagnification'] = 10000
coords = ['_wrpCoordinateX1', '_wrpCoordinateY1', '_wrpCoordinateZ1']
df['rlnCoordinateX'] = df['wrpCoordinateX1']/angpix
df['rlnCoordinateY'] = df['wrpCoordinateY1']/angpix
df['rlnCoordinateZ'] = df['wrpCoordinateZ1']/angpix
df['rlnAngleRot'] = df['wrpAngleRot1']
df['rlnAngleTilt'] = df['wrpAngleTilt1']
df['rlnAnglePsi'] = df['wrpAnglePsi1']

out_attrs = ['rlnMicrographName', 'rlnRandomSubset', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnPixelSize', 'rlnVoltage', 'rlnDetectorPixelSize', 'rlnMagnification']
#out_attrs += ['rlnOriginX', 'rlnOriginY', 'rlnOriginZ']
print(f'write starfile in Relion\'s format to {file_name}_relion.star')
starfile.write(df[out_attrs], file_name + '_relion.star')
sys.exit(0)

