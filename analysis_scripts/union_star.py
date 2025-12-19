import starfile
import pandas as pd
import numpy as np
import os
import sys
#from io import StringIO

file_name = sys.argv[1]
assert file_name.endswith('.star')
file_name1 = sys.argv[2]
assert file_name1.endswith('.star')
out_file = sys.argv[3]
assert out_file.endswith('.star')

df = starfile.read(file_name)
df1 = starfile.read(file_name1)
print("length of these two starfiles: ", len(df), len(df1))
df = pd.concat([df, df1])
starfile.write(df, f'{out_file}')
print(f'write out {len(df)} particles to {out_file}')
sys.exit(0)
