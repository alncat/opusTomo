'''
Lightweight parser for starfiles
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from pathlib import Path

from . import mrc
from .mrc import LazyImage

class Starfile():

    def __init__(self, headers, df, multibodies=None, multibody_headers=None):
        assert headers == list(df.columns), f'{headers} != {df.columns}'
        self.headers = headers
        self.df = df
        self.multibodies = multibodies
        self.multibody_headers = multibody_headers

    def __len__(self):
        return len(self.df)

    @classmethod
    def get_block(self, f, block_name):
        # get to data block
        block_found = False
        while 1:
            for line in f:
                if line.startswith(block_name):
                    block_found = True
                    break
            if not block_found:
                return "", None
            break
        # get to header loop
        while 1:
            for line in f:
                if line.startswith('loop_'):
                    break
            break
        # get list of column headers
        while 1:
            headers = []
            for line in f:
                if line.startswith('_'):
                    headers.append(line)
                else:
                    break
            break
        # assume all subsequent lines until empty line is the body
        headers = [h.strip().split()[0] for h in headers]
        body = [line]
        for line in f:
            if line.strip() == '':
                break
            body.append(line)
        # put data into an array and instantiate as dataframe
        words = [l.strip().split() for l in body]
        words = np.array(words)
        assert words.ndim == 2, f"Uneven # columns detected in parsing {set([len(x) for x in words])}. Is this a RELION 3.1 starfile?"
        assert words.shape[1] == len(headers), f"Error in parsing. Number of columns {words.shape[1]} != number of headers {len(headers)}"
        data = {h:words[:,i] for i,h in enumerate(headers)}
        df = pd.DataFrame(data=data)
        return headers, df


    @classmethod
    def load_multibody(self, starfile, relion31=False):
        f = open(starfile,'r')
        # get to data block
        BLOCK = 'data_particles' if relion31 else 'data_'
        headers, df = Starfile.get_block(f, BLOCK)
        multibodies = []
        multibody_headers = []
        while 1:
            header, df_tmp = Starfile.get_block(f, 'data_images_body')
            if header == "":
                break
            print(header)
            multibodies.append(df_tmp)
            multibody_headers.append(header)
        return self(headers, df, multibodies=multibodies, multibody_headers=multibody_headers)

    @classmethod
    def load(self, starfile, relion31=False):
        f = open(starfile,'r')
        # get to data block
        BLOCK = 'data_particles' if relion31 else 'data_'
        while 1:
            for line in f:
                if line.startswith(BLOCK):
                    break
            break
        # get to header loop
        while 1:
            for line in f:
                if line.startswith('loop_'):
                    break
            break
        # get list of column headers
        while 1:
            headers = []
            for line in f:
                if line.startswith('_'):
                    headers.append(line)
                else:
                    break
            break
        # assume all subsequent lines until empty line is the body
        headers = [h.strip().split()[0] for h in headers]
        body = [line]
        for line in f:
            if line.strip() == '':
                break
            body.append(line)
        # put data into an array and instantiate as dataframe
        words = [l.strip().split() for l in body]
        words = np.array(words)
        assert words.ndim == 2, f"Uneven # columns detected in parsing {set([len(x) for x in words])}. Is this a RELION 3.1 starfile?"
        assert words.shape[1] == len(headers), f"Error in parsing. Number of columns {words.shape[1]} != number of headers {len(headers)}"
        data = {h:words[:,i] for i,h in enumerate(headers)}
        df = pd.DataFrame(data=data)
        return self(headers, df)

    def write(self, outstar):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_\n\n')
        f.write('loop_\n')
        f.write('\n'.join(self.headers))
        f.write('\n')
        for i in self.df.index:
            # TODO: Assumes header and df ordering is consistent
            f.write(' '.join([str(v) for v in self.df.loc[i]]))
            f.write('\n')
        #f.write('\n'.join([' '.join(self.df.loc[i]) for i in range(len(self.df))]))

    def write_subset(self, outstar, label):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_\n\n')
        f.write('loop_\n')
        f.write('\n'.join(self.headers))
        f.write('\n')
        for i in self.df.index:
            if label[i]:
                # TODO: Assumes header and df ordering is consistent
                f.write(' '.join([str(v) for v in self.df.loc[i]]))
                f.write('\n')

    def get_angpix(self,):
        '''
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        mag = self.df['_rlnMagnification']
        dec_pixel = self.df['_rlnDetectorPixelSize']

        return dataset

    def get_subtomos(self, datadir=None, key='_rlnImageName', lazy=True,):
        '''
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        particles = self.df[key]

        # format is index@path_to_mrc
        #particles = [x for x in particles]
        mrcs = [x for x in particles]
        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'
        header = mrc.parse_header(mrcs[0])
        D = header.D # image size along one dimension in pixels
        dtype = header.dtype
        ## get the number of bytes in extended header
        extbytes = header.fields['next']
        start = 1024+extbytes # start of image data
        dtype = header.dtype

        stride = dtype().itemsize*D*D*D
        dataset = [LazyImage(f, (D,D,D), dtype, start) for f in mrcs]
        #read lazy tomos
        #dataset = []
        #for f in mrcs:
        #    tomo, header = mrc.parse_tomo(f)
        #    dataset.append(tomo)
        #print(dataset)

        if not lazy:
            dataset = np.array([x.get() for x in dataset])
        return dataset

    def get_3dctfs(self, datadir=None, lazy=True):
        '''
        Return ctfs of particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        particles = self.df['_rlnCtfImage']

        # format is index@path_to_mrc
        #particles = [x for x in particles]
        mrcs = [Path(x) for x in particles]
        mrcs = [x.with_suffix('.star') for x in mrcs]
        #print(mrcs)
        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'

        #parse the information of starfile
        ctfs = []
        for star in mrcs:
            f = open(star,'r')
            # get to data block
            BLOCK = 'data_images'
            headers, df = Starfile.get_block(f, BLOCK)
            #print(headers)
            tilt = df['_rlnAngleTilt'].astype(float).to_numpy()
            defocus = df['_rlnDefocusU'].astype(float).to_numpy()
            voltage = df['_rlnVoltage'].astype(float).to_numpy()
            cs = df['_rlnSphericalAberration'].astype(float).to_numpy()
            w = df['_rlnAmplitudeContrast'].astype(float).to_numpy()
            bfactor = df['_rlnCtfBfactor'].astype(float).to_numpy()
            scale = df['_rlnCtfScalefactor'].astype(float).to_numpy()
            #print(scale)
            def_tlt = np.stack([tilt, defocus, voltage, cs, w, bfactor, scale], axis=1)
            #print(def_tlt.shape)
            ctfs.append(def_tlt)
        #print(ctfs)

        #header = mrc.parse_header(mrcs[0])
        #D = header.D # image size along one dimension in pixels
        #dtype = header.dtype
        ### get the number of bytes in extended header
        #extbytes = header.fields['next']
        #start = 1024+extbytes # start of image data
        #dtype = header.dtype

        #stride = dtype().itemsize*D*D*D
        #dataset = [LazyImage(f, (D,D,D), dtype, start) for f in mrcs]
        #read lazy tomos
        #dataset = []
        #for f in mrcs:
        #    tomo, header = mrc.parse_tomo(f)
        #    dataset.append(tomo)
        #print(dataset)

        return ctfs

    def get_particles(self, datadir=None, lazy=True):
        '''
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        particles = self.df['_rlnImageName']

        # format is index@path_to_mrc
        particles = [x.split('@') for x in particles]
        ind = [int(x[0])-1 for x in particles] # convert to 0-based indexing
        mrcs = [x[1] for x in particles]
        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'
        header = mrc.parse_header(mrcs[0])
        D = header.D # image size along one dimension in pixels
        dtype = header.dtype
        stride = dtype().itemsize*D*D
        dataset = [LazyImage(f, (D,D), dtype, 1024+ii*stride) for ii,f in zip(ind, mrcs)]
        if not lazy:
            dataset = np.array([x.get() for x in dataset])
        return dataset

def prefix_paths(mrcs, datadir):
    mrcs1 = ['{}/{}'.format(datadir, os.path.basename(x)) for x in mrcs]
    mrcs2 = ['{}/{}'.format(datadir, x) for x in mrcs]
    try:
        for path in set(mrcs1):
            assert os.path.exists(path)
        mrcs = mrcs1
    except:
        for path in set(mrcs2):
            assert os.path.exists(path), f'{path} not found'
        mrcs = mrcs2
    return mrcs

def csparc_get_particles(csfile, datadir=None, lazy=True):
    metadata = np.load(csfile)
    ind = metadata['blob/idx'] # 0-based indexing
    mrcs = metadata['blob/path'].astype(str).tolist()
    if datadir is not None:
        mrcs = prefix_paths(mrcs, datadir)
    for path in set(mrcs):
        assert os.path.exists(path), f'{path} not found'
    D = metadata[0]['blob/shape'][0]
    dtype = np.float32
    stride = np.float32().itemsize*D*D
    dataset = [LazyImage(f, (D,D), dtype, 1024+ii*stride) for ii,f in zip(ind, mrcs)]
    if not lazy:
        dataset = np.array([x.get() for x in dataset])
    return dataset




