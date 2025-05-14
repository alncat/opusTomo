'''
Lightweight parser for starfiles
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from pathlib import Path
import torch

from . import mrc
from . import lie_tools
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

    def write_df(self, df, outstar):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_images\n\n')
        f.write('loop_\n')
        f.write('\n'.join(df.columns))
        f.write('\n')
        for i in df.index:
            # TODO: Assumes header and df ordering is consistent
            f.write(' '.join([str(v) for v in df.loc[i]]))
            f.write('\n')

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

    def get_drgn_subtomos(self, datadir=None, key='_rlnImageName', lazy=True,):
        '''
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        #particles = self.df[key]
        #group
        particles = self.df.groupby('_rlnGroupName')[key].apply(list)
        #ind = [int(x[0])-1 for x in particles] # convert to 0-based indexing

        # format is index@path_to_mrc
        #particles = [x for x in particles]
        mrcs = []
        inds = []
        for part in particles:
            mrc_i = []
            ind_i = []
            for x in part:
                ind_ii, mrc_ii = x.split('@')
                mrc_i.append(mrc_ii)
                ind_i.append(int(ind_ii)-1)
            inds.append(ind_i)
            mrcs.append(mrc_i)
        #mrcs = [[x for x in part] for part in particles]

        #if datadir is not None:
        #    mrcs = [prefix_paths(mrcs, datadir) for mrc in mrcs]
        #for path in set(mrcs):
        #    assert os.path.exists(path), f'{path} not found'
        header = mrc.parse_header(mrcs[0][0])
        D = header.D # image size along one dimension in pixels
        dtype = header.dtype
        ## get the number of bytes in extended header
        extbytes = header.fields['next']
        start = 1024+extbytes # start of image data
        dtype = header.dtype
        print("start: ", start)
        #print(inds)

        stride = dtype().itemsize*D*D
        dataset = []
        for i in range(len(particles)):
            data = []
            for j in range(len(mrcs[i])):
                #dataset = [[LazyImage(f, (D,D), dtype, start, 1024+ii*stride) for f in mrc] for mrc in mrcs]
                ii = inds[i][j]
                data.append(LazyImage(mrcs[i][j], (D,D), dtype, start+ii*stride))
            dataset.append(data)
        #read lazy tomos
        #dataset = []
        #for f in mrcs:
        #    tomo, header = mrc.parse_tomo(f)
        #    dataset.append(tomo)
        #print(dataset)

        if not lazy:
            dataset = np.array([[x.get() for x in d] for d in dataset])
        return dataset

    def get_drgn3dctfs(self, datadir=None, lazy=True):
        '''
        Return ctfs of particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        '''
        particles = self.df.groupby(['_rlnGroupName'])

        # format is index@path_to_mrc
        #particles = [x for x in particles]
        #parse the information of starfile
        ctfs = []
        rots = []
        rots_0 = []
        trans = []
        df_subtomos = pd.DataFrame(columns=['_rlnImageName', '_rlnCtfImage', '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'])

        # define directory
        directory = Path("./subtomos")
        # check directory
        directory.mkdir(parents=True, exist_ok=True)
        for name, df in particles:
            #print(headers)
            #tilt = df['_rlnAngleTilt'].astype(float).to_numpy()
            #Hack, just use the last before micrograph name
            mic_name = df['_rlnMicrographName'].str.split('_').str[-1].str.split('.').str[0]
            #print(mic_name)
            tilt = mic_name.astype(float).to_numpy()
            defocusu = df['_rlnDefocusU'].astype(float).to_numpy()
            defocusv = df['_rlnDefocusV'].astype(float).to_numpy()
            defocusangle = df['_rlnDefocusAngle'].astype(float).to_numpy()
            voltage = df['_rlnVoltage'].astype(float).to_numpy()
            cs = df['_rlnSphericalAberration'].astype(float).to_numpy()
            w = df['_rlnAmplitudeContrast'].astype(float).to_numpy()
            bfactor = df['_rlnCtfBfactor'].astype(float).to_numpy()
            scale = df['_rlnCtfScalefactor'].astype(float).to_numpy()
            rot = df['_rlnAngleRot'].astype(float).to_numpy()
            tilt = df['_rlnAngleTilt'].astype(float).to_numpy()
            psi = df['_rlnAnglePsi'].astype(float).to_numpy()
            #print(scale)
            name = name[0]
            image_name = name + '.mrc'
            ctf_name = name + '_ctf.mrc'
            rot_i = np.stack([rot, tilt, psi], axis=1)
            rots.append(rot_i)
            rots_0.append(rot_i[0])
            rot_i = torch.from_numpy(rot_i)
            R_i = lie_tools.euler_to_SO3(rot_i)
            R_i = R_i @ R_i[0].T
            #R_i = torch.transpose(R_i, -1, -2) @ R_i[0]
            euler_i = lie_tools.so3_to_euler(R_i.float())
            R_i_veri = lie_tools.euler_to_SO3(euler_i)
            assert torch.abs(torch.min(torch.sum(R_i * R_i_veri, dim=(-1,-2))) - 3) < 1e-4
            axis_i = lie_tools.rot_to_axis(torch.transpose(R_i, -1, -2))
            tilt_angle = axis_i[0]*torch.sign(axis_i[1][:, 1])
            df['_rlnAngleRot'] = euler_i[:, 0]
            df['_rlnAngleTilt'] = euler_i[:, 1]
            df['_rlnAnglePsi'] = euler_i[:, 2]
            df['_rlnCtfBfactor'] = -bfactor/4.
            #print(tilt_angle)
            #print(euler_i)
            #print(torch.max(torch.acos(axis_i[1][1:, 1].abs()))*180/np.pi)

            subtomo = [image_name, ctf_name, rot_i[0][0].item(), rot_i[0][1].item(), rot_i[0][2].item()]
            df_subtomos.loc[len(df_subtomos)] = subtomo
            self.write_df(df, './subtomos/'+name+'_subtomo.star')

            def_tlt = np.stack([tilt_angle.cpu().numpy(), defocusu, defocusv, defocusangle, voltage, cs, w, bfactor, scale], axis=1)
            df['_rlnAngleRot'] = 0.
            df['_rlnAngleTilt'] = tilt_angle
            df['_rlnAnglePsi'] = 0.
            self.write_df(df, './subtomos/'+name+'_ctf.star')
            #save as starfile
            #print(axis_i)
            #print(def_tlt.shape)
            ctfs.append(def_tlt)
        self.write_df(df_subtomos, './subtomos/subtomos.star',)
        #print(ctfs)

        return ctfs, rots, rots_0

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

    def get_warp3dctfs(self, datadir=None, lazy=True, tilt_step=2, tilt_range=50):
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
        mrc_files = [Path(x) for x in particles]
        csvs = [x.with_suffix('.csv') for x in mrc_files]

        #print(mrc_files)
        if datadir is not None:
            #mrcs = prefix_paths(mrcs, datadir)
            mrc_files = ['{}/{}'.format(datadir, x) for x in mrc_files]
            csvs = ['{}/{}'.format(datadir, x) for x in csvs]
        ctfs = []
        tilt_range = int(tilt_range)
        tilt_step = int(tilt_step)
        len_tilt = ((tilt_range*2)//tilt_step+1)
        for csv in csvs:
            dummy_tlt = np.zeros((len_tilt, 7))
            dummy_tlt[:, 0] = np.linspace(-tilt_range, tilt_range, len_tilt)
            dummy_tlt[:, 1] = 2e4
            dummy_tlt[:, 2] = 300
            dummy_tlt[:, 3] = 2.7
            assert os.path.exists(csv), f'{csv} not found'
            #parse csv
            df = pd.read_csv(csv, skipinitialspace=True)
            #df.columns = df.columns.str.strip()
            tilt = df['TiltAngle'].astype(float).to_numpy()
            defocus = -df['Defocus'].astype(float).to_numpy()*1e10 #defocus from warp is in m, -(dfu + dfv)/2
            voltage = df['Voltage'].astype(float).to_numpy()/1e3
            cs = df['Cs'].astype(float).to_numpy()*1e3
            w = df['Amplitude'].astype(float).to_numpy()
            bfactor = -df['Bfactor'].astype(float).to_numpy()*1e20/4. #bfactor from warp is in m^2, scale it down a little bit
            scale = df['Scale'].astype(float).to_numpy()
            defocus_delta = -df['DefocusDelta'].astype(float).to_numpy()*1e10 # -(dfu - dfv)/2
            dfu = defocus + defocus_delta # dfu/2 + dfv/2 + dfu/2 - dfv/2
            dfv = defocus - defocus_delta # dfu/2 + dfv/2 - dfu/2 + dfv/2

            dfangle = df['AstigmatismAngle'].astype(float)/np.pi*180
            #print(scale)
            #def_tlt = np.stack([tilt, dfu, dfv, dfangle, voltage, cs, w, bfactor, scale], axis=1)
            def_tlt = np.stack([tilt, defocus, voltage, cs, w, bfactor, scale], axis=1)
            mask = np.isclose(def_tlt[:, 0, None], dummy_tlt[:, 0], atol=tilt_step/2.-0.1)
            #print(def_tlt[:, 0], dummy_tlt[np.where(mask)[1]][:, 0],)
            mask_indices = np.where(mask)[1]
            dummy_tlt[mask_indices] = def_tlt
            if dummy_tlt[dummy_tlt[:, -1] != 0.].shape[0] != def_tlt.shape[0]:
                print(mask_indices, dummy_tlt, def_tlt)
            assert np.sum(np.abs(dummy_tlt[dummy_tlt[:, -1] != 0.] - def_tlt)) == 0.
            ctfs.append(dummy_tlt)

        #header = mrc.parse_header(mrc_files[0])
        #Dx = header.fields['nx'] # image size along one dimension in pixels
        #Dy = header.fields['ny']
        #Dz = header.fields['nz']
        #dtype = header.dtype
        ### get the number of bytes in extended header
        #extbytes = header.fields['next']
        #start = 1024+extbytes # start of image data
        #dtype = header.dtype

        #stride = dtype().itemsize*Dx*Dy*Dz
        #dataset = [LazyImage(f, (Dx,Dy,Dz), dtype, start) for f in mrcs]

        #_, header = mrc.parse_tomo(mrc_files[0])
        #dataset = [mrc.parse_tomo(f, header)[0] for f in mrc_files]
        dataset = None

        return dataset, mrc_files, ctfs


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
        mrc_files = [Path(x) for x in particles]
        mrcs = [x.with_suffix('.star') for x in mrc_files]
        #print(mrcs)
        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
            mrc_files = ['{}/{}'.format(datadir, x) for x in mrc_files]
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'

        #parse the information of starfile
        ctfs = []
        for star in mrcs:
            f = open(star,'r')
            # get to data block
            BLOCK = 'data_images'
            headers, df = Starfile.get_block(f, BLOCK)
            tilt = df['_rlnAngleTilt'].astype(float).to_numpy()
            defocus = df['_rlnDefocusU'].astype(float).to_numpy()
            #average defocus!
            if '_rlnDefocusV' in df:
                defocus += df['_rlnDefocusV'].astype(float).to_numpy()
                defocus /= 2.
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

        return ctfs, mrc_files

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




