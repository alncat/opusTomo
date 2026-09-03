'''
Lightweight parser for starfiles
'''

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import fcntl
import hashlib
import json

import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from pathlib import Path
import torch

from . import mrc
from . import lie_tools
from .mrc import LazyImage

_WARP_CTF_CACHE_VERSION = 1


def _warp_ctf_cache_key(csvs, tilt_step, tilt_range, tilt_limit):
    digest = hashlib.sha256()
    digest.update(f'{_WARP_CTF_CACHE_VERSION}|{tilt_step}|{tilt_range}|{tilt_limit}\n'.encode())
    for path in csvs:
        digest.update(os.fsencode(path))
        digest.update(b'\0')
    return digest.hexdigest()


def _default_warp_ctf_cache_path(star_path, tilt_step, tilt_range, tilt_limit):
    config = json.dumps(
        {
            'version': _WARP_CTF_CACHE_VERSION,
            'tilt_step': tilt_step,
            'tilt_range': tilt_range,
            'tilt_limit': tilt_limit,
        },
        sort_keys=True,
    )
    tag = hashlib.sha256(config.encode()).hexdigest()[:12]
    return Path(f'{star_path}.warp_ctf_{tag}.npy')


def _load_warp_ctf_cache(cache_path, cache_key):
    metadata_path = Path(f'{cache_path}.json')
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get('version') != _WARP_CTF_CACHE_VERSION
            or metadata.get('key') != cache_key
        ):
            return None
        cached = np.load(cache_path, mmap_mode='c')
        if list(cached.shape) != metadata.get('shape') or str(cached.dtype) != metadata.get('dtype'):
            return None
        return cached
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_warp_ctf_cache_metadata(cache_path, cache_key, array):
    metadata = {
        'version': _WARP_CTF_CACHE_VERSION,
        'key': cache_key,
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }
    metadata_path = Path(f'{cache_path}.json')
    temporary_path = Path(f'{metadata_path}.tmp-{os.getpid()}')
    temporary_path.write_text(json.dumps(metadata, sort_keys=True))
    os.replace(temporary_path, metadata_path)


@contextmanager
def _warp_ctf_cache_lock(cache_path):
    lock_path = Path(f'{cache_path}.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'a+') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_WARP_CTF_COLUMNS = (
    'TiltAngle',
    'Defocus',
    'Voltage',
    'Cs',
    'Amplitude',
    'Bfactor',
    'Scale',
    'DefocusDelta',
    'AstigmatismAngle',
)


def _parse_warp_ctf_csv(csv_path, tilt_step, tilt_range, tilt_limit):
    """Parse one Warp per-particle CSV without constructing a pandas DataFrame."""
    with open(csv_path, 'r') as f:
        columns = [column.strip() for column in f.readline().split(',')]
        missing = [column for column in _WARP_CTF_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f'{csv_path} is missing Warp CTF columns: {missing}')
        values = np.loadtxt(f, delimiter=',', ndmin=2)

    column = {name: values[:, columns.index(name)] for name in _WARP_CTF_COLUMNS}
    tilt = column['TiltAngle']
    defocus = -column['Defocus'] * 1e10
    defocus_delta = -column['DefocusDelta'] * 1e10
    def_tlt = np.stack(
        [
            tilt,
            defocus + defocus_delta,
            defocus - defocus_delta,
            np.rad2deg(column['AstigmatismAngle']),
            column['Voltage'] / 1e3,
            column['Cs'] * 1e3,
            column['Amplitude'],
            -column['Bfactor'] * 1e20 / 4.0,
            column['Scale'],
        ],
        axis=1,
    )

    len_tilt = int((tilt_range * 2) / tilt_step) + 1
    dummy_tlt = np.zeros((len_tilt, 9), dtype=np.float64)
    dummy_tlt[:, 0] = np.linspace(-tilt_range, tilt_range, len_tilt)
    dummy_tlt[:, 1:3] = 2e4
    dummy_tlt[:, 4] = 300
    dummy_tlt[:, 5] = 2.7

    dummy_angles = dummy_tlt[:, 0]
    def_angles = def_tlt[:, 0]
    idx = np.argmin(np.abs(def_angles[:, None] - dummy_angles[None, :]), axis=1)
    dist = np.abs(def_angles - dummy_angles[idx])
    valid = dist <= tilt_step / 2
    _, counts = np.unique(idx[valid], return_counts=True)
    if np.any(counts > 1) or np.sum(valid) < len(def_angles):
        print('Warning: multiple def_tlt or insufficient rows map to the same dummy_tlt row', csv_path)
    dummy_tlt[idx[valid]] = def_tlt[valid]
    if tilt_limit is not None:
        dummy_tlt[np.abs(dummy_tlt[:, 0]) > tilt_limit, -1] = 0
    return dummy_tlt.astype(np.float32)


def _iter_warp_ctfs(csvs, tilt_step, tilt_range, tilt_limit, workers):
    parse = partial(
        _parse_warp_ctf_csv,
        tilt_step=tilt_step,
        tilt_range=tilt_range,
        tilt_limit=tilt_limit,
    )
    workers = max(1, int(workers))
    if workers == 1:
        for path in csvs:
            yield parse(path)
        return

    # ThreadPoolExecutor.map submits its whole input eagerly on supported Python versions.
    # Bound it to small batches so a 400k-particle run does not allocate 400k futures.
    batch_size = max(1024, workers * 32)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(csvs), batch_size):
            yield from executor.map(parse, csvs[start:start + batch_size])


def _build_warp_ctf_cache(
    cache_path, cache_key, csvs, tilt_step, tilt_range, tilt_limit, workers
):
    started = dt.now()
    print(
        f'Building Warp CTF cache from {len(csvs)} CSV files with {workers} workers: '
        f'{cache_path}',
        flush=True,
    )
    len_tilt = int((tilt_range * 2) / tilt_step) + 1
    temporary_path = Path(f'{cache_path}.tmp-{os.getpid()}')
    try:
        output = np.lib.format.open_memmap(
            temporary_path,
            mode='w+',
            dtype=np.float32,
            shape=(len(csvs), len_tilt, 9),
        )
        for i, ctf_params in enumerate(
            _iter_warp_ctfs(csvs, tilt_step, tilt_range, tilt_limit, workers)
        ):
            output[i] = ctf_params
        output.flush()
        del output
        os.replace(temporary_path, cache_path)
        cached = np.load(cache_path, mmap_mode='c')
        _write_warp_ctf_cache_metadata(cache_path, cache_key, cached)
        print(f'Finished Warp CTF cache in {dt.now() - started}: {cache_path}', flush=True)
        return cached
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
        col_count = 1
        for col in self.headers:
            f.write(f'{col} #{col_count}\n')
            col_count += 1
        for i in self.df.index:
            # TODO: Assumes header and df ordering is consistent
            f.write(' '.join([str(v) for v in self.df.loc[i]]))
            f.write('\n')
        #f.write('\n'.join([' '.join(self.df.loc[i]) for i in range(len(self.df))]))

    def write_df(self, df, outstar):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_\n\n')
        f.write('loop_\n')
        #f.write('\n'.join(df.columns))
        col_count = 1
        for col in df.columns:
            f.write(f'{col} #{col_count}\n')
            col_count += 1
        for i in df.index:
            # TODO: Assumes header and df ordering is consistent
            f.write(' '.join([str(v) for v in df.loc[i]]))
            f.write('\n')

    def write_ind(self, outstar, ind):
        f = open(outstar, 'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_\n\n')
        f.write('loop_\n')
        col_count = 1
        for col in self.headers:
            f.write(f'{col} #{col_count}\n')
            col_count += 1
        count = 0
        for i in ind:
            f.write(' '.join([str(v) for v in self.df.loc[i]]))
            f.write('\n')

    def write_subset(self, outstar, label):
        f = open(outstar,'w')
        f.write('# Created {}\n'.format(dt.now()))
        f.write('\n')
        f.write('data_\n\n')
        f.write('loop_\n')
        col_count = 1
        for col in self.headers:
            f.write(f'{col} #{col_count}\n')
            col_count += 1
        count = 0
        for i in self.df.index:
            if label[i]:
                count += 1
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

        if datadir is not None:
            mrcs = [prefix_paths(x, datadir) for x in mrcs]
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

    def get_drgn3dctfs(self, datadir=None, lazy=True, tilt_step=3., tilt_range=60):
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
        len_tilt = (int((tilt_range*2)/tilt_step)+1)
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
            df['_rlnCtfBfactor'] = np.abs(bfactor)/4.
            #print(tilt_angle)
            #print(euler_i)
            #print(torch.max(torch.acos(axis_i[1][1:, 1].abs()))*180/np.pi)

            subtomo = [image_name, ctf_name, rot_i[0][0].item(), rot_i[0][1].item(), rot_i[0][2].item()]
            df_subtomos.loc[len(df_subtomos)] = subtomo
            self.write_df(df, './subtomos/'+name+'_subtomo.star')

            dummy_tlt = np.zeros((len_tilt, 9))
            dummy_tlt[:, 0] = np.linspace(-tilt_range, tilt_range, len_tilt)
            dummy_tlt[:, 1] = 2e4 #dfu
            dummy_tlt[:, 2] = 2e4 #dfv
            dummy_tlt[:, 4] = 300 #volt
            dummy_tlt[:, 5] = 2.7 #cs


            def_tlt = np.stack([tilt_angle.cpu().numpy(), defocusu, defocusv, defocusangle, voltage, cs, w, np.abs(bfactor)/4., scale], axis=1)
            #sorted_def_tlt = def_tlt[def_tlt[:, 0].argsort()]
            df['_rlnAngleRot'] = 0.
            df['_rlnAngleTilt'] = tilt_angle
            df['_rlnAnglePsi'] = 0.
            dummy_tlt[:len(def_tlt)] = def_tlt

            #mask = np.isclose(sorted_def_tlt[:, 0, None], dummy_tlt[:, 0], atol=tilt_step/2.-0.1)
            ##print(def_tlt[:, 0], dummy_tlt[np.where(mask)[1]][:, 0],)
            #print(sorted_def_tlt, dummy_tlt, mask)
            #mask_indices = np.where(mask)[1]
            #dummy_tlt[mask_indices] = sorted_def_tlt
            #if dummy_tlt[dummy_tlt[:, -1] != 0.].shape[0] != def_tlt.shape[0]:
            #    print(mask_indices, dummy_tlt, def_tlt)
            #assert np.sum(np.abs(dummy_tlt[dummy_tlt[:, -1] != 0.] - sorted_def_tlt)) == 0.

            dummy_tlt = pd.DataFrame(dummy_tlt)
            dummy_tlt.columns = ['_rlnAngleTilt', '_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnVoltage', '_rlnSphericalAberration',
                                 '_rlnAmplitudeContrast', '_rlnCtfBfactor', '_rlnCtfScalefactor']
            #self.write_df(df, './subtomos/'+name+'_ctf.star')
            self.write_df(dummy_tlt, './subtomos/'+name+'_ctf.star')
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

    def get_warp3dctfs(self, datadir=None, lazy=True, tilt_step=2, tilt_range=50, tilt_limit=None,
                       cache_path=None, cache_workers=16):
        '''
        Return ctfs of particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
            cache_path: Optional binary sidecar. A validated cache is memory-mapped;
                otherwise one process builds it while concurrent processes wait.
            cache_workers: Number of bounded threads used only for the first build.
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
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_key = _warp_ctf_cache_key(csvs, tilt_step, tilt_range, tilt_limit)
            cached = _load_warp_ctf_cache(cache_path, cache_key)
            if cached is not None:
                return None, mrc_files, cached
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with _warp_ctf_cache_lock(cache_path):
                # Another distributed rank may have completed the cache while this one waited.
                cached = _load_warp_ctf_cache(cache_path, cache_key)
                if cached is None:
                    cached = _build_warp_ctf_cache(
                        cache_path,
                        cache_key,
                        csvs,
                        tilt_step,
                        tilt_range,
                        tilt_limit,
                        cache_workers,
                    )
            return None, mrc_files, cached

        ctfs = list(
            _iter_warp_ctfs(csvs, tilt_step, tilt_range, tilt_limit, cache_workers)
        )

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




