import numpy as np
import torch
from torch.utils import data
import os
import multiprocessing as mp
from multiprocessing import Pool
import torch.nn.functional as F

from . import fft
from . import mrc
from . import utils
from . import starfile
from . import lie_tools

log = utils.log
def load_structs(mrcs_txt_star, datadir):
    '''
    Load particle stack from a file containing paths to .mrcs files

    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    # not exactly sure what the default behavior should be for the data paths if parsing a starfile
    try:
        star = np.loadtxt(mrcs_txt_star, dtype='S4')
        #read in the list of struct files
        #read in the mainchains of all structures
        particles = []
        sidechains = []
        sidechain_lens = []
        maps = []
        filenames = []
        print(star)
        for i in range(len(star)):
            filename = star[i].astype('U')
            if not filename.isdigit():
                particles.append(np.load(datadir+'/struct/main4/'+star[i].astype('U')+'.main4.npy'))
                sidechains.append(np.load(datadir+'/struct/main4/'+star[i].astype('U')+'.side9.npy'))
                sidechain_lens.append(np.load(datadir+'/struct/main4/'+star[i].astype('U')+'.side9l.npy'))
                map_i = (datadir + '/struct/' + star[i].astype('U') + '/' + 'deposited.mrc')
            else:
                particles.append(np.load(datadir+'/cryostru/main4/'+star[i].astype('U')+'.main4.npy'))
                sidechains.append(np.load(datadir+'/cryostru/main4/'+star[i].astype('U')+'.side9.npy'))
                sidechain_lens.append(np.load(datadir+'/cryostru/main4/'+star[i].astype('U')+'.side9l.npy'))
                map_i = (datadir + '/cryostru/EMD_' + star[i].astype('U')[0] + '/' + star[i].astype('U') + '/' + 'emd_normalized_map.mrc')
            vol_i, header_i = mrc.parse_tomo(map_i)
            #correct_order = header_i.get_order()
            #particles[-1] = np.stack([particles[-1][:, correct_order[0]], particles[-1][:, correct_order[1]], particles[-1][:, correct_order[2]]], axis=-1)
            #if np.any(vol_i.origin != 0.):
            print(star[i].astype('U'), vol_i.Apix, vol_i.origin, vol_i.shape, header_i.get_order())
            maps.append(vol_i)
            filenames.append(star[i].astype('U'))
    except Exception as e:
        raise RuntimeError(e)
    return maps, particles, filenames, sidechains, sidechain_lens

def load_subtomos(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            star = starfile.Starfile.load(mrcs_txt_star, relion31=relion31)
            particles = star.get_subtomos(datadir=datadir, lazy=lazy,)# key='_rlnCtfImage')
            ctfs, ctf_files = star.get_3dctfs(datadir=datadir, lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
                ctfs = star.get_ctfs(datadir=datadir, lazy=lazy)
            else: raise RuntimeError(e)
    else:
        raise NotImplementedError
    return particles, ctfs, ctf_files

def load_warp_subtomos(mrcs_txt_star, lazy=False, datadir=None, relion31=False, tilt_step=2, tilt_range=50, tilt_limit=None):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            star = starfile.Starfile.load(mrcs_txt_star, relion31=relion31)
            particles = star.get_subtomos(datadir=datadir, lazy=lazy,)# key='_rlnCtfImage')
            warp_ctfs, ctf_files, ctf_params = star.get_warp3dctfs(datadir=datadir, lazy=lazy, tilt_step=tilt_step, tilt_range=tilt_range, tilt_limit=tilt_limit)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
                ctfs, ctfs_files, ctf_params = star.get_warp3dctfs(datadir=datadir, lazy=lazy, tilt_step=tilt_step, tilt_range=tilt_range, tilt_limit=tilt_limit)
            else: raise RuntimeError(e)
    else:
        raise NotImplementedError
    return particles, ctf_params, ctf_files, warp_ctfs

def load_drgn_subtomos(mrcs_txt_star, lazy=False, datadir=None, relion31=False, tilt_step=3, tilt_range=60):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            star = starfile.Starfile.load(mrcs_txt_star, relion31=relion31)
            particles = star.get_drgn_subtomos(datadir=datadir, lazy=lazy,)# key='_rlnCtfImage')
            ctfs, rots, rots0 = star.get_drgn3dctfs(datadir=datadir, lazy=lazy, tilt_step=tilt_step, tilt_range=tilt_range)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
                ctfs, rots, rots0 = star.get_drgn3dctfs(datadir=datadir, lazy=lazy, tilt_step=tilt_step, tilt_range=tilt_range)
            else: raise RuntimeError(e)
    else:
        raise NotImplementedError
    return particles, ctfs, rots, rots0

def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    '''
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    '''
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star) # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir, lazy=lazy)
            else: raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    return particles

class LazyStructMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 datadir=None, apix=1.125):
        #assert not keepreal, 'Not implemented error'
        particles, structs, filenames, sidechains, sidechain_lens = load_structs(mrcfile, datadir=datadir)
        assert len(particles) == len(structs)
        N = len(particles)
        ny, nx, nz = particles[0].get().shape
        log('Loaded {} images, the first one is of dimension {}x{}x{}'.format(N, ny, nx, nz))
        self.particles = particles
        self.sidechains = sidechains
        self.sidechain_lens = sidechain_lens
        self.filenames = filenames
        self.N = N
        self.invert_data = invert_data
        self.real_data = real_data
        #if norm is None:
        #    norm = self.estimate_normalization()
        self.norm = norm
        self.apix = apix
        self.structs = structs
        self.training_cubic_size = 64
        self.D = self.training_cubic_size + 1#set D to training size
        self.training_length = 512
        x_idx = torch.linspace(-1., 1., self.training_cubic_size) #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        self.grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0) #(1, D, H, W, 3)

    def estimate_normalization(self, n=100):
        assert self.real_data
        n = min(n,self.N)
        imgs = np.asarray([np.mean(self.particles[i].get()) for i in range(0,self.N, self.N//n)])
        log('first image: {}'.format(imgs[0]))
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        return norm

    def get(self, i):
        sample_R = lie_tools.random_biased_SO3(1, bias=0.25) #(1, 3, 3)
        grid_R = self.grid @ sample_R.unsqueeze(1).unsqueeze(1)
        img = self.particles[i].get()
        stru = self.structs[i]
        sidechains = self.sidechains[i]
        sidechain_lens = self.sidechain_lens[i]
        ori_apix = self.particles[i].Apix
        origin = self.particles[i].origin
        order = [2 - x for x in self.particles[i].order]
        order.reverse()
        #order = self.particles[i].order
        stru = stru - origin*ori_apix
        sidechains = sidechains - origin*ori_apix
        #stru = stru + origin*ori_apix
        #sidechains = sidechains + origin*ori_apix

        #NOTE: permute the map to get correct order
        img = np.transpose(img, axes=order)
        #print(self.filenames[i], img.shape, order)

        center = np.array(img.shape)//2
        min_coords = np.min(stru/ori_apix, axis=0).astype(np.int32)
        #if np.any(min_coords < 0):
        #    stru = stru + center*ori_apix
        #    min_coords = np.min(stru/ori_apix, axis=0).astype(np.int32)
        mean_coords = np.mean(stru/ori_apix, axis=0).astype(np.int32)
        min_coords = np.min(stru/ori_apix, axis=0).astype(np.int32)
        max_coords = np.max(stru/ori_apix, axis=0).astype(np.int32)
        coord_range = max_coords - min_coords

        #compute crop size
        scale = ori_apix/self.apix
        crop_size = int(self.training_cubic_size/scale)
        assert np.all(max_coords >= crop_size), f"{self.filenames[i]}, {min_coords}, {max_coords}, {origin}"

        #print(self.filenames[i], ori_apix, self.particles[i].origin)
        #for j in range(3):
        #    if max_coords[j] - crop_size - 2 < min_coords[j]:
        #        min_coords[j] = max_coords[j] - crop_size
        #print("coords: ", min_coords, max_coords, img.shape, mean_coords)
        #assert mean_coords == center, f"{stru.shape}, {mean_coords}, {center}"
        #assert np.all(min_coords > 0), f"{self.filenames[i]}, {min_coords}, {max_coords}"
        L = len(stru)//4
        assert len(stru) % 4 == 0 and L > 0
        stru = stru.reshape(L, 4, 3)

        #start = [np.random.randint(max(min_coords[j]-4, 0), max(max_coords[j]-crop_size, min_coords[j])) for j in range(3)]
        while True:
            start_idx = np.random.randint(L)
            start = (stru[start_idx, 1, :]/ori_apix).astype(np.int32)
            #the start must fill a cubic of size crop_size
            start = np.minimum(np.array(img.shape) - crop_size, start - crop_size//2)
            start = np.maximum(start, 0)
            #start = np.array(start)
            end = np.array([crop_size]*3) + start
            #print("sampled: ", start, end, img.shape)
            sidechains = sidechains[np.logical_and(np.all(stru[:, 1, :] < (end)*ori_apix, axis=-1), np.all(stru[:, 1, :] >= (start)*ori_apix, axis=-1))]
            sidechain_lens = sidechain_lens[np.logical_and(np.all(stru[:, 1, :] < (end)*ori_apix, axis=-1), np.all(stru[:, 1, :] >= (start)*ori_apix, axis=-1))]
            stru = stru[np.logical_and(np.all(stru[:, 1, :] < (end)*ori_apix, axis=-1), np.all(stru[:, 1, :] >= (start)*ori_apix, axis=-1))]

            #crop struct to 512 long
            max_len = 512
            if stru.shape[0] > max_len:
                crop_idx = np.random.randint(0, stru.shape[0]-max_len)
                stru = stru[crop_idx:crop_idx+max_len]
                sidechains = sidechains[crop_idx:crop_idx+max_len]
                sidechain_lens = sidechain_lens[crop_idx:crop_idx+max_len]
            assert stru.shape[0] > 0, f"{self.filenames[i]}, {stru.shape}"

            #NOTE: remember the x y z cooresponds to 2, 1, 0 in index
            img = img[start[2]:end[2],
                      start[1]:end[1],
                      start[0]:end[0]]

            output_size = int(img.shape[-1]*scale)
            #print(output_size, self.particles[i].Apix, self.apix)
            #cropping fourier
            img = torch.from_numpy(img)
            #img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=[self.training_cubic_size]*3, mode='trilinear', align_corners=utils.ALIGN_CORNERS)
            #sample R
            #print(img.shape, grid_R.shape)
            img = F.grid_sample(img.unsqueeze(0).unsqueeze(0), grid_R, mode='bilinear', align_corners=utils.ALIGN_CORNERS)
            img = (img - img.mean())/(img.std() + 1e-5)
            img = img.squeeze()
            if img.std() > 0.5:
                break
        #select structure that is inside the box
        #stru = stru[np.logical_and(np.all(stru[:, 1, :] < (end)*ori_apix, axis=-1), np.all(stru[:, 1, :] >= (start)*ori_apix, axis=-1))]
        #center stru
        #stru = torch.from_numpy(stru).float()
        center = (self.training_cubic_size - 1.)/2.
        stru = stru - (start)*ori_apix - center*self.apix
        stru = stru @ np.transpose(sample_R.numpy(), [0, 2, 1]) #(L, 4, 3), (1, 3, 3)
        #stru = stru + (crop_size/2)*ori_apix
        stru = stru + center*self.apix

        #sidechain_lens_o = sidechain_lens
        sidechain_lens = sidechain_lens[np.logical_and(np.all(stru[:, 1, :] < self.training_cubic_size*self.apix, axis=-1), np.all(stru[:, 1, :] >= 0., axis=-1))]
        #sidechains_o = sidechains
        sidechains = sidechains[np.logical_and(np.all(stru[:, 1, :] < self.training_cubic_size*self.apix, axis=-1), np.all(stru[:, 1, :] >= 0., axis=-1))]

        stru = stru[np.logical_and(np.all(stru[:, 1, :] < self.training_cubic_size*self.apix, axis=-1), np.all(stru[:, 1, :] >= 0., axis=-1))]

        #retrieve sidechains
        sidechains = sidechains.reshape(stru.shape[0]*10, 3)
        sidechain_lens = sidechain_lens.reshape(stru.shape[0]*10)
        sidechains = sidechains[sidechain_lens]

        sidechains = sidechains - (start)*ori_apix - center*self.apix
        sidechains = sidechains @ np.transpose(sample_R.numpy(), [0, 2, 1]) #(L, 4, 3), (1, 3, 3)
        sidechains = sidechains + center*self.apix
        #if self.filenames[i] == '6WCA':
        #    print(sidechains_o)

        #convert to a single list
        #sidechains_out = None
        #for j in range(len(sidechains)):
        #    if sidechain_lens[j] > 0:
        #        if sidechains_out is None:
        #            sidechains_out = sidechains[j, :sidechain_lens[j]]
        #        else:
        #            sidechains_out = np.concatenate([sidechains_out, sidechains[j, :sidechain_lens[j]]], axis=0)
        #if sidechains_out is None:
        #    print(sidechains_o, sidechain_lens_o)
        return img, stru, sidechains, self.filenames[i]

    def get_batch(self, batch,):
        imgs = []
        strus = []
        sidechains = []
        filenames = []
        for i in range(len(batch)):
            img, stru, sidechain, filename = self.get(batch[i])
            imgs.append(img)
            strus.append(stru)
            sidechains.append(sidechain)
            filenames.append(filename)
        return np.concatenate(imgs, axis=0), strus, sidechain, filenames

    def __len__(self):
        return self.N

    # the getter of the dataset
    def __getitem__(self, index):
        return self.get(index), index

class LazyTomoDRGNMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 window=True, datadir=None, relion31=False, window_r=0.85, in_mem=False, downfrac=0.75, tilt_step=3, tilt_range=60):
        #assert not keepreal, 'Not implemented error'
        assert mrcfile.endswith('.star')
        particles, ctfs, rots, rots0 = load_drgn_subtomos(mrcfile, True, datadir=datadir, relion31=relion31, tilt_step=tilt_step, tilt_range=60)
        assert len(particles) == len(ctfs)
        N = len(particles)
        ny, nx = particles[0][0].get().shape
        nz = len(particles[0])
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{}x{} images'.format(N, ny, nx, nz))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        self.real_data = real_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        log('Image Mean, Std are {} +/- {}'.format(*self.norm))
        self.window = window_cos_mask(ny, window_r, .95) if window else None
        self.in_mem = in_mem
        self.ctfs = ctfs
        self.rots = rots
        self.rots0 = rots0
        print("ctf is of shape: ", self.ctfs[0].shape)

    def estimate_normalization(self, n=100):
        assert self.real_data
        n = min(n,self.N)
        if self.real_data:
            imgs = []
            for i in range(0, self.N, self.N//n):
                imgs.extend([self.particles[i][j].get() for j in range(len(self.particles[i]))])
            imgs = np.asarray(imgs)
        #if self.invert_data: imgs *= -1
        #log('first image: {}'.format(imgs[0]))
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        return norm

    def get(self, i):
        part = [self.particles[i][j].get() for j in range(len(self.particles[i]))]
        part = np.asarray(part)
        #part *= -1/self.norm[1]
        ctf = self.ctfs[i]
        rot = self.rots[i]
        return part, ctf, rot

    def __len__(self):
        return self.N

    # the getter of the dataset
    def __getitem__(self, index):
        return self.get(index), index

class LazyTomoWARPMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 window=True, datadir=None, relion31=False, window_r=0.85, in_mem=False, downfrac=0.75,
                 tilt_step=2, tilt_range=50, tilt_limit=None, read_ctf=False, use_float16=False, rank=0):
        #assert not keepreal, 'Not implemented error'
        assert mrcfile.endswith('.star')
        log(f"the maximum tilt_range of loading tilt series is {tilt_range}, \
            the tilt_step of loaded tilt series is {tilt_step}, \
            the tilt_limit of loaded tilt series is {tilt_limit}")
        particles, ctfs, ctf_files, warp_ctfs = load_warp_subtomos(mrcfile, True, datadir=datadir, relion31=relion31,
                                                        tilt_step=tilt_step, tilt_range=tilt_range, tilt_limit=tilt_limit)
        self.tilt_step = tilt_step
        self.tilt_range = tilt_range
        assert len(particles) == len(ctf_files)
        N = len(particles)
        ny, nx, nz = particles[0].get().shape
        assert ny == nx == nz, "Images must be cubic"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{}x{} images, while float16: {}'.format(N, ny, nx, nz, use_float16))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        self.real_data = real_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        log('Subtomogram Mean, Std are {} +/- {}'.format(*self.norm))
        self.window = window_cos_mask(ny, window_r, .95) if window else None
        self.in_mem = in_mem
        self.ctfs = np.stack(ctfs, axis=0)
        self.ctf_files = ctf_files
        self.warp_ctfs = warp_ctfs
        self.read_ctf = read_ctf
        self.use_float16 = use_float16
        if rank == 0:
            log("The 3DCTFs are of shape: {}".format(self.ctfs.shape))
            log("The first 3DCTF is: {}".format(self.ctfs[0]))

    def estimate_normalization(self, n=100):
        assert self.real_data
        n = min(n,self.N)
        if self.real_data:
            imgs = np.asarray([self.particles[i].get().astype(np.float32) for i in range(0,self.N, self.N//n)])
        #if self.invert_data: imgs *= -1
        #log('first image: {}'.format(imgs[0]))
        imgs = np.nan_to_num(imgs[:, :self.D//4, :, :])
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        return norm

    def get(self, i):
        part = np.nan_to_num(self.particles[i].get())
        #standardize it
        if self.use_float16:
            part = (part.astype(np.float16) - self.norm[0])/self.norm[1]
        else:
            part = (part - self.norm[0])/self.norm[1]
        ctf = self.ctfs[i]
        if self.read_ctf:
            ctf = self.warp_ctfs[i].get()
        return part, ctf, self.ctf_files[i]

    def __len__(self):
        return self.N

    # the getter of the dataset
    def __getitem__(self, index):
        return self.get(index), index


class LazyTomoMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 window=True, datadir=None, relion31=False, window_r=0.85, in_mem=False, downfrac=0.75, use_float16=False, rank=0):
        #assert not keepreal, 'Not implemented error'
        assert mrcfile.endswith('.star')
        particles, ctfs, ctf_files = load_subtomos(mrcfile, True, datadir=datadir, relion31=relion31)
        assert len(particles) == len(ctfs)
        N = len(particles)
        ny, nx, nz = particles[0].get().shape
        assert ny == nx == nz, "Images must be cubic"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{}x{} images, float16: {}'.format(N, ny, nx, nz, use_float16))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        self.real_data = real_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        log('Image Mean, Std are {} +/- {}'.format(*self.norm))
        self.window = window_cos_mask(ny, window_r, .95) if window else None
        self.in_mem = in_mem
        self.ctfs = np.stack(ctfs, axis=0)
        self.ctf_files = ctf_files
        self.use_float16 = use_float16
        if rank == 0:
            log("ctf is of shape: {}".format(self.ctfs.shape))

    def estimate_normalization(self, n=100):
        assert self.real_data
        n = min(n,self.N)
        if self.real_data:
            imgs = np.asarray([self.particles[i].get() for i in range(0,self.N, self.N//n)])
        #if self.invert_data: imgs *= -1
        #log('first image: {}'.format(imgs[0]))
        imgs = imgs[:, :self.D//4, :, :]
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        return norm

    def get(self, i):
        part = self.particles[i].get()
        #standardize it
        if self.use_float16:
            part = (part.astype(np.float16) - self.norm[0])/self.norm[1]
        else:
            part = (part - self.norm[0])/self.norm[1]

        #part *= -1/self.norm[1]
        ctf = self.ctfs[i]
        return part, ctf, self.ctf_files[i]

    def get_batch(self, batch):
        imgs = []
        ctfs = []
        for i in range(len(batch)):
            img, ctf = self.get(batch[i])
            imgs.append(img)
            ctfs.append(ctf)
        return np.concatenate(imgs, axis=0), np.concatenate(ctfs, axis=0)

    def __len__(self):
        return self.N

    # the getter of the dataset
    def __getitem__(self, index):
        return self.get(index), index

class LazyMRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file -- images loaded on the fly
    '''
    def __init__(self, mrcfile, norm=None, real_data=True, keepreal=False, invert_data=False, ind=None,
                 window=True, datadir=None, relion31=False, window_r=0.85, in_mem=True, downfrac=0.75):
        #assert not keepreal, 'Not implemented error'
        particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
        N = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        self.particles = particles
        self.N = N
        self.D = ny + 1 # after symmetrizing HT
        self.invert_data = invert_data
        self.real_data = real_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        #self.window = window_mask(ny, window_r, .99) if window else None
        self.window = window_cos_mask(ny, window_r, .95) if window else None
        if in_mem:
            log('Reading all images into memory!')
            #downsample images
            particles = []
            self.down_size = int((ny*downfrac)//2)*2
            down_scale = self.down_size/ny
            log(f'downsample to {self.down_size}')
            if self.down_size == ny:
                for i in range(0, self.N):
                    particles.append(torch.tensor(self.particles[i].get()).unsqueeze(0))
            else:
                for i in range(0, self.N):
                    y_fft = fft.torch_fft2_center(torch.tensor(self.particles[i].get())).unsqueeze(0)
                    y_fft_s = torch.fft.fftshift(y_fft, dim=(-2))
                    y_fft_crop = utils.crop_fft(y_fft_s, self.down_size)
                    particles.append(y_fft_crop)
            self.particles = torch.concat(particles, dim=0)
            #self.particles = np.asarray([self.particles[i].get() for i in range(0, self.N)])
        self.in_mem = in_mem

    def estimate_normalization(self, n=1000):
        n = min(n,self.N)
        if self.real_data:
            imgs = np.asarray([self.particles[i].get() for i in range(0,self.N, self.N//n)])
        else:
            imgs = np.asarray([fft.ht2_center(self.particles[i].get()) for i in range(0,self.N, self.N//n)])
        if self.invert_data: imgs *= -1
        if not self.real_data:
            imgs = fft.symmetrize_ht(imgs)
        log('first image: {}'.format(imgs[0]))
        norm = [np.mean(imgs), np.std(imgs)]
        if self.real_data:
            log('Image Mean, Std are {} +/- {}'.format(*norm))
        else:
            log('Normalizing HT by {} +/- {}'.format(*norm))
        norm[0] = 0
        return norm

    def get(self, i):
        if self.in_mem:
            part = self.particles[i]
            if self.down_size != self.D-1:
                #padding zeros
                part = utils.pad_fft(part, self.D-1)
                part = torch.fft.ifftshift(part, dim=(-2))
                part = fft.torch_ifft2_center(part)
            return part
            #return self.particles[i]
        img = self.particles[i].get()
        if self.real_data:
            return img
        if self.window is not None:
            img *= self.window
        img = fft.ht2_center(img).astype(np.float32)
        if self.invert_data: img *= -1
        img = fft.symmetrize_ht(img)
        img = (img - self.norm[0])/self.norm[1]
        return img

    def get_batch(self, batch):
        imgs = []
        for i in range(len(batch)):
            imgs.append(self.get(batch[i]))
        return np.concatenate(imgs, axis=0)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get(index), index

class SplitBatchSampler(data.Sampler):
    def __init__(self, batch_size, ind, split, rank=0, size=1):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        #filter poses_ind not in split
        ind = ind[np.isin(ind.numpy(), split.numpy())]
        self.num_samples = len(ind)
        #filter particles not in this rank
        ind_size = self.num_samples//size
        self.ind = ind[rank*ind_size:(rank+1)*ind_size]
        self.num_samples = len(self.ind)
        print("num_samples: ", self.num_samples)
        print(self.ind)

    def __iter__(self,):
        rand_perms = torch.randperm(self.num_samples)
        #print("rand_perms: ", rand_perms)
        for i in range(self.num_samples//self.batch_size):
            sample_ind = rand_perms[i*self.batch_size:(i+1)*self.batch_size]
            yield self.ind[sample_ind]

    def __len__(self,):
        return self.num_samples//self.batch_size

class ClassSplitBatchDistSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind, split, rank=0, size=1):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors

        #filter poses_ind not in split
        poses_ind_new = []
        for x in self.poses_ind:
            filtered= x[np.isin(x.numpy(), split.numpy())]
            #filter particles not in this rank
            local_size = len(filtered)//size
            poses_ind_new.append(filtered[rank*local_size:(rank+1)*local_size])
        self.poses_ind = poses_ind_new
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]
        print(self.ns)

        self.num_samples = sum(self.ns)
        self.rank = rank
        self.size = size
        if rank == 0:
            print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(len(x)) for x in self.poses_ind]
        #print("rand_perms: ", rand_perms)
        #print("ns: ", self.ns)
        #print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if not found:
                break
            start = current_ind[rand_pose]
            current_ind[rand_pose] += self.batch_size
            current_num_samples += self.batch_size
            sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
            #indexing poses_ind
            yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

class ClassSplitBatchHvdSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind, split, rank=0, size=1):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors

        #filter poses_ind not in split
        poses_ind_new = []
        for x in self.poses_ind:
            filtered= x[np.isin(x.numpy(), split.numpy())]
            #filter particles not in this rank
            local_size = len(filtered)//size
            poses_ind_new.append(filtered[rank*local_size:(rank+1)*local_size])
        self.poses_ind = poses_ind_new
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]
        print(self.ns)

        self.num_samples = sum(self.ns)
        self.rank = rank
        self.size = size
        if rank == 0:
            print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(len(x)) for x in self.poses_ind]
        #print("rand_perms: ", rand_perms)
        #print("ns: ", self.ns)
        #print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if not found:
                break
            start = current_ind[rand_pose]
            current_ind[rand_pose] += self.batch_size
            current_num_samples += self.batch_size
            sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
            #indexing poses_ind
            yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

class ClassSplitBatchSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind, split):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors
        #filter poses_ind not in split
        poses_ind_new = []
        for x in self.poses_ind:
            poses_ind_new.append(x[np.isin(x.numpy(), split.numpy())])
        self.poses_ind = poses_ind_new
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]

        self.num_samples = sum(self.ns)
        print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(n) for n in self.ns]
        #print("rand_perms: ", rand_perms)
        print("ns: ", self.ns)
        print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if found:
                start = current_ind[rand_pose]
                current_ind[rand_pose] += self.batch_size
                current_num_samples += self.batch_size

                sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
                #indexing poses_ind
                yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

class ClassBatchSampler(data.Sampler):
    def __init__(self, batch_size, poses_ind):
        #self.weights = torch.as_tensor(weights)
        self.batch_size = batch_size
        self.poses_ind = poses_ind # list of torch tensors
        self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]

        self.num_samples = sum(self.ns)
        print("num_samples: ", self.num_samples)

    def __iter__(self,):
        current_num_samples = 0
        current_ind = [0 for _ in self.ns]
        rand_perms = [torch.randperm(n) for n in self.ns]
        #print("rand_perms: ", rand_perms)
        print("ns: ", self.ns)
        print("current_ind: ", current_ind)
        for _ in range(self.num_samples//self.batch_size):
            #rand_tensor = torch.multinomial(self.weights, 1, self.replacement)
            found = False
            while not found and current_num_samples < self.num_samples:
                rand_pose = torch.randint(high=len(self.ns), size=(1,), dtype=torch.int64)
                if current_ind[rand_pose] < self.ns[rand_pose]:
                    found = True
            if found:
                start = current_ind[rand_pose]
                current_ind[rand_pose] += self.batch_size
                current_num_samples += self.batch_size

                sample_ind = rand_perms[rand_pose][start:start + self.batch_size]
                #indexing poses_ind
                yield self.poses_ind[rand_pose][sample_ind]
        print("final_ind: ", current_ind)

    def __len__(self,):
        return self.num_samples//self.batch_size

def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r-in_rad)/(out_rad-in_rad)))
    return mask

def window_cos_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, D, endpoint=False, dtype=np.float32))
    r = (x0**2 + x1**2)**.5
    mask = np.minimum(1., np.maximum(0.0, (r-in_rad)/(out_rad - in_rad)))
    mask = 0.5 + 0.5*np.cos(mask*np.pi)
    return mask

class VolData(data.Dataset):
    '''
    Class representing an .mrcs stack file
    '''
    def __init__(self, mrcfile, norm=None, invert_data=False, datadir=None, relion31=False, max_threads=16, window_r=0.85):
        particles = load_particles(mrcfile, False, datadir=datadir, relion31=relion31)
        N, ny, nx = particles.shape
        assert N == ny == nx, "Images must be cubic"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))

        if invert_data: particles *= -1

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        #particles = (particles - norm[0])/norm[1]
        #log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.volume = torch.from_numpy(self.particles)
        self.N = N
        self.D = particles.shape[1] # ny
        self.norm = norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self):
        return self.volume

    def center_of_mass(self,):
        mass = ((self.volume > 0)*self.volume).sum()
        x_idx = torch.linspace(0, self.N-1, self.N) - self.N/2 #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1)
        vol = ((self.volume > 0).float()*self.volume).unsqueeze(-1)
        center = vol*grid
        center = center.sum(dim=(0,1,2))
        assert mass.item() > 0
        center /= mass
        center = torch.where(center > 0, (center + 0.5).int(), (center - 0.5).int()).float()
        centered = (grid - center)*vol
        radius = (centered).pow(2)
        r = torch.sqrt(radius.sum(dim=(0,1,2))/mass)
        #principal axes
        matrix = -centered.unsqueeze(-1) * centered.unsqueeze(-2)
        radius_sum = torch.eye(3) * (radius.sum(dim=-1, keepdim=True).unsqueeze(-1))
        #matrix = ((matrix+radius_sum)*vol.unsqueeze(-1)).sum(dim=(0, 1, 2))
        matrix = ((-matrix)*vol.unsqueeze(-1)).sum(dim=(0, 1, 2))
        eigvals, eigvecs = np.linalg.eig(matrix.numpy())
        indices = np.argsort(eigvals)
        eigvals = eigvals[indices]
        #print(matrix, eigvals[indices])
        eigvecs = torch.from_numpy(eigvecs[:, indices].T) # eigvecs[0] is the first eigen vector with largest eigenvalues
        print(eigvecs @ (matrix @ eigvecs.T), eigvals)
        print("r:", r, "eigvals:", np.sqrt(eigvals/mass))
        r = np.sqrt(eigvals/mass)

        return center, r, eigvecs


class MRCData(data.Dataset):
    '''
    Class representing an .mrcs stack file
    '''
    def __init__(self, mrcfile, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, relion31=False, max_threads=16, window_r=0.85):
        if keepreal:
            raise NotImplementedError
        if ind is not None:
            particles = load_particles(mrcfile, True, datadir=datadir, relion31=relion31)
            particles = np.array([particles[i].get() for i in ind])
        else:
            particles = load_particles(mrcfile, False, datadir=datadir, relion31=relion31)
        N, ny, nx = particles.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))

        # Real space window
        if window:
            log(f'Windowing images with radius {window_r}')
            particles *= window_mask(ny, window_r, .99)

        # compute HT
        log('Computing FFT')
        max_threads = min(max_threads, mp.cpu_count())
        if max_threads > 1:
            log(f'Spawning {max_threads} processes')
            with Pool(max_threads) as p:
                particles = np.asarray(p.map(fft.ht2_center, particles), dtype=np.float32)
        else:
            particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32)
            log('Converted to FFT')

        if invert_data: particles *= -1

        # symmetrize HT
        log('Symmetrizing image data')
        particles = fft.symmetrize_ht(particles)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.N = N
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            log('Normalized real space images by {}'.format(particles_real.std()))
            self.particles_real /= particles_real.std()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class PreprocessedMRCData(data.Dataset):
    '''
    '''
    def __init__(self, mrcfile, norm=None, ind=None):
        particles = load_particles(mrcfile, False)
        if ind is not None:
            particles = particles[ind]
        log(f'Loaded {len(particles)} {particles.shape[1]}x{particles.shape[1]} images')
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))
        self.particles = particles
        self.N = len(particles)
        self.D = particles.shape[1] # ny + 1 after symmetrizing HT
        self.norm = norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

class TiltMRCData(data.Dataset):
    '''
    Class representing an .mrcs tilt series pair
    '''
    def __init__(self, mrcfile, mrcfile_tilt, norm=None, keepreal=False, invert_data=False, ind=None, window=True, datadir=None, window_r=0.85):
        if ind is not None:
            particles_real = load_particles(mrcfile, True, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, True, datadir)
            particles_real = np.array([particles_real[i].get() for i in ind], dtype=np.float32)
            particles_tilt_real = np.array([particles_tilt_real[i].get() for i in ind], dtype=np.float32)
        else:
            particles_real = load_particles(mrcfile, False, datadir)
            particles_tilt_real = load_particles(mrcfile_tilt, False, datadir)

        N, ny, nx = particles_real.shape
        assert ny == nx, "Images must be square"
        assert ny % 2 == 0, "Image size must be even"
        log('Loaded {} {}x{} images'.format(N, ny, nx))
        assert particles_tilt_real.shape == (N, ny, nx), "Tilt series pair must have same dimensions as untilted particles"
        log('Loaded {} {}x{} tilt pair images'.format(N, ny, nx))

        # Real space window
        if window:
            m = window_mask(ny, window_r, .99)
            particles_real *= m
            particles_tilt_real *= m

        # compute HT
        particles = np.asarray([fft.ht2_center(img) for img in particles_real]).astype(np.float32)
        particles_tilt = np.asarray([fft.ht2_center(img) for img in particles_tilt_real]).astype(np.float32)
        if invert_data:
            particles *= -1
            particles_tilt *= -1

        # symmetrize HT
        particles = fft.symmetrize_ht(particles)
        particles_tilt = fft.symmetrize_ht(particles_tilt)

        # normalize
        if norm is None:
            norm  = [np.mean(particles), np.std(particles)]
            norm[0] = 0
        particles = (particles - norm[0])/norm[1]
        particles_tilt = (particles_tilt - norm[0])/norm[1]
        log('Normalized HT by {} +/- {}'.format(*norm))

        self.particles = particles
        self.particles_tilt = particles_tilt
        self.norm = norm
        self.N = N
        self.D = particles.shape[1]
        self.keepreal = keepreal
        if keepreal:
            self.particles_real = particles_real
            self.particles_tilt_real = particles_tilt_real

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], self.particles_tilt[index], index

    def get(self, index):
        return self.particles[index], self.particles_tilt[index]

# TODO: LazyTilt
