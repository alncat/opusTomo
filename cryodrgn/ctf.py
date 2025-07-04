import numpy as np
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import utils
from . import lie_tools
log = utils.log

class FourierFeatures:
    def __init__(self, stop, dim=256):
        self.start = 0
        self.stop = stop
        self.step = 1
        x_idx = torch.arange(self.start, self.stop, self.step) - self.stop //2
        freqs = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        self.freqs = torch.stack(freqs, dim=-1)*np.pi/stop
        #self.spacing = spacing
        #w = 2.**(torch.arange(self.spacing))*np.pi
        #bfactor = (-(torch.arange(self.spacing)/self.stop)**2 * np.pi ** 2 * 0.5).exp()
        #print("bfactor: ", bfactor)
        #bfactor = bfactor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fourier_dim = dim//6
        temperature = 10000
        omega = torch.arange(fourier_dim,) / (fourier_dim - 1)
        omega = 1. / (temperature ** omega)
        D, H, W, C = self.freqs.shape
        self.freqs = (self.freqs.unsqueeze(-1)*omega.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0))#.view(-1, D, H, W)
        self.freqs = torch.cat([self.freqs.sin(), self.freqs.cos()], dim=-1).view(D, H, W, -1)
        self.freqs = F.pad(self.freqs, (0, dim - fourier_dim*6))
        log("fourierfeatures: {}".format(self.freqs.shape))

    def get_embedding(self, inputs):

        # Create Base 2 Fourier features
        #w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
        #w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
        # Compute features
        #h = jnp.repeat(inputs, len(freqs), axis=-1)
        #h = self.freqs.to(inputs.get_device())*(inputs.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2))
        #h = h.sum(dim=-1).unsqueeze(0)
        #h = torch.cat([h.sin(), h.cos()], dim=1)
        h = self.freqs.to(inputs.get_device())
        return h

def compute_positional_embedding(inputs, spacing=8):
    #w = 2.**(torch.arange(spacing, device=inputs.get_device()))*np.pi
    w = torch.arange(spacing, device=inputs.get_device())*np.pi
    B, C = inputs.shape
    embeddings = (inputs.unsqueeze(-1)*w).view(B, -1)
    embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
    return embeddings

def compute_ctf_embedding(dim, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    #assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    half_dim = dim//2
    lam = 12.2643247 / (volt + 0.978466e-6 * volt**2)**.5
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=dfu.get_device()) * -embeddings)
    if bfactor is None:
        embeddings = torch.cat(((embeddings*(dfu*torch.cos(dfang)*lam)).cos(), (embeddings*(dfu*torch.sin(dfang)*lam)).cos(),
                            (embeddings*(dfv*torch.cos(dfang)*lam)).cos(), (embeddings*(dfv*torch.sin(dfang)*lam)).cos()), dim=-1)
    else:
        embeddings = torch.cat(((embeddings*(dfu*torch.cos(dfang)*lam)).cos(), (embeddings*(dfu*torch.sin(dfang)*lam)).cos(),
                            (embeddings*(dfv*torch.cos(dfang)*lam)).cos(), (embeddings*(dfv*torch.sin(dfang)*lam)).cos(),
                            (embeddings*bfactor*100).cos()), dim=-1)
    return embeddings


def compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, scale=None, mtf=None, Apix=1., rweight=True):
    '''
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    '''
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180
    delta_w = torch.atan2(w, (1-w**2)**.5) #atan(w/sqrt(1-w**2))

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2643247 / (volt + 0.978466e-6 * volt**2)**.5 #wavelength of eletron with relativity correction, 300kv, 0.0197
    x = freqs[...,0]
    y = freqs[...,1]
    ang = torch.atan2(y,x)
    s2 = x**2 + y**2
    #print(s2.shape, dfu.shape)
    df = .5*(dfu + dfv + (dfu-dfv)*torch.cos(2.*(ang-dfang)))
    gamma = 2*np.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    #ctf = (1-w**2)**.5*torch.sin(gamma) - w*torch.cos(gamma) #equivalent to sin(gamma - atan(w, sqrt(1-w**2)))
    ctf = torch.sin(gamma - delta_w)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor/4*s2 * 4*np.pi**2)
        if rweight:
            # the cosine blurring filter in aretomo, but make it less dramatic since we have bfactor
            ctf *= 0.9 + 0.1*torch.cos(s2.sqrt()*2.*np.pi)
    if scale is not None:
        #print(ctf.shape, scale.shape)
        ctf *= scale
    #applying a sigmoid function
    if mtf is not None:
        mtf_curve = (1. - torch.sigmoid(s2.sqrt()*mtf))*2.
        ctf *= mtf_curve
    return -ctf

def compute_3dctfaniso(y, centered_freqs, freqs, tilts, dfu, dfv, dfang, volt, cs, w, bfactor=None, scale=None,
                       phase_shift=0, Apix=1., plot=True, phaseflipped=False):
    # compute a ctf in 3d volume
    # compute a stack of ctfs
    #print(freqs.shape, tilts.shape, dfu.shape, bfactor.shape)
    # consider only isotropic ctf so far
    # centered freqs are in the range [-Y//2+1, Y//2]
    centered_freqs = centered_freqs.to(y.get_device())
    freqs = freqs.to(y.get_device())
    dfu = dfu.unsqueeze(-1)
    volt = volt.unsqueeze(-1)
    cs = cs.unsqueeze(-1)
    w = w.unsqueeze(-1)
    bfactor = -bfactor.unsqueeze(-1)
    scale = scale.unsqueeze(-1)
    #print(volt, cs, w, Apix)
    dfv = dfv.unsqueeze(-1)
    dfang = dfang.unsqueeze(-1)
    #if rot_angs is not None:
    #    #dfang: (B, T, 1, 1), rot_angs: (B, 1)
    #    dfang += rot_angs.unsqueeze(-2).unsqueeze(-2)
    ctfs = compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor/(4*np.pi**2), scale, Apix=Apix, rweight=False)
    #print(bfactor[0].squeeze()/(4*np.pi**2))

    # the transpose of tilt rotation
    # xtilt = R xori
    #NOTE: use this if the tilt angle in ctf stars are flipped! (by my script), namely, follow relion's convention
    Rtilts = lie_tools.rot_2d(-tilts.squeeze(-1).float()).unsqueeze(-3)
    # test other tilt direction
    #NOTE: use this if the tilt angle is the same as given
    #Rtilts = lie_tools.rot_2d(tilts.squeeze(-1).float()).unsqueeze(-3)
    grid_rotated = centered_freqs @ Rtilts #(D, W, 2) x (N, T, 1, 2, 2)
    # grid_rotated now is (N, T, D, W, 2)
    #print(Rtilts.shape, tilts[0,], grid_rotated.shape)
    # generate tilts
    # shift ctfs to image center
    N = ctfs.shape[0]
    T = ctfs.shape[1]
    Y = ctfs.shape[-2]
    X = ctfs.shape[-1]
    #print(N, T, Y, X)
    ctfs = torch.fft.fftshift(ctfs, dim=(-2)) # fftshift move the input by Y//2
    ctfs = ctfs.unsqueeze(-2) #N, T, H, 1, W,


    # now padding zeros to the z dim, make it 3d
    pad_size = Y//2
    # tensor is ordered as, after padding the zero frequency is centered at Y//2
    ctfs = F.pad(ctfs, (0, 0, pad_size, pad_size-1)) #N, T, H, D, W
    ctfs /= scale.sum(dim=1, keepdim=True).unsqueeze(-1) #normalize ctf by number of tilts
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        #print(ctfs.shape,)
        cmap = 'gray'
        axes[0].imshow(ctfs[0, -1, 0].cpu().numpy(), cmap=cmap)

    # rotate the grid

    #centered_freqs #(-z//2, z//2-1), map it to -1, 1 by (+ z//2 )/(Y-1)
    # also need to map (0, z//2) to -1, 1 by
    centered_freqs_Z = grid_rotated[..., 1]
    centered_freqs_Z = (centered_freqs_Z*Y + Y//2)/(Y - 1)
    centered_freqs_Z = centered_freqs_Z*2. - 1.
    centered_freqs_X = grid_rotated[..., 0]
    centered_freqs_X = (centered_freqs_X*Y)/(X - 1)
    centered_freqs_X = centered_freqs_X*2. - 1.
    grid_rotated[..., 0] = centered_freqs_X
    grid_rotated[..., 1] = centered_freqs_Z

    #reshape ctfs to N*T, H, D, W
    ctfs = ctfs.view(N*T, Y, Y, X)
    grid_rotated = grid_rotated.view(N*T, Y, X, 2)
    #sampling ctfs
    ctfs_3d = F.grid_sample(ctfs, grid_rotated, mode='bicubic', align_corners=True)

    #print the some ctfs
    #summing along the tilt channel to obtain 3d ctf
    ctfs_3d = ctfs_3d.view(N, T, Y, Y, X)#.sum(dim=1) #(N, Y, Z, X)
    if plot:
        axes[1].imshow(ctfs_3d[0, -1, 0].cpu().numpy(), cmap=cmap)
        plt.show()
    ctfs_3d = ctfs_3d.sum(dim=1)
    #rearrange ctf from x, z, y to x, y, z order, n y z x -> n z y x
    ctfs_3d = torch.permute(ctfs_3d, [0, 2, 1, 3])
    #print(ctfs_3d[0, Y//2-1, Y//2-1, 0])
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(ctfs_3d[0,:,Y//2, :].cpu().numpy(), cmap=cmap)
        print(y.shape, y[0,:,Y//2-1,Y//2-1:].shape)
        #axes[1].imshow(y[0,:,Y//2,Y//2:].cpu().numpy(), cmap=cmap)
        #plt.show()

    #lastly, shift the ctf back to fftw style, by -Y//2
    #ctfs_3d = torch.fft.ifftshift(ctfs_3d, dim=(-3, -2,))
    #diff = (ctfs_3d[0, :, Y//2-1, :] - y[0, :, Y//2, Y//2-1:])
    if plot:
        #diff = (ctfs_3d[0, :-1, Y//2-1, :-1] - y[0, 1:, Y//2, Y//2:])
        #plt.imshow(diff.cpu().numpy(), cmap=cmap)
        #print(diff.abs().mean())
        axes[1].imshow(ctfs_3d[0, :, 0, :].cpu().numpy(), cmap=cmap)
        plt.show()
    # if the input subtomograms are phaseflipped, then ctf should have no phase information
    return ctfs_3d

def compute_3dctf(y, centered_freqs, freqs, tilts, dfu, volt, cs, w, bfactor=None, scale=None, phase_shift=0, Apix=1., plot=True, use_warp=True):
    # compute a ctf in 3d volume
    # compute a stack of ctfs
    #print(freqs.shape, tilts.shape, dfu.shape, bfactor.shape)
    # consider only isotropic ctf so far
    # centered freqs are in the range [-Y//2+1, Y//2]
    centered_freqs = centered_freqs.to(y.get_device())
    freqs = freqs.to(y.get_device())
    dfu = dfu.unsqueeze(-1)
    volt = volt.unsqueeze(-1)
    cs = cs.unsqueeze(-1)
    w = w.unsqueeze(-1)
    bfactor = bfactor.unsqueeze(-1)
    scale = scale.unsqueeze(-1)
    #print(volt, cs, w, Apix)
    dfv = dfu
    dfang = torch.zeros_like(dfu)
    ctfs = compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift,
                       bfactor/(4*np.pi**2), scale, Apix=Apix, rweight=False)

    #print(bfactor[0].squeeze()/(4*np.pi**2))

    # the transpose of tilt rotation
    # xtilt = R xori
    #NOTE: use this if the tilt angle in ctf stars are flipped! (by my script)
    Rtilts = lie_tools.rot_2d(-tilts.squeeze(-1).float()).unsqueeze(-3)
    # test other tilt direction
    #NOTE: use this if the tilt angle is the same as given
    #Rtilts = lie_tools.rot_2d(tilts.squeeze(-1).float()).unsqueeze(-3)
    grid_rotated = centered_freqs @ Rtilts #(D, W, 2) x (N, T, 1, 2, 2)
    # grid_rotated now is (N, T, D, W, 2)
    #print(Rtilts.shape, tilts[0,], grid_rotated.shape)
    # generate tilts
    # shift ctfs to image center
    N = ctfs.shape[0]
    T = ctfs.shape[1]
    Y = ctfs.shape[-2]
    X = ctfs.shape[-1]
    #print(N, T, Y, X)
    ctfs = torch.fft.fftshift(ctfs, dim=(-2)) # fftshift move the input by Y//2
    ctfs = ctfs.unsqueeze(-2) #N, T, H, 1, W,


    # now padding zeros to the z dim, make it 3d
    pad_size = Y//2
    # tensor is ordered as, after padding the zero frequency is centered at Y//2
    ctfs = F.pad(ctfs, (0, 0, pad_size, pad_size-1)) #N, T, H, D, W
    if use_warp:
        ctfsweight = torch.ones_like(ctfs)*scale.unsqueeze(-1)
    else:
        ctfs /= scale.sum(dim=1, keepdim=True).unsqueeze(-1) #normalize ctf by number of tilts
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        #print(ctfs.shape,)
        cmap = 'gray'
        axes[0].imshow(ctfs[0, -1, 0].cpu().numpy(), cmap=cmap)

    # rotate the grid

    #centered_freqs #(-z//2, z//2-1), map it to -1, 1 by (+ z//2 )/(Y-1)
    # also need to map (0, z//2) to -1, 1 by
    centered_freqs_Z = grid_rotated[..., 1]
    centered_freqs_Z = (centered_freqs_Z*Y + Y//2)/(Y - 1)
    centered_freqs_Z = centered_freqs_Z*2. - 1.
    centered_freqs_X = grid_rotated[..., 0]
    centered_freqs_X = (centered_freqs_X*Y)/(X - 1)
    centered_freqs_X = centered_freqs_X*2. - 1.
    grid_rotated[..., 0] = centered_freqs_X
    grid_rotated[..., 1] = centered_freqs_Z

    #reshape ctfs to N*T, H, D, W
    ctfs = ctfs.view(N*T, Y, Y, X)
    grid_rotated = grid_rotated.view(N*T, Y, X, 2)
    #sampling ctfs
    ctfs_3d = F.grid_sample(ctfs, grid_rotated, mode='bicubic', align_corners=True)
    if use_warp:
        ctfsweight = ctfsweight.view(N*T, Y, Y, X)
        ctfsweight = F.grid_sample(ctfsweight, grid_rotated, mode='bilinear', align_corners=True)

    #print the some ctfs
    #summing along the tilt channel to obtain 3d ctf
    ctfs_3d = ctfs_3d.view(N, T, Y, Y, X)#.sum(dim=1) #(N, Y, Z, X)
    if use_warp:
        ctfsweight = ctfsweight.view(N, T, Y, Y, X)
    if plot:
        axes[1].imshow(ctfs_3d[0, -1, 0].cpu().numpy(), cmap=cmap)
        plt.show()
    ctfs_3d = ctfs_3d.sum(dim=1) #(N, Y, Z, X)
    if use_warp:
        ctfsweight = ctfsweight.sum(dim=1) + 1e-3
        #print(ctfsweight.min(), ctfsweight.max())
        ctfs_3d /= (ctfsweight)
    #rearrange ctf from x, z, y to x, y, z order
    ctfs_3d = torch.permute(ctfs_3d, [0, 2, 1, 3])
    #print(ctfs_3d[0, Y//2-1, Y//2-1, 0])
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(ctfs_3d[0,:,Y//2, :].cpu().numpy(), cmap=cmap)
        print(y.shape, y[0,:,Y//2-1,Y//2-1:].shape)
        #axes[1].imshow(y[0,:,Y//2,Y//2:].cpu().numpy(), cmap=cmap)
        #plt.show()

    #lastly, shift the ctf back to fftw style, by -Y//2
    #ctfs_3d = torch.fft.ifftshift(ctfs_3d, dim=(-3, -2,))
    #diff = (ctfs_3d[0, :, Y//2-1, :] - y[0, :, Y//2, Y//2-1:])
    if plot:
        #diff = (ctfs_3d[0, :-1, Y//2-1, :-1] - y[0, 1:, Y//2, Y//2:])
        #plt.imshow(diff.cpu().numpy(), cmap=cmap)
        #print(diff.abs().mean())
        axes[1].imshow(ctfs_3d[0, :, 0, :].cpu().numpy(), cmap=cmap)
        plt.show()

    return ctfs_3d

def compute_ctf_np(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    '''
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 array of 2D spatial frequencies
        dfu (float): DefocusU (Angstrom)
        dfv (float): DefocusV (Angstrom)
        dfang (float): DefocusAngle (degrees)
        volt (float): accelerating voltage (kV)
        cs (float): spherical aberration (mm)
        w (float): amplitude contrast ratio
        phase_shift (float): degrees
        bfactor (float): envelope fcn B-factor (Angstrom^2)
    '''
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[:,0]
    y = freqs[:,1]
    ang = np.arctan2(y,x)
    s2 = x**2 + y**2
    df = .5*(dfu + dfv + (dfu-dfv)*np.cos(2*(ang-dfang)))
    gamma = 2*np.pi*(-.5*df*lam*s2 + .25*cs*lam**3*s2**2) - phase_shift
    ctf = np.sqrt(1-w**2)*np.sin(gamma) - w*np.cos(gamma)
    if bfactor is not None:
        ctf *= np.exp(-bfactor/4*s2)
    return np.require(ctf,dtype=freqs.dtype)

def print_ctf_params(params):
    assert len(params) == 9
    log('Image size (pix)  : {}'.format(int(params[0])))
    log('A/pix             : {}'.format(params[1]))
    log('DefocusU (A)      : {}'.format(params[2]))
    log('DefocusV (A)      : {}'.format(params[3]))
    log('Dfang (deg)       : {}'.format(params[4]))
    log('voltage (kV)      : {}'.format(params[5]))
    log('cs (mm)           : {}'.format(params[6]))
    log('w                 : {}'.format(params[7]))
    log('Phase shift (deg) : {}'.format(params[8]))

def plot_ctf(D,Apix,ctf_params):
    assert len(ctf_params) == 7
    import matplotlib.pyplot as plt
    import seaborn as sns
    freqs = np.stack(np.meshgrid(np.linspace(-.5,.5,D,endpoint=False),np.linspace(-.5,.5,D,endpoint=False)),-1)/Apix
    freqs = freqs.reshape(-1,2)
    c = compute_ctf_np(freqs, *ctf_params)
    sns.heatmap(c.reshape(D, D))

def load_ctf_for_training(D, ctf_params_pkl):
    assert D%2 == 0
    ctf_params = utils.load_pkl(ctf_params_pkl)
    assert ctf_params.shape[1] == 9
    # Replace original image size with current dimensions
    Apix = ctf_params[0,0]*ctf_params[0,1]/D
    ctf_params[:,0] = D
    ctf_params[:,1] = Apix
    print_ctf_params(ctf_params[0])
    # Slice out the first column (D)
    return ctf_params[:,1:]

def load_group_for_training(group_pkl):
    group = utils.load_pkl(group_pkl)
    return group

