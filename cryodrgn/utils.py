from datetime import datetime as dt
import os, sys
import numpy as np
import pickle
import collections
import functools
from torch import nn
import torch

_verbose = False
def standardize_image(y):
    y_std = torch.std(y, (-1,-2), keepdim=True)
    y_mean = torch.mean(y, (-1,-2), keepdim=True)
    y = (y - y_mean)/y_std
    return y

def plot_image(axes, y_image, i):
    y_image_std = np.std(y_image)
    y_image_mean = np.mean(y_image)
    axes[i].imshow((y_image - y_image_mean)/y_image_std, cmap='gray')

def log(msg):
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    sys.stdout.flush()

def vlog(msg):
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()

def flog(msg, outfile):
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile,'a') as f:
            f.write(msg+'\n')
    except Exception as e:
        log(e)

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack((
            1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
            ), dim=1).view(-1, 3, 3)

def load_pkl(pkl):
    with open(pkl,'rb') as f:
        x = pickle.load(f)
    return x

def save_pkl(data, out_pkl, mode='wb'):
    if mode == 'wb' and os.path.exists(out_pkl):
        vlog(f'Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)

def R_from_eman(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion1(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(y), np.sin(y)
    cc = cb*ca
    cs = cb*sa
    sc = sb*ca
    ss = sb*sa
    #Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    #Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    #Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    #R = np.dot(np.dot(Ry,Rb),Ra)
    R = np.zeros((3, 3))
    R[0,0] = cg*cc - sg*sa
    R[0,1] = -sg*cc - cg*sa
    R[0,2] = sc
    R[1,0] = cg*cs + sg*ca
    R[1,1] = -sg*cs + cg*ca
    R[1,2] = ss
    R[2,0] = -cg*sb
    R[2,1] = sg*sb
    R[2,2] = cb
    return R

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1,3)
    euler[:,0] += 90
    euler[:,2] -= 90
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix()*f
    return rot

def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3,3):
        rot = rot.reshape(1,3,3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    euler = RR.from_matrix(rot*f).as_euler('zxz', degrees=True)
    euler[:,0] -= 90
    euler[:,2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi/180
    return euler

def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[1.,0.,0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt

@memoized
def _zero_sphere_helper(D):
    xx = np.linspace(-1, 1, D, endpoint=True if D % 2 == 1 else False)
    z,y,x = np.meshgrid(xx,xx,xx)
    coords = np.stack((x,y,z),-1)
    r = np.sum(coords**2,axis=-1)**.5
    return np.where(r>1)

def zero_sphere(vol):
    '''Zero values of @vol outside the sphere'''
    assert len(set(vol.shape)) == 1, 'volume must be a cube'
    D = vol.shape[0]
    tmp = _zero_sphere_helper(D)
    vlog('Zeroing {} pixels'.format(len(tmp[0])))
    vol[tmp] = 0
    return vol

