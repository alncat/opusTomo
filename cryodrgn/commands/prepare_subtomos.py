'''
Train a VAE for heterogeneous reconstruction with known pose
'''
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse
import pickle
from datetime import datetime as dt
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

import cryodrgn
from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import fft
from cryodrgn import lie_tools
from cryodrgn import dataset
from cryodrgn import ctf

from cryodrgn.pose import PoseTracker
from cryodrgn.models import HetOnlyVAE
from cryodrgn.lattice import Lattice, Grid, CTFGrid
from cryodrgn.group_stat import GroupStat
from cryodrgn.beta_schedule import get_beta_schedule, LinearSchedule
from cryodrgn.pose_encoder import PoseEncoder

log = utils.log
vlog = utils.vlog

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('-r', '--ref_vol', type=os.path.abspath, help='Input volume (.mrcs)')
    parser.add_argument('--zdim', type=int, required=False, help='Dimension of latent variable')
    parser.add_argument('--poses', type=os.path.abspath, required=False, help='Image poses (.pkl)')
    parser.add_argument('--masks', type=os.path.abspath, required=False, help='Masks related parameters (.pkl)')
    parser.add_argument('--symm', type=os.path.abspath, required=False, help='Symmetric Operators (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    parser.add_argument('--angpix', type=float, help='angstrom per pixel')
    parser.add_argument('--group', metavar='pkl', type=os.path.abspath, help='group assignments (.pkl)')
    parser.add_argument('--group-stat', metavar='pkl', type=os.path.abspath, help='group statistics (.pkl)')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--latents', type=os.path.abspath, help='Image latent encodings (.pkl)')
    parser.add_argument('--split', metavar='pkl', help='Initialize training from a split checkpoint')
    parser.add_argument('--valfrac', type=float, default=0.2, help='the fraction of images held for validation')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval in N_IMGS (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0,100000), help='Random seed')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85,  help='Windowing radius (default: %(default)s)')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--relion31', action='store_true', help='Flag if relion3.1 star format')
    group.add_argument('--lazy-single', default=True, action='store_true', help='Lazy loading if full dataset is too large to fit in memory')
    group.add_argument('--preprocessed', action='store_true', help='Skip preprocessing steps if input data is from cryodrgn preprocess_mrcs')
    group.add_argument('--max-threads', type=int, default=16, help='Maximum number of CPU cores for FFT parallelization (default: %(default)s)')

    group = parser.add_argument_group('Tilt series')
    group.add_argument('--tilt', help='Particles (.mrcs)')
    group.add_argument('--tilt-deg', type=float, default=45, help='X-axis tilt offset in degrees (default: %(default)s)')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch-size', type=int, default=20, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=5e-5, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--lamb', type=float, default=0.5, help='restraint strength for umap prior (default: %(default)s)')
    group.add_argument('--downfrac', type=float, default=0.5, help='downsample to (default: %(default)s) of original size')
    group.add_argument('--templateres', type=int, default=192, help='define the output size of 3d volume (default: %(default)s)')
    group.add_argument('--bfactor', type=float, default=2., help='apply bfactor (default: %(default)s) to reconstruction')
    group.add_argument('--beta', default='cos', help='Choice of beta schedule')
    group.add_argument('--beta-control', default=0.5, type=float, help='restraint strength for KL target. (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--tmp-prefix', type=str, default='tmp', help='prefix for naming intermediate reconstructions')
    group.add_argument('--amp', action='store_true', help='Use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')
    group.add_argument('--num-gpus', type=int, default=4, help='number of gpus used for training')
    parser.add_argument('--write-ctf', default=False, action='store_true', help='save CTF as mrc')
    group = parser.add_argument_group('Pose SGD')
    group.add_argument('--do-pose-sgd', action='store_true', help='Refine poses with gradient descent')
    group.add_argument('--pretrain', type=int, default=1, help='Number of epochs with fixed poses before pose SGD (default: %(default)s)')
    group.add_argument('--emb-type', choices=('s2s2','quat'), default='quat', help='SO(3) embedding type for pose SGD (default: %(default)s)')
    group.add_argument('--pose-lr', type=float, default=3e-4, help='Learning rate for pose optimizer (default: %(default)s)')
    group.add_argument('--pose-enc', action='store_true', help='predict pose parameter using encoder')
    group.add_argument('--pose-only', action='store_true', help='train pose encoder only')
    group.add_argument('--plot', action='store_true', help='plot intermediate result')
    group.add_argument('--estpose', default=False, action='store_true', help='estimate pose')
    group.add_argument('--warp', default=False, action='store_true', help='using subtomograms from warp')
    group.add_argument('--tilt-step', type=int, default=2, help='the interval between successive tilts (default: %(default)s)')
    group.add_argument('--tilt-range', type=int, default=50, help='the range of tilt angles (default: %(default)s)')

    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers', dest='qlayers', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--enc-dim', dest='qdim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--encode-mode', default='grad', choices=('conv','resid','mlp','tilt',
                                                                  'fixed', 'affine', 'fixed_blur', 'deform', 'grad',
                                                                  'vq'),
                                            help='Type of encoder network (default: %(default)s)')
    group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    group.add_argument('--use-real', action='store_true', help='Use real space image for encoder (for convolutional encoder)')
    group.add_argument('--optimize-b', action='store_true', help='optimize b factor')

    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf','none', 'vanilla'), default='vanilla', help='Type of positional encoding (default: %(default)s)')
    group.add_argument('--template-type', choices=('conv'), default='conv', help='Type of template decoding method (default: %(default)s)')
    group.add_argument('--warp-type', choices=('blurmix', 'diffeo', 'deform'), help='Type of warp decoding method (default: %(default)s)')
    #group.add_argument('--symm', help='Type of symmetry of the 3D volume (default: %(default)s)')
    group.add_argument('--num-struct', type=int, default=1, help='Num of structures (default: %(default)s)')
    group.add_argument('--deform-size', type=int, default=2, help='Num of structures (default: %(default)s)')
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding (default: image D)')
    group.add_argument('--domain', choices=('hartley','fourier'), default='fourier', help='Decoder representation domain (default: %(default)s)')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    return parser


def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'
    def flog(msg): # HACK: switch to logging module
        return utils.flog(msg, LOG)
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    flog('Use cuda {}'.format(use_cuda))
    if use_cuda:
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
        pass
    else:
        log('WARNING: No GPUs detected')

    # set beta schedule
    assert args.beta_control, "Need to set beta control weight for schedule {}".format(args.beta)
    beta_schedule = get_beta_schedule(args.beta)

    # load index filter
    if args.ind is not None:
        flog('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind,'rb'))
    else: ind = None

    # load dataset
    if args.ref_vol is not None:
        flog(f'Loading reference volume from {args.ref_vol}')
        ref_vol = dataset.VolData(args.ref_vol).get()

        #flog(f'Loading fixed mask from {args.ref_vol}')
        #mask_vol = dataset.VolData(args.ref_vol).get()

    else:
        ref_vol = None
    flog(f'Loading dataset from {args.particles}')
    if args.tilt is None:
        tilt = None
        args.use_real = args.encode_mode == 'conv'
        args.real_data = args.pe_type == 'vanilla'

        if args.lazy_single and not args.warp:
            data = dataset.LazyTomoDRGNMRCData(args.particles, norm=args.norm,
                                       real_data=args.real_data, invert_data=args.invert_data,
                                       ind=ind, keepreal=args.use_real, window=False,
                                       datadir=args.datadir, relion31=args.relion31, window_r=args.window_r, downfrac=args.downfrac,
                                       tilt_step=args.tilt_step, tilt_range=args.tilt_range)
        else:
            raise NotImplementedError("Use --lazy-single for on-the-fly image loading")

    Nimg = data.N
    D = data.D #data dimension
    log('Subtomograms Converted!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
