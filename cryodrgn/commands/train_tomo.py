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
torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.7+

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
    parser.add_argument('--zaffinedim', type=int, default=4, required=False, help='Dimension of latent variable')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--masks', type=os.path.abspath, required=False, help='Masks related parameters (.pkl)')
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
    group.add_argument('--accum-step', type=int, default=2, help='gradient accumulation step for optimizer (default: %(default)s)')
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
    group.add_argument('--ctfalpha', type=float, default=0, help='the degree of ctf correction to experimental subtomogram (default: %(default)s)')
    group.add_argument('--ctfbeta', type=float, default=1, help='the degree of ctf correction to reconstruction of decoder (default: %(default)s)')

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

def train_batch(model, lattice, y, yt, rot, trans, optim, beta,
                beta_control=None, tilt=None, ind=None, grid=None, ctf_grid=None,
                ctf_params=None, yr=None, use_amp=False, save_image=False, vanilla=True,
                group_stat=None, do_scale=False, it=None, enc=None,
                args=None, euler=None, posetracker=None, data=None, backward=True, update_params=True,
                snr2=1., body_poses=None, ctf_filename=None):

    if backward:
        model.train()
    else:
        model.eval()
    if trans is not None:
        y, yt = preprocess_input(y, yt, lattice, trans, vanilla=vanilla)
    z_mu, z_logstd, z, y_recon, y_recon_tilt, losses, y, mus, \
        euler_samples, y_recon_ori, neg_mus, mask_sum, body_poses_pred = run_batch(
                                                                 model, lattice, y, yt, rot,
                                                                 tilt=tilt, ind=ind, ctf_params=ctf_params,
                                                                 yr=yr, vanilla=vanilla, ctf_grid=ctf_grid,
                                                                 grid=grid, save_image=save_image,
                                                                 group_stat=group_stat, do_scale=do_scale,
                                                                 trans=trans, it=it, enc=enc,
                                                                 args=args, euler=euler,
                                                                 posetracker=posetracker, data=data,
                                                                 snr2=snr2, body_poses=body_poses, ctf_filename=ctf_filename)
    #if update_params:
    #    optim.zero_grad()

    loss, gen_loss, snr, mu2, std2, mmd, c_mmd, top_euler, mse = loss_function(z_mu, z_logstd, y, y_recon,
                                        beta, y_recon_tilt, beta_control, vanilla=vanilla,
                                        group_stat=group_stat, ind=ind, mask_sum=mask_sum,
                                        losses=losses, args=args, it=it, zs=z,
                                        mus=mus, neg_mus=neg_mus, y_recon_ori=y_recon_ori, euler_samples=euler_samples,
                                        snr2=snr2, body_poses=body_poses, body_poses_pred=None)

    #if top_euler is not None and not update_params:
    #    posetracker.set_euler(top_euler, ind)

    #if use_amp:
    #    with amp.scale_loss(loss, optim) as scaled_loss:
    #        scaled_loss.backward()
    if backward:
        loss.backward()
    if update_params:
        optim.step()
        optim.zero_grad()
    return z_mu, loss.item(), gen_loss.item(), snr.item(), losses['l2'].mean().item(), losses['tvl2'].mean().item(), \
            mu2.item()/args.zdim, std2.item()/args.zdim, mmd.item(), c_mmd.item(), mse.item(), body_poses_pred

def data_augmentation(y, trans, ctf_grid, grid, window_r, downfrac=0.5):
    with torch.no_grad():
        y_fft = fft.torch_fft2_center(y)
        # undo experimental image translation
        #trans = torch.clamp(trans, min=-10, max=10)
        #rand_t = torch.randn_like(trans)*2.5
        # correct translation
        y_fft = ctf_grid.translate_ft(y_fft, -trans)#+rand_t)#+trans.round())
        #y_fft = ctf_grid.translate_ft(y_fft, -trans.round())

        #window y
        y = fft.torch_ifft2_center(y_fft)
        mask_real = grid.get_circular_mask(window_r)
        y *= mask_real
        y_fft = fft.torch_fft2_center(y)

        # apply b factor
        #random_b = np.random.rand() - 0.5
        random_b = (np.random.rand() - 0.5)*0.5*torch.ones(y_fft.shape[0])
        #random_b = 0.5*(torch.rand(y_fft.shape[0]) - 0.5)
        b_fact = ctf_grid.get_b_factor_bc(b=random_b).unsqueeze(1)
        #random_b = 0.5*(torch.rand(y_fft.shape[0]) - 0.5)
        #random_d = torch.rand(y_fft.shape[0])
        #d_fact = ctf_grid.get_ddefocus_bc(b=random_d).unsqueeze(1)
        y_fft_ori = y_fft*b_fact

        #random_b = np.random.rand() - 0.5
        #random_b = 0.25*(torch.rand(y_fft.shape[0]) - 0.5)
        #b_fact = ctf_grid.get_b_factor_bc(b=random_b).unsqueeze(1)
        #y_fft *= b_fact

        #print(y.shape, y_fft.shape)
        # window y
        #y = fft.torch_ifft2_center(y_fft)
        #mask_real = grid.get_circular_mask(window_r)
        #y *= mask_real

        # downsample in frequency space by applying cos filter
        #y_fft = utils.mask_image_fft(y, mask, ctf_grid)

        down_size = (int(y.shape[-1]*downfrac)//2)*2
        down_scale = down_size/y.shape[-1]
        y_fft_s = torch.fft.fftshift(y_fft, dim=(-2))
        y_fft_crop = utils.crop_fft(y_fft_s, down_size)*(down_scale)**2
        y_fft = torch.fft.ifftshift(y_fft_crop, dim=(-2))
        y = fft.torch_ifft2_center(y_fft)

        y_fft_s = torch.fft.fftshift(y_fft_ori, dim=(-2))
        y_fft_crop = utils.crop_fft(y_fft_s, down_size)*(down_scale)**2
        y_fft_ori = torch.fft.ifftshift(y_fft_crop, dim=(-2))

        #y_rand = ctf_grid.sample_local_translation(y_fft, 1, 1.)
        #y = fft.torch_ifft2_center(y_rand)
        return y, y_fft_ori, random_b

def preprocess_input(y, yt, lattice, trans, vanilla=True):
    # center the image
    B = y.size(0)
    D = lattice.D
    if vanilla:
        #add a channel dimension
        return y.unsqueeze(1), yt
    y = lattice.translate_ht(y.view(B,-1), trans.unsqueeze(1)).view(B,D,D)
    if yt is not None: yt = lattice.translate_ht(yt.view(B,-1), trans.unsqueeze(1)).view(B,D,D)
    return y, yt

def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def sample_neighbors(posetracker, data, euler, rot, ind, ctf_params, ctf_grid, grid, args, W, out_size):
    device = euler.get_device()
    B = euler.shape[0]
    mus, idices, top_mus, neg_mus = posetracker.sample_full_neighbors(euler.cpu(), ind.cpu(), num_pose=8)
    other_euler = None #posetracker.get_euler(idices).to(device).view(B, -1, 3)
    other_rot, other_tran = None, None #posetracker.get_pose(idices)
    other_y, other_y_fft = None, None
    other_c = None
    # put all together
    others = {"y": other_y, "rots": other_rot, "y_fft": other_y_fft, "ctf": other_c, "euler": other_euler}
    #print(other_euler.shape, other_tran.shape, other_y.shape, idices.shape)
    return mus, others, top_mus, neg_mus

def run_batch(model, lattice, y, yt, rot, tilt=None, ind=None, ctf_params=None,
              yr=None, vanilla=True, ctf_grid=None, grid=None, save_image=False,
              group_stat=None, do_scale=True, trans=None, it=None, enc=None,
              args=None, euler=None, posetracker=None, data=None, snr2=1., body_poses=None, ctf_filename=None):
    use_tilt = yt is not None
    use_ctf = ctf_params is not None
    B = y.size(0)
    D = lattice.D
    W = y.size(-1)
    out_size = D - 1
    # get real mask
    mask_real = None

    if use_ctf:
        if vanilla:
            # ctf has already been computed
            c = ctf_params
            #freqs = ctf_grid.freqs2d.view(-1, 2).unsqueeze(0)/ctf_params[0,0].view(1,1,1) #(1, (-x+1, x)*x, 2)
            #c = ctf.compute_ctf(freqs, *torch.split(ctf_param[:,1:], 1, 1), bfactor=args.bfactor).view(B,D-1,-1) #(B, )

    # encode
    if vanilla:
        z_mu, z_logvar, z = 0., 0., 0.

    # add bfactors to ctf_params, the second from last column stores bfactor, the last column stores scale
    #random_b = (np.random.normal())/3.
    #random_b = np.random.gamma(1., 0.6)
    random_b = torch.randn_like(c[..., 0, -2])/3.
    #c[...,-2] = c[...,-2] + (args.bfactor+random_b.unsqueeze(-1))*(4*np.pi**2)

    plot = args.plot and it % (args.log_interval) == B
    if plot:
        f, axes = plt.subplots(2, 3)
    # decode
    if not vanilla:
        mask = lattice.get_circular_mask(D//2) # restrict to circular mask
        y_recon = model(lattice.coords[mask]/lattice.extent/2 @ rot, z).view(B,-1)
    else:
        #w_filt    = ctf_grid.shell_to_grid(group_stat.get_wiener_filter(ind))
        if args.encode_mode in ['grad']:
            d_i = 0
            with torch.no_grad():
                diff = y
                #assert diff.shape[-1] == model.encoder_image_size, "y shape {y.shape[-1]} should equal with {model.encoder_image_size}"

            if plot:
                print(f"ctf {c.shape}, y {y.shape}")
                #print(c[...,-1])
                #utils.plot_image(axes, exp_fac.detach().cpu().numpy(), 0)
                #utils.plot_image(axes, i_c[d_i,d_i,...].detach().cpu().numpy(), d_i, 0, log=True)
                #utils.plot_image(axes, diff[d_i,d_i,...].detach().cpu().numpy(), d_i, 2, log=True)
                #utils.plot_image(axes, y[d_i,d_i,...].detach().cpu().numpy(), d_i, 1, log=True)
                #correlations = F.cosine_similarity(diff[:,d_i,...].view(B,-1), y.view(B,-1))
                #print(correlations)
        else:
            diff = None

        # encode images to latents, appending b_factors
        z, encout = model.vanilla_encode(diff, rot, trans, eulers=euler, num_gpus=args.num_gpus, snr2=snr2,
                                         body_poses=body_poses,)
                                         #ctf_param=torch.cat((ctf_param[:,1:], random_b.unsqueeze(-1).to(ctf_param.get_device())), dim=-1))
        # set y to centered one
        if args.encode_mode in ["grad"]:
            y = encout['rotated_x']
        #print(z - encout['z_mu'])
        # sample nearest neighbors
        #posetracker.set_emb(encout["z_mu"][:, :args.zdim], ind)
        posetracker.set_emb(encout["z_mu"], ind)
        mus, others, top_mus, neg_mus = sample_neighbors(posetracker, data, euler, rot,
                                                ind, ctf_params, ctf_grid, grid, args, W, out_size)
        mus = mus.to(z.get_device())
        neg_mus = neg_mus.to(z.get_device())
        others = None
        #neg_idices = None

        # decode latents
        decout = model.vanilla_decode(rot, trans, z=z, save_mrc=save_image, eulers=euler,
                                      ref_fft=y, ctf_param=c, encout=encout, mask=mask_real, body_poses=body_poses,
                                      ctf_grid=ctf_grid, estpose=args.estpose, ctf_filename=ctf_filename, write_ctf=args.write_ctf, bfactor=args.bfactor)

        if decout["affine"] is not None:
            posetracker.set_pose(decout["affine"][0].detach(), decout["affine"][1].detach(), ind)

        y_recon_fft = None
        y_ref_fft   = None #torch.view_as_complex(decout["y_ref_fft"])
        y_ffts      = {"y_recon_fft":y_recon_fft, "y_ref_fft":y_ref_fft}

        #print(y_recon_fft.shape, y_recon_fft.dtype, y_ref_fft.shape)
        losses = decout["losses"]
        if 'losses' in encout:
            losses.update(encout["losses"])
        # set the encoding of each particle
        if args.encode_mode == "grad":
            # keep only the compositional encoding
            z_mu = encout["z_mu"]#[:, :args.zdim]
            z_logstd = encout["z_logstd"]#[:, :args.zdim]
            #z = encout["encoding"]
        y_recon_ori = decout["y_recon_ori"]
        # retrieve mask
        #mask = decout["mask"]
        mask_sum = decout["mask_sum"]
        # retrieve nearest neighbor in the same batch
        if args.encode_mode == "grad":
            z_nn = encout["z_knn"]
            z_diff = (z_mu.unsqueeze(1) - z_nn).pow(2).sum(-1)
            #print(diff)
            losses["knn"] = torch.log(1 + z_diff).mean()
            #print(y_recon_ori[:, :1, ...].shape)

    if use_ctf:
        if vanilla:
            euler_samples = None #decout["euler_samples"]
            y_recon = decout["y_recon"]
            y_ref   = decout["y_ref"]

            d_i, d_j, d_k = 1, 1, 0
            if plot:
                #print(trans)
                #correlations = F.cosine_similarity(y_recon_ori[:,d_k,...].view(B,-1), y_ref[:,d_k,...].view(B,-1))
                #utils.plot_image(axes, y_recon_ori[0,0,...].detach().cpu().numpy(), 0, 0, log=True, log_msg="y_recon_ori")
                #utils.plot_image(axes, y_recon_ori[d_i,d_k,...].detach().cpu().numpy(), d_j, 0, log=True)
                #print(encout["rotated_x"].shape)
                #utils.plot_image(axes, encout["rotated_x"][0,...].detach().cpu().numpy(), 0, 0, log=True)
                #utils.plot_image(axes, encout["rotated_x"][d_i,...].detach().cpu().numpy(), d_j, 0, log=True, log_msg="rotated_x")
                utils.plot_image(axes, diff[0,0,...].detach().cpu().numpy(), 0, 0, log=True, log_msg="y0")
                utils.plot_image(axes, diff[d_j,0,...].detach().cpu().numpy(), d_j, 0, log=True, log_msg="y1")
                #utils.plot_image(axes, y_ref[d_i,d_k,...].detach().cpu().numpy(), d_j, 2, log=True, log_msg="y_ref")
                #utils.plot_image(axes, y[d_i,...].detach().numpy(), 1)
                #log("correlations w.o. mask: {}".format(correlations.detach().cpu().numpy()))

            if group_stat is not None:
                if do_scale:
                    group_scales = group_stat.get_group_scales(ind)
                    if it and it % 1000 == 0:
                        log("group_scales: {}".format(group_scales))
                    #y_recon_fft *= group_scales.unsqueeze(-1).unsqueeze(-1)
            if plot:
                utils.plot_image(axes, y_recon[0,0,...].detach().cpu().numpy(), 0, 1, log=True, log_msg="y_recon0")
                utils.plot_image(axes, decout["y_recon"][d_i,0,...].detach().cpu().numpy(), d_j, 1)
                #utils.plot_image(axes, y_ref[d_i,d_k,...].detach().cpu().numpy(), d_j, 2)
                utils.plot_image(axes, y_ref[0,0,...].detach().cpu().numpy(), 0, 2, log=True, log_msg="y_ref0")
                utils.plot_image(axes, y_ref[d_i,d_k,...].detach().cpu().numpy(), d_j, 2)

                print('z projection of y_recon and y_ref: ', y_recon.shape, y_ref.shape, B)
                correlations = F.cosine_similarity(y_recon[:,d_k,...].reshape(B,-1), y_ref[:,d_k,...].reshape(B,-1))
                log("correlations with mask: {}".format(correlations.detach().cpu().numpy()))
                log(f"mean correlations {correlations.mean()}")
    # decode the tilt series
    y_recon_tilt = None

    return z_mu, z_logstd, z, y_recon, y_recon_tilt, losses, y_ref, mus, euler_samples, y_recon_ori, neg_mus, mask_sum, decout["affine"]

def loss_function(z_mu, z_logstd, y, y_recon, beta,
                  y_recon_tilt=None, beta_control=None, vanilla=False,
                  group_stat=None, ind=None, mask_sum=None, losses=None,
                  args=None, it=None, y_ffts=None, zs=None, mus=None,
                  neg_mus=None, y_recon_ori=None, euler_samples=None, snr2=None,
                  body_poses=None, body_poses_pred=None):
    # reconstruction error
    B = y.size(0)
    C = losses["y_recon2"].size(1)
    W = y.size(-1)
    mask_sum = mask_sum.float()
    #print(B, W, C, mask_sum, W**3*np.pi*0.05)
    mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum)*W**3*np.pi*0.05)
    top_euler = None
    if C > 1:
        #y_recon2 = (y_recon**2).sum(dim=(-1,-2,-3)).view(B, -1)
        #l2_diff = (-2.*y_recon*y).sum(dim=(-1,-2,-3)).view(B, -1)
        ##l2_diff_std = l2_diff.std().detach()
        ##print(l2_diff_std/2., W*0.125)
        #l2_diff = l2_diff + y_recon2
        y_recon2 = losses["y_recon2"]
        l2_diff = losses["ycorr"] + y_recon2
        #print(l2_diff)
        #print(y_recon.shape, y.shape, mask_sum.shape, l2_diff)
        probs = F.softmax(-l2_diff.detach()/(W*0.125), dim=-1).detach()
        #print(probs)
        #get argmax
        #inds = torch.argmax(probs, dim=-1, keepdim=True)
        #inds = inds.unsqueeze(-1).repeat(1, 1, 3)
        ##get euler
        ##print(inds, euler_samples)
        #top_euler = torch.gather(euler_samples, 1, inds).squeeze(1).cpu()

        #get k argmax
        #inds_ret = torch.topk(probs, 16, dim=-1)
        #inds = inds_ret.indices
        #vals = inds_ret.values
        #l2_diff_top_k = torch.gather(l2_diff, 1, inds)
        #em_l2_loss = ((l2_diff_top_k*vals/mask_sum).sum(-1)).mean()
        #print(vals, inds, l2_diff_top_k, em_l2_loss, euler_samples)

        #print(top_euler)
        #print(probs, euler_samples)
        #print(l2_diff.shape, probs.shape, mask_sum.shape)
        em_l2_loss = ((l2_diff*probs/mask_sum.unsqueeze(1)).sum(-1))#.mean()
        #calculate snr
        #print(y.shape, em_l2_loss.shape, mask_sum.shape, y.pow(2).sum(dim=(-1,-2)))
        #y2  = y.pow(2).sum(dim=(-1,-2,-3)).squeeze()/mask_sum.squeeze()
        y2 = losses["y2"].squeeze()/mask_sum.squeeze()
        mse = em_l2_loss.detach() + y2
        snr = (y_recon2*probs).sum(-1).squeeze()/mask_sum.squeeze()/mse
        snr = snr.mean()
        #print(mse.shape, y2.shape)
        #snr = (1. - (mse/y2).mean())
        em_l2_loss = em_l2_loss.mean() #(alpha*x)^2 sigma2
        #print(em_l2_loss)
    else:
        em_l2_loss = (-2.*y_recon*y + y_recon**2).sum(dim=(-1,-2,-3))
        #print(em_l2_loss.shape, mask_sum.shape)
        em_l2_loss = torch.mean(em_l2_loss/mask_sum)#/(B*C)

    gen_loss = em_l2_loss
    assert torch.isnan(gen_loss).item() is False

    # set a unified mask_sum
    mask_sum = mask_sum.max()
    # latent loss
    kld = losses['kldiv'].mean() if 'kldiv' in losses else torch.tensor(0.)
    if body_poses_pred is not None:
        body_rots_pred, body_rots, body_trans_pred, body_trans = body_poses_pred
        rot_loss = lie_tools.rotation_loss(body_rots_pred, body_rots).mean()
        tran_loss = lie_tools.translation_loss(body_trans_pred, body_trans).mean()
    else:
        rot_loss, tran_loss = torch.tensor(0.), torch.tensor(0.)
    # total loss
    mu2, std2 = torch.tensor(0.), torch.tensor(0.)

    if "mu2" in losses:
        mu2 = losses["mu2"]
    if "std2" in losses:
        std2 = losses["std2"]
        #z_snr = (mu2/std2)
        z_snr = std2
        #z_mu_diff = z_mu.unsqueeze(1) - z_mu.unsqueeze(0) #(B, B, z)
        #print(z_mu_diff.shape, z_mu.shape)
        #logvar_diff = z_logstd.unsqueeze(1) - z_logstd.unsqueeze(0) #(B, B, z)
        #var_diff    = (2.*logvar_diff).exp()
        #z_mu_diff2  = z_mu_diff.pow(2)/((2.*z_logstd).exp() + 1e-6).unsqueeze(0) #(B, B, z) / (1, B, z)

    #print(losses["kldiv"].shape, losses["tvl2"].shape)

    lamb = args.lamb * (1. - torch.exp(-torch.clamp(0.5*(snr.detach()/0.01)**2, max=16))) #*torch.clamp(snr.abs().sqrt().detach(), max=1.)
    eps = 1e-3
    #kld, mu2 = utils.compute_kld(z_mu, z_logstd)
    #cross_corr = utils.compute_cross_corr(z_mu)
    loss = gen_loss + beta_control*beta*(kld)/mask_sum + torch.clamp(snr.abs().sqrt().detach(), max=0.5)*torch.mean(1e-1*losses['tvl2'] + 3e-1*losses['l2'])/(mask_sum)
    if body_poses_pred is not None:
        loss = loss #+ (rot_loss*body_rots_pred.shape[1] + tran_loss*body_trans_pred.shape[1])*4./mask_sum
    # compute mmd
    #mmd = utils.compute_smmd(z_mu, z_logstd, s=.5)
    #c_mmd = utils.compute_cross_smmd(z_mu, mus, s=1/16, adaptive=False)
    # matching z dim to image space
    # compute cross entropy
    c_en = (z_mu.unsqueeze(1) - mus).pow(2).sum(-1) + eps #(B, P)
    c_neg_en = (z_mu.unsqueeze(1) - neg_mus).pow(2).sum(-1) + eps #(B, N)
    #prob = ((-c_en).exp() + eps)/((-c_en).exp() + (-c_neg_en).exp().sum(-1, keepdim=True) + eps)
    #print(c_en.shape, c_neg_en.shape, prob.shape)
    #c_mmd = -torch.log(prob).mean()
    c_mmd = torch.log(1 + c_en).mean() + 3*torch.log(1 + 1./c_neg_en).mean()
    # compute cross entropy based on deconvoluted image
    #deconv_recon = y_recon_ori[:,0,...]
    #deconv_recon = utils.downsample_image(deconv_recon, deconv_recon.shape[-1]//2)
    #probs = deconv_recon.unsqueeze(1) - deconv_recon.unsqueeze(0)
    #probs = (probs.pow(2).mean(dim=(-1, -2)))
    #probs_med = torch.median(probs).detach() #(n)
    #probs = (-probs/probs_med).exp()
    # spread probs
    diff = (z_mu.unsqueeze(1) - z_mu.unsqueeze(0)).pow(2).sum(dim=(-1)) + eps
    #spread_probs = (-diff).exp()
    #spread_probs = (spread_probs + eps)/(spread_probs.sum(-1) + eps)
    #mmd = -torch.log(spread_probs).mean()
    diag_mask = (~torch.eye(B, dtype=bool).to(z_mu.get_device())).float()
    mmd = torch.log(1 + 1./diff)*torch.clip(diff.detach(), max=1)*diag_mask
    mmd = mmd.mean()
    #print("c_mmd: ", c_mmd, "mmd: ", mmd)
    loss += lamb*(c_mmd + mmd)*((beta+0.05)/1.05)/mask_sum
    if "knn" in losses:
        loss += lamb*losses["knn"]*((beta+0.05)/1.05)/mask_sum
    #loss = loss/(1 + lamb*beta)
    #print(mmd)

    if it % (args.log_interval) == B and args.plot:
        #group_stat.plot_variance(ind[0])
        log(f"mask_sum {mask_sum.detach().cpu()}, lamb {lamb}")
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        #print(probs)
        torch.set_printoptions(profile='default')
        #print(logvar_diff.mean().detach(), z_mu_diff2.mean().detach(), var_diff.mean().detach())
        plt.savefig(args.tmp_prefix+'.png')
        #plt.show()

    return loss, gen_loss, snr, mu2.mean(), z_snr.mean(), rot_loss, tran_loss, top_euler, y2.mean()

def eval_z(model, lattice, data, batch_size, device, trans=None, use_tilt=False, ctf_params=None, use_real=False):
    assert not model.training
    z_mu_all = []
    z_logvar_all = []
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False)
    for minibatch in data_generator:
        ind = minibatch[-1]
        y = minibatch[0].to(device)
        yt = minibatch[1].to(device) if use_tilt else None
        B = len(ind)
        D = lattice.D
        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B,*lattice.freqs2d.shape)/ctf_params[ind,0].view(B,1,1)
            c = ctf.compute_ctf(freqs, *torch.split(ctf_params[ind,1:], 1, 1)).view(B,D,D)
        if trans is not None:
            y = lattice.translate_ht(y.view(B,-1), trans[ind].unsqueeze(1)).view(B,D,D)
            if yt is not None: yt = lattice.translate_ht(yt.view(B,-1), trans[ind].unsqueeze(1)).view(B,D,D)
        if use_real:
            input_ = (torch.from_numpy(data.particles_real[ind]).to(device),)
        else:
            input_ = (y,yt) if yt is not None else (y,)
        if ctf_params is not None:
            assert not use_real, "Not implemented"
            input_ = (x*c.sign() for x in input_) # phase flip by the ctf
        z_mu, z_logvar = _unparallelize(model).encode(*input_)
        z_mu_all.append(z_mu.detach().cpu().numpy())
        z_logvar_all.append(z_logvar.detach().cpu().numpy())
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    return z_mu_all, z_logvar_all

def save_checkpoint(model, optim, posetracker, pose_optimizer, epoch,
                    z_mu, z_logvar, out_weights, out_z, vanilla=False, pose_encoder=None, out_pose=""):
    '''Save model weights, latent encoding z, and decoder volumes'''
    # save model weights
    torch.save({
        'epoch':epoch,
        'model_state_dict':_unparallelize(model).state_dict(),
        'encoder_state_dict':_unparallelize(model.encoder).state_dict(),
        'decoder_state_dict':_unparallelize(model.decoder).state_dict(),
        'optimizer_state_dict':optim.state_dict(),
        'pose_state_dict':posetracker.state_dict(),
        'pose_optimizer_state_dict':pose_optimizer.state_dict() if pose_optimizer is not None else None,
        }, out_weights)
    # save z
    if vanilla:
        posetracker.save_emb(out_z)
        posetracker.save(out_pose)
    if not vanilla:
        with open(out_z,'wb') as f:
            pickle.dump(z_mu, f)
            pickle.dump(z_logvar, f)

def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        keepreal=args.use_real,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        ctf=args.ctf,
                        poses=args.poses,
                        do_pose_sgd=args.do_pose_sgd,
                        real_data=args.real_data,
                        downfrac=args.downfrac)
    if args.tilt is not None:
        dataset_args['particles_tilt'] = args.tilt
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(qlayers=args.qlayers,
                      qdim=args.qdim,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      encode_mode=args.encode_mode,
                      enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      pe_dim=args.pe_dim,
                      domain=args.domain,
                      activation=args.activation,
                      template_type=args.template_type,
                      down_vol_size=model.down_vol_size,
                      Apix=model.decoder.Apix,
                      templateres=model.templateres)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args)
    config['seed'] = args.seed
    with open(out_config,'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv,
                    version=cryodrgn.__version__)
        pickle.dump(meta, f)

def get_latest(args):
    # assumes args.num_epochs > latest checkpoint
    log('Detecting latest checkpoint...')
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    log(f'Loading {args.load}')
    if args.do_pose_sgd:
        i = args.load.split('.')[-2]
        args.poses = f'{args.outdir}/pose.{i}.pkl'
        assert os.path.exists(args.poses)
        log(f'Loading {args.poses}')
    return args

def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'
    def flog(msg): # HACK: switch to logging module
        return utils.flog(msg, LOG)
    if args.load == 'latest':
        args = get_latest(args)
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
            data = dataset.LazyTomoMRCData(args.particles, norm=args.norm,
                                       real_data=args.real_data, invert_data=args.invert_data,
                                       ind=ind, keepreal=args.use_real, window=False,
                                       datadir=args.datadir, relion31=args.relion31, window_r=args.window_r, downfrac=args.downfrac)
        elif args.lazy_single and args.warp:
            data = dataset.LazyTomoWARPMRCData(args.particles, norm=args.norm,
                                       real_data=args.real_data, invert_data=args.invert_data,
                                       ind=ind, keepreal=args.use_real, window=False,
                                       datadir=args.datadir, relion31=args.relion31, window_r=args.window_r, downfrac=args.downfrac,
                                       tilt_step=args.tilt_step, tilt_range=args.tilt_range)
        else:
            raise NotImplementedError("Use --lazy-single for on-the-fly image loading")

    Nimg = data.N
    D = data.D #data dimension

    if args.encode_mode == 'conv':
        assert D-1 == 64, "Image size must be 64x64 for convolutional encoder"
    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        if args.num_gpus is not None:
            num_gpus = min(args.num_gpus, torch.cuda.device_count())
        else:
            num_gpus = torch.cuda.device_count()
        args.batch_size *= num_gpus

    # load poses
    #if args.do_pose_sgd: assert args.domain == 'hartley', "Need to use --domain hartley if doing pose SGD"
    do_pose_sgd = args.do_pose_sgd
    do_deform   = args.warp_type == 'deform' or args.encode_mode == 'grad'
    # use D-1 instead of D
    posetracker = PoseTracker.load(args.poses, Nimg, D-1, 's2s2' if do_pose_sgd else None, ind,
                                   deform=do_deform, deform_emb_size=args.zdim, latents=args.latents, batch_size=args.batch_size)
    posetracker.to(device)
    pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr) if do_pose_sgd else None

    # load masks
    if args.masks:
        masks_params = torch.load(args.masks)
    else:
        masks_params = None

    # load ctf
    if args.ctf is not None:
        flog('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D-1, args.ctf)
        log('first ctf params is: {}'.format(ctf_params[0,:]))
        if args.ind is not None: ctf_params = ctf_params[ind]
        assert ctf_params.shape == (Nimg, 8)
        ctf_params = torch.tensor(ctf_params).to(device)
    else: ctf_params = None

    if args.group is not None:
        grp_vol_size = 128
        group_assignment = ctf.load_group_for_training(args.group)
        #group_assignment = posetracker.euler_groups
        if args.group_stat is not None:
            group_stat = GroupStat.load(group_assignment, args.group_stat, device, D, vol_size=grp_vol_size,
                                        optimize_b=args.optimize_b).to(device)
        else:
            group_stat = GroupStat(group_assignment, device, D, vol_size=grp_vol_size, optimize_b=args.optimize_b).to(device)
    else:
        group_assignment = None
        group_stat = None

    # instantiate model
    lattice = Lattice(D, extent=0.5)
    grid = Grid(D, device)
    ctf_grid = None #CTFGrid(D, device)
    if args.enc_mask is None:
        args.enc_mask = D//2
    if args.enc_mask > 0:
        assert args.enc_mask <= D//2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        enc_mask = None
        in_dim = lattice.D**2 if not args.use_real else (lattice.D-1)**2
    else:
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                activation=activation, ref_vol=ref_vol, Apix=args.angpix,
                template_type=args.template_type, warp_type=args.warp_type,
                device=device, symm=None, ctf_grid=ctf_grid,
                deform_emb_size=args.deform_size,
                downfrac=args.downfrac,
                templateres=args.templateres,
                tmp_prefix=args.tmp_prefix,
                masks_params=masks_params,
                z_affine_dim=args.zaffinedim,
                ctf_alpha=args.ctfalpha,
                ctf_beta=args.ctfbeta)

    # use downsampled ctf grid
    ctf_grid = CTFGrid(model.render_size+1, device)

    flog(model)
    print("template_type: ", args.template_type)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    flog('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    flog('{} parameters in decoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))
    if args.estpose:
        flog('OPUS-TOMO will estimate pose using neural network')

    # save configuration
    out_config = '{}/config.pkl'.format(args.outdir)
    save_config(args, data, lattice, model, out_config)
    # move model to gpu
    model = model.to(device)

    # set model parameters to be encoder and decoder
    #model_parameters = list(model.parameters())+list(group_stat.parameters())
    model_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters()) #+ list(group_stat.parameters())
    pose_encoder = None
    optim = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.wd)

    #if args.encode_mode == "grad":
    #    discriminator_parameters = list(model.shape_encoder.parameters())
    #    optimD = torch.optim.AdamW(discriminator_parameters, lr=args.lr, weight_decay=args.wd)

    # learning rate scheduler
    #warm_up_epochs = 2
    #max_num_epochs = 12
    #warm_up_with_cosine_lr = lambda epoch: (epoch + 1)/warm_up_epochs if epoch < warm_up_epochs else \
    #                                 0.5 * (math.cos((epoch - warm_up_epochs)/(max_num_epochs - warm_up_epochs) * \
    #                                                 math.pi) + 1.)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warm_up_with_cosine_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.98)

    # Mixed precision training with AMP
    if args.amp:
        assert args.batch_size % 8 == 0, "Batch size must be divisible by 8 for AMP training"
        assert (D-1) % 8 == 0, "Image size must be divisible by 8 for AMP training"
        assert args.pdim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdim % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        # Also check zdim, enc_mask dim? Add them as warnings for now.
        if args.zdim % 8 != 0:
            log('Warning: z dimension is not a multiple of 8 -- AMP training speedup is not optimized')
        if in_dim % 8 != 0:
            log('Warning: Masked input image dimension is not a mutiple of 8 -- AMP training speedup is not optimized')
        model, optim = amp.initialize(model, optim, opt_level='O1')

    # restart from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        print(checkpoint.keys())
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        #print(pretrained_dict, model_dict)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        if True:
            pretrained_dict = checkpoint['encoder_state_dict']
            model_dict = model.encoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "transformer" not in k and "mask" not in k and "grid" not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.encoder.load_state_dict(model_dict)

            pretrained_dict = checkpoint['decoder_state_dict']
            model_dict = model.decoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "transformer" not in k and "mask" not in k and "grid" not in k and "radius" not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.decoder.load_state_dict(model_dict)

        pretrained_pose_dict = checkpoint['pose_state_dict']
        pose_dict = posetracker.state_dict()
        pretrained_pose_dict = {k: v for k, v in pretrained_pose_dict.items() if k in pose_dict}

        posetracker.load_state_dict(pretrained_pose_dict)

        #pose_optimizer.load_state_dict(checkpoint['pose_optimizer_state_dict'])
        #model.load_state_dict(checkpoint['model_state_dict'])
        #optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        model.train()
    else:
        start_epoch = 0

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        if args.num_gpus is not None:
            num_gpus = min(args.num_gpus, torch.cuda.device_count())
        else:
            num_gpus = torch.cuda.device_count()
        device_ids = [x for x in range(num_gpus)]
        log(f'Using {num_gpus} GPUs!')
        #model = nn.DataParallel(model, device_ids=device_ids)
        model.encoder = nn.DataParallel(model.encoder, device_ids=device_ids)
        model.decoder = nn.DataParallel(model.decoder, device_ids=device_ids)
        #patch_replication_callback(model)
    elif args.multigpu:
        log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # create classwise sampler
    if not os.path.exists(args.split):
        rand_split = torch.randperm(Nimg)
        torch.save(rand_split, args.split)
    else:
        log(f'loading train validation split from {args.split}')
        rand_split = torch.load(args.split)
        assert len(rand_split) == Nimg, "the split file should have length {Nimg}"
    Nimg_train = int(Nimg*(1. - args.valfrac))
    Nimg_test = int(Nimg*args.valfrac)
    train_split, val_split = rand_split[:Nimg_train], rand_split[Nimg_train:]
    train_sampler = dataset.ClassSplitBatchSampler(args.batch_size, posetracker.poses_ind, train_split)
    val_sampler = dataset.ClassSplitBatchSampler(args.batch_size, posetracker.poses_ind, val_split)
    print("Nimg_train: ", Nimg_train, len(train_split))
    data_generator = DataLoader(data, batch_sampler=train_sampler, pin_memory=True, num_workers=16)
    val_data_generator = DataLoader(data, batch_sampler=val_sampler, pin_memory=True, num_workers=16)

    #assert args.downfrac*(D-1) >= 128
    log(f'image will be downsampled to {args.downfrac} of original size {D-1}')
    log(f'reconstruction will be blurred by bfactor {args.bfactor}')

    # learning rate scheduler
    # training loop
    # data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.num_epochs

    vanilla = args.pe_type == "vanilla"
    global_it = 0
    bfactor = args.bfactor
    lamb = args.lamb
    if args.log_interval % args.batch_size != 0:
        args.log_interval = args.batch_size*16
    assert args.accum_step >= 1

    for epoch in range(start_epoch, num_epochs):
        t2 = dt.now()
        gen_loss_accum = 0
        loss_accum = 0
        snr_accum = 0
        eq_loss_accum = 0
        batch_it = 0
        # moving average
        ema_mu = 0.99
        gen_loss_ema, gen_loss_var_ema = 0.0, 0.
        snr_ema = 0.001
        mse_ema = 0.01
        mse_var_ema = 0.1
        mmd_ema = 0.
        mmd_var_ema = 0.1
        c_mmd_ema, c_mmd_var_ema = 0., 0.1
        update_it = 0
        beta_control = args.beta_control
        #increasing bfactor slowly
        args.bfactor = bfactor*(1. + 0.25/(1. + 3.*math.exp(-0.1*epoch)))
        beta_max    = 1. #0.98 ** (epoch)
        log('learning rate {}, bfactor: {}, beta_max: {}, beta_control: {} for epoch {}'.format(
                        lr_scheduler.get_last_lr(), args.bfactor, beta_max, beta_control, epoch))

        loop = tqdm(enumerate(data_generator), total=len(data_generator), leave=True, colour='green', file=sys.stdout)
        optim.zero_grad()
        for batch_idx, minibatch in loop:
        #for minibatch in data_generator:
            ind = minibatch[-1]#.to(device)
            y = minibatch[0][0].to(device, non_blocking=True)
            ctf_param = minibatch[0][1].float().to(device, non_blocking=True)
            ctf_filename = minibatch[0][2]
            #apixs = torch.ones(ctf_param.shape[:-1]).to(device)*args.angpix
            #ctf_param = torch.cat([apixs.unsqueeze(-1), ctf_param], dim=-1)
            # compute ctf!
            freqs = ctf_grid.freqs2d.unsqueeze(0)/args.angpix #(1, (-x+1, x)*x, 2)
            #res = torch.split(ctf_param, 1, -1)
            #print(res, res[0].shape)
            #ctf3d = ctf.compute_3dctf(y, ctf_grid.centered_freqs, freqs, *torch.split(ctf_param, 1, -1))#.view(B,D-1,-1) #(B, )
            yt = None
            B = len(ind)
            if B % args.num_gpus != 0:
                continue
            batch_it += B
            global_it = Nimg_train*epoch + batch_it
            save_image = (batch_it % (args.log_interval)) == B

            beta_control = args.beta_control*snr_ema
            beta = beta_schedule(global_it) * beta_max

            yr = None#torch.from_numpy(data.particles_real[ind.numpy()]).to(device) if args.use_real else None
            if do_pose_sgd:
                pose_optimizer.zero_grad()
            rot, tran = posetracker.get_pose(ind)
            euler = posetracker.get_euler(ind)
            rot = rot.to(device)
            tran = tran.to(device)
            euler = euler.to(device)
            body_euler, body_trans = posetracker.get_body_pose(ind)
            if body_euler is not None:
                body_euler = body_euler.to(device)
                body_trans = body_trans.to(device)

            o_rot = lie_tools.hopf_to_SO3(euler[:, :3])
            ## perturb rotation by symm ops
            #samples = torch.multinomial(symm_ops_weights, o_rot.shape[0], replacement=True)

            ###rand_z = o_rot @ symm_ops[samples].to(o_rot.get_device())
            ###print(rand_z)
            ####pixrad = hp.max_pixrad(64)
            #rand_z = lie_tools.random_biased_SO3(o_rot.shape[0], bias=256*np.sqrt(3)).to(o_rot.get_device())
            #rand_z = o_rot @ rand_z
            #rand_e = lie_tools.so3_to_hopf(rand_z)
            ##print(rand_e - euler[:, :3])
            #euler = rand_e

            #print("euler, trans: ", euler.shape, tran.shape, y.shape)
            #ctf_param = ctf_params[ind] if ctf_params is not None else None
            z_mu, loss, gen_loss, snr, l1_loss, tv_loss, mu2, std2, mmd, c_mmd, mse, body_poses_pred = \
                                        train_batch(model, lattice, y, yt, rot, tran, optim, beta,
                                              beta_control=beta_control, tilt=tilt, ind=ind,
                                              grid=grid, ctf_params=ctf_param, ctf_grid=ctf_grid,
                                              yr=yr, use_amp=args.amp, vanilla=vanilla,
                                              save_image=save_image, group_stat=group_stat,
                                              it=batch_it, enc=None,
                                              args=args, euler=euler,
                                              posetracker=posetracker, data=data, update_params=(update_it%args.accum_step == args.accum_step - 1),
                                              snr2=snr_ema, body_poses=(body_euler, body_trans), ctf_filename=ctf_filename)
            update_it += 1
            if do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()
            #if args.encode_mode == 'grad':
            #    posetracker.set_emb(z_mu, ind)

            # logging
            # compute moving average of generator loss, snr, and mmd
            delta         = snr - snr_ema
            snr_ema += (1-ema_mu)*delta
            delta         = gen_loss - gen_loss_ema
            gen_loss_ema += (1-ema_mu)*delta
            gen_loss_var_ema = ema_mu*(gen_loss_var_ema + (1-ema_mu)*delta**2)
            delta         = mse - mse_ema
            mse_ema += (1-ema_mu)*delta
            mse_var_ema = ema_mu*(mse_var_ema + (1-ema_mu)*delta**2)
            delta         = mmd - mmd_ema
            mmd_ema += (1-ema_mu)*delta
            mmd_var_ema = ema_mu*(mmd_var_ema + (1-ema_mu)*delta**2)
            delta         = c_mmd - c_mmd_ema
            c_mmd_ema += (1-ema_mu)*delta
            c_mmd_var_ema = ema_mu*(mmd_var_ema + (1-ema_mu)*delta**2)

            gen_loss_accum += gen_loss*B
            snr_accum += snr*B
            loss_accum += loss*B

            #if batch_it % args.log_interval == 0:
            #    log('# [Train Epoch: {}/{}] [{}/{} images] ' #gen_loss={:.6f}, '\
            #        'snr2_mu={:.3f}, beta={:.3f}, '                               \
            #        'loss={:.4f}, l1={:.3f}, tv={:.3f}, '                     \
            #        'mu={:.3f}, std={:.3f}, gen_loss_mu={:.4f}, ' \
            #        'gen_loss_std={:.3f}, mse_mu={:.3f}, mse_std={:.3f}, ' \
            #        'rot_mu={:.4f}, rot_std={:.4f}, trans_mu={:.4f}, trans_std={:.4f}'.format(epoch+1, num_epochs, batch_it,
            #                                        Nimg_train, snr_ema, beta, loss, l1_loss, tv_loss,
            #                                         np.sqrt(mu2), np.sqrt(std2), gen_loss_ema, np.sqrt(gen_loss_var_ema),
            #                                        mse_ema, np.sqrt(mse_var_ema), mmd_ema, np.sqrt(mmd_var_ema), c_mmd_ema, np.sqrt(c_mmd_var_ema)))
            loop.set_description(f'Train Epoch: [{epoch+1}/{num_epochs}]')
            loop.set_postfix(beta=beta, loss=loss, snr=snr, mu=np.sqrt(mu2), std=np.sqrt(std2),)
            if batch_it % args.log_interval == 0:
                tqdm.write("Additional info at {}, l1={:.5f}, tv={:.5f}, snr={:.4f}, mse={:.4f}, gen_loss={:.5f}".format(
                                        batch_it, l1_loss, tv_loss, snr_ema, mse_ema, gen_loss_ema), file=sys.stdout)

            if batch_it % (args.log_interval*10) == 0:
                out_z = '{}/z.{}.pkl'.format(args.outdir, epoch)
                log('save {}'.format(out_z))
                posetracker.save_emb(out_z)
                out_pose = '{}/pose.{}.pkl'.format(args.outdir, epoch)
                log('save {}'.format(out_pose))
                posetracker.save(out_pose)

        flog('# =====> Epoch: {} Average training gen_loss = {:.6}, SNR2 = {:.6f}, '\
             'total loss = {:.6f}; Finished in {}'.format(epoch+1,
                                                         gen_loss_accum/Nimg_train,
                                                         snr_accum/Nimg_train, loss_accum/Nimg_train, dt.now()-t2))

        # validation
        gen_loss_accum, snr_accum, loss_accum = 0, 0, 0
        loop = tqdm(enumerate(val_data_generator), total=len(val_data_generator), leave=True, file=sys.stdout)
        for batch_idx, minibatch in loop:
        #for minibatch in val_data_generator:
            ind = minibatch[-1]
            yt = None
            y = minibatch[0][0].to(device, non_blocking=True)
            ctf_param = minibatch[0][1].float().to(device, non_blocking=True)
            ctf_filename = minibatch[0][2]
            B = len(ind)
            if B % args.num_gpus != 0:
                continue
            batch_it += B
            save_image = False
            beta = beta_schedule(global_it)

            yr = None #torch.from_numpy(data.particles_real[ind.numpy()]).to(device) if args.use_real else None
            rot, tran = posetracker.get_pose(ind)
            euler = posetracker.get_euler(ind)
            rot = rot.to(device)
            tran = tran.to(device)
            euler = euler.to(device)
            body_euler, body_trans = posetracker.get_body_pose(ind)
            if body_euler is not None:
                body_euler = body_euler.to(device)
                body_trans = body_trans.to(device)
            #ctf_param = ctf_params[ind] if ctf_params is not None else None
            z_mu, loss, gen_loss, snr, l1_loss, tv_loss, mu2, std2, mmd, c_mmd, mse, body_poses_pred = \
                                        train_batch(model, lattice, y, yt, rot, tran, optim, beta,
                                              beta_control=beta_control, tilt=tilt, ind=ind,
                                              grid=grid, ctf_params=ctf_param, ctf_grid=ctf_grid,
                                              yr=yr, use_amp=args.amp, vanilla=vanilla,
                                              save_image=save_image, group_stat=group_stat,
                                              it=batch_it, enc=None,
                                              args=args, euler=euler,
                                              posetracker=posetracker, data=data, backward=False, update_params=False,
                                              snr2=snr_ema, body_poses = (body_euler, body_trans), ctf_filename=ctf_filename)
            if do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()
            # logging
            gen_loss_accum += gen_loss*B
            snr_accum += snr*B
            loss_accum += loss*B

        flog('# =====> Epoch: {} Average validation gen_loss = {:.6}, SNR2 = {:.6f}, '\
             'total loss = {:.6f}; Finished in {}'.format(epoch+1,
                                                         gen_loss_accum/(Nimg_test+1),
                                                         snr_accum/(Nimg_test+1), loss_accum/(Nimg_test+1), dt.now()-t2))


        if args.checkpoint and epoch % args.checkpoint == 0:
            out_weights = '{}/weights.{}.pkl'.format(args.outdir,epoch)
            out_z = '{}/z.{}.pkl'.format(args.outdir, epoch)
            out_pose = '{}/pose.{}.pkl'.format(args.outdir, epoch)
            model.eval()
            with torch.no_grad():
                if not vanilla:
                    z_mu, z_logvar = eval_z(model, lattice, data, args.batch_size,
                                            device, posetracker.trans, tilt is not None, ctf_params, args.use_real)
                else:
                    z_mu = None
                    z_logvar = None

                save_checkpoint(model, optim, posetracker, pose_optimizer,
                                epoch, z_mu, z_logvar, out_weights, out_z, vanilla=vanilla, out_pose=out_pose)
            if args.do_pose_sgd and epoch >= args.pretrain:
                out_pose = '{}/pose.{}.pkl'.format(args.outdir, epoch)
                posetracker.save(out_pose)
        #update learning rate
        lr_scheduler.step()
    # save model weights, latent encoding, and evaluate the model on 3D lattice
    #out_weights = '{}/weights.pkl'.format(args.outdir)
    #out_z = '{}/z.pkl'.format(args.outdir)
    #model.eval()
    #with torch.no_grad():
    #    if not vanilla:
    #        z_mu, z_logvar = eval_z(model, lattice, data, args.batch_size, device, posetracker.trans, tilt is not None, ctf_params, args.use_real)
    #    else:
    #        z_mu = None
    #        z_logvar = None
    #    save_checkpoint(model, optim, posetracker, pose_optimizer, epoch, z_mu, z_logvar, out_weights, out_z, vanilla=vanilla)

    #if args.do_pose_sgd and epoch >= args.pretrain:
    #    out_pose = '{}/pose.pkl'.format(args.outdir)
    #    posetracker.save(out_pose)
    td = dt.now()-t1
    flog('Finsihed in {} ({} per epoch)'.format(td, td/(num_epochs-start_epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)
