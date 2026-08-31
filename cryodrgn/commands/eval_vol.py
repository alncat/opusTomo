'''
Evaluate the decoder at specified values of z
'''
import numpy as np
import os
import argparse
from datetime import datetime as dt
import pprint

import torch
import torch.nn as nn

from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import config
from cryodrgn.lattice import Lattice
from cryodrgn.models import HetOnlyVAE

log = utils.log
vlog = utils.vlog

def add_args(parser):
    #parser.add_argument('weights', help='Model weights')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('-c', '--config', metavar='PKL', required=True, help='CryoDRGN config.pkl file')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .mrc or directory')
    parser.add_argument('--prefix', default='reference', help='Prefix when writing out multiple .mrc files (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')

    group = parser.add_argument_group('Specify z values')
    group.add_argument('-z', type=np.float32, nargs='*', help='Specify one z-value')
    group.add_argument('--z-start', type=np.float32, nargs='*', help='Specify a starting z-value')
    group.add_argument('--z-end', type=np.float32, nargs='*', help='Specify an ending z-value')
    group.add_argument('-n', type=int, default=10, help='Number of structures between [z_start, z_end]')
    group.add_argument('--zfile', help='Text file with z-values to evaluate')
    group.add_argument('--deform', action='store_true', help='deforming the structure')
    group.add_argument('--template-z', help='path for template encoding')
    group.add_argument('--template-z-ind', type=int, help='the index of the selected template encoding')
    group.add_argument('--masks', help='path for the masks')
    group.add_argument('--num-bodies', type=int, default=0, help='number of rigid bodies')

    group = parser.add_argument_group('Volume arguments')
    group.add_argument('--Apix', type=float, default=1, help='Pixel size to add to .mrc header (default: %(default)s A/pix)')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('-d','--downsample', type=int, help='Downsample volumes to this box size (pixels)')

    group = parser.add_argument_group('Overwrite architecture hyperparameters in config.pkl')
    group.add_argument('--norm', nargs=2, type=float)
    group.add_argument('-D', type=int, help='Box size')
    group.add_argument('--enc-layers', dest='qlayers', type=int, help='Number of hidden layers')
    group.add_argument('--enc-dim', dest='qdim', type=int, help='Number of nodes in hidden layers')
    group.add_argument('--zdim', type=int,  help='Dimension of latent variable')
    group.add_argument('--encode-mode', default='grad', choices=('conv','resid','mlp','tilt', 'grad'), help='Type of encoder network')
    group.add_argument('--dec-layers', dest='players', type=int, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, help='Number of nodes in hidden layers')
    group.add_argument('--enc-mask', type=int, help='Circular mask radius for image encoder')
    group.add_argument('--pe-type', default='vanilla', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf','none', 'vanilla'), help='Type of positional encoding')
    group.add_argument('--template-type', default='conv', choices=('conv'), help='Type of template decoding method (default: %(default)s)')
    group.add_argument('--warp-type', choices=('blurmix', 'diffeo', 'deform'), help='Type of warp decoding method (default: %(default)s)')
    group.add_argument('--symm', help='Type of symmetry of the 3D volume (default: %(default)s)')
    group.add_argument('--num-struct', type=int, default=1, help='Num of structures (default: %(default)s)')
    group.add_argument('--deform-size', type=int, default=2, help='Num of structures (default: %(default)s)')

    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--domain', choices=('hartley','fourier'))
    group.add_argument('--l-extent', type=float, help='Coordinate lattice size')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    return parser

# the mask buffers VanillaDecoder registers from masks_params, and how they were normalised
MASK_BUFFERS = ('com_bodies', 'in_relatives', 'rotate_directions', 'orient_bodies',
                'principal_axes', 'radius')

def masks_params_from_checkpoint(decoder_state, vol_size):
    '''Rebuild the mask_params.pkl dict from the buffers a multi-body checkpoint carries.

    VanillaDecoder.__init__ stores the geometry as buffers, dividing the lengths by vol_size,
    so the pkl written by prepare_multi is recoverable from any multi-body checkpoint. This
    lets eval_vol run without --masks: masks_params is what switches the decoder's affine head
    on, and building the decoder without it drops template.affine_out.* in the state-dict
    filter below and then fails in forward() on `affine[1][i]`.
    '''
    missing = [k for k in MASK_BUFFERS if k not in decoder_state]
    if missing:
        raise ValueError(f'the checkpoint is multi-body but is missing the mask buffers '
                         f'{missing}; pass --masks <mask_params.pkl> instead')
    com = decoder_state['com_bodies']
    return dict(com_bodies=com,
                in_relatives=decoder_state['in_relatives']*vol_size + com,
                rotate_directions=decoder_state['rotate_directions']*vol_size,
                orient_bodies=decoder_state['orient_bodies'],
                principal_axes=decoder_state['principal_axes'],
                radii_bodies=decoder_state['radius']*vol_size)

def check_inputs(args):
    if args.z_start:
        assert args.z_end, "Must provide --z-end with argument --z-start"
    assert sum((bool(args.z), bool(args.z_start), bool(args.zfile))) == 1, "Must specify either -z OR --z-start/--z-end OR --zfile"

def main(args):
    #check_inputs(args)
    t1 = dt.now()

    ## set the device
    use_cuda = torch.cuda.is_available()
    log('Use cuda {}'.format(use_cuda))
    device = torch.device('cuda' if use_cuda else 'cpu')
    #if use_cuda:
    #    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #else:
    #    log('WARNING: No GPUs detected')

    log(args)
    cfg = config.overwrite_config(args.config, args)
    log('Loaded configuration:')
    pprint.pprint(cfg)

    in_dim = -1
    enc_mask = -1
    D = cfg['lattice_args']['D'] # image size + 1
    zdim = cfg['model_args']['zdim']
    if "z_affine_dim" in cfg['model_args']:
        z_affine_dim = cfg['model_args']['z_affine_dim']
    else:
        z_affine_dim = 4
    norm = cfg['dataset_args']['norm']
    lattice = Lattice(D, extent=0.5)
    downfrac = cfg['dataset_args']['downfrac']
    crop_vol_size = cfg['model_args']['down_vol_size']
    # float() guards old configs that stored Apix as a (possibly CUDA) tensor; fov/render_apix
    # and the mrc header below all need a plain python float.
    Apix = float(cfg['model_args']['Apix'])
    templateres = cfg['model_args']['templateres']
    #args.Apix = down_vol_size/((D - 1)*downfrac*0.85)*Apix
    window_r = crop_vol_size/(int((D-1)*downfrac)//2*2)
    # Physical field of view (A) is conserved through the Fourier resample + crop. Capture
    # it before downfrac is mutated below, so the mrc header can carry the TRUE voxel spacing
    # (fov/render_size) instead of the nominal --Apix. That absorbs the even-box rounding of
    # render_size and keeps box*apix == physical size exactly.
    fov = (D - 1) * Apix * downfrac
    downfrac *= Apix/args.Apix

    log("Apix: changing from training apix {} to target apix {}".format(Apix, args.Apix))
    log("the output volume by convnet will further downsample by downfrac: {} to achieve desired apix".format(downfrac))
    assert templateres is not None
    log("templateres: output volume of convnet is of size {}".format(templateres))
    log("the final output volume rendered by spatial transformer is of size {}".format(int((D-1)*downfrac*window_r)))

    if args.downsample:
        assert args.downsample % 2 == 0, "Boxsize must be even"
        assert args.downsample <= D - 1, "Must be smaller than original box size"

    decoder_state = None
    if args.load:
        log('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load, map_location=device)
        print(checkpoint.keys())
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = checkpoint['decoder_state_dict']
        decoder_state = pretrained_dict
        if "principal_axes" in pretrained_dict:
            args.num_bodies = pretrained_dict["principal_axes"].shape[0]
            print("principal_axes: ", pretrained_dict["principal_axes"], "num_bodies: ", args.num_bodies)

    # load masks. This runs after the checkpoint peek above so a multi-body run can be
    # evaluated without --masks -- the geometry is in the checkpoint either way. Silently
    # leaving masks_params None on a multi-body checkpoint used to build a decoder with no
    # affine head, which then failed inside forward() with
    # "'NoneType' object is not subscriptable".
    if args.masks:
        masks_params = torch.load(args.masks, map_location=device)
    elif args.num_bodies > 0 and decoder_state is not None:
        masks_params = masks_params_from_checkpoint(decoder_state, D - 1)
        log('--masks was not given; rebuilt the {}-body mask parameters from the buffers in '
            '{}'.format(args.num_bodies, args.load))
    else:
        masks_params = None
        if args.deform:
            raise ValueError(
                '--deform needs a multi-body model, but --masks was not given and '
                + ('no --load checkpoint was given either' if decoder_state is None else
                   f'{args.load} has no principal_axes (it was not trained with --masks)')
                + '. Pass --masks <mask_params.pkl> from `dsdsh prepare_multi`, or --load a '
                  'multi-body checkpoint.')

    #create and load model
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                in_dim, zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                activation=activation, ref_vol=None, Apix=args.Apix,
                template_type=args.template_type, warp_type=args.warp_type,
                num_struct=args.num_struct,
                device=device, symm=args.symm, ctf_grid=None,
                deform_emb_size=args.deform_size, downfrac=downfrac,
                templateres=templateres, window_r=window_r, masks_params=masks_params,
                num_bodies=args.num_bodies, z_affine_dim=z_affine_dim)

    vanilla = args.pe_type == "vanilla"

    # true voxel spacing of the rendered (vanilla) volume = fov / render_size. Cropping to
    # down_vol_size does not change the spacing, so this is exact regardless of the even-box
    # rounding in render_size. The vanilla save calls below write this to the mrc header.
    render_apix = fov / model.render_size
    log("output: vanilla volumes will write true voxel apix {:.4f} to headers "
        "(render_size={}, target --Apix={})".format(render_apix, model.render_size, args.Apix))

    if args.load:
        log('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load, map_location=device)
        print(checkpoint.keys())
        #pretrained_dict = checkpoint['model_state_dict']
        #model_dict = model.state_dict()
        ##print(pretrained_dict, model_dict)
        ## 1. filter out unnecessary keys
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        ## 2. overwrite entries in the existing state dict
        #model_dict.update(pretrained_dict)
        ## 3. load the new state dict
        #model.load_state_dict(model_dict)

        if vanilla:
            pretrained_dict = checkpoint['encoder_state_dict']
            model_dict = model.encoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "grid" not in k and "mask" not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            #model.encoder.load_state_dict(model_dict)

            pretrained_dict = checkpoint['decoder_state_dict']
            #overwrite ref_mask
            if "principal_axes" in pretrained_dict:
                masks_params = {}
                print("principal_axes: ", pretrained_dict["principal_axes"], "vol_size: ", D-1)
                print("rotate_directions: ", pretrained_dict["rotate_directions"])
                model.decoder.com_bodies = pretrained_dict["com_bodies"]
                model.decoder.rotate_directions = pretrained_dict["rotate_directions"]
                model.decoder.orient_bodies = pretrained_dict["orient_bodies"]
                model.decoder.orient_bodiesT = pretrained_dict["orient_bodiesT"]
                model.decoder.principal_axes = pretrained_dict["principal_axes"]
                model.decoder.principal_axesT = pretrained_dict["principal_axesT"]
                model.decoder.radius = pretrained_dict["radius"]

            if "ref_mask" in pretrained_dict:
                model.decoder.ref_mask =  pretrained_dict["ref_mask"]
            model_dict = model.decoder.state_dict()
            # 1. filter out unnecessary keys
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "grid" not in k and "mask" not in k}
            for k in list(pretrained_dict.keys()):
                #if "affine_head" in k or "second_order_head" in k:
                if k not in model_dict or pretrained_dict[k].shape != model_dict[k].shape:
                    if k in model_dict:
                        print(k, pretrained_dict[k].shape, model_dict[k].shape)
                    #if k != "ref_mask":
                    del pretrained_dict[k]
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.decoder.load_state_dict(model_dict)

    model = model.to(device)

    model.eval()

    ### Multiple z ###
    if args.z_start or args.zfile:

        # --deform prepends a fixed template latent to every z below, so load it here for ALL
        # z sources. It used to be loaded only on the --zfile + vanilla path, which left
        # template_z undefined (NameError) for --z-start or non-vanilla runs.
        template_z = None
        if args.deform:
            assert args.template_z is not None, "--deform requires --template-z"
            assert args.template_z_ind is not None, "--deform requires --template-z-ind"
            # atleast_2d for the same reason as the z file below: a single-row template
            # loads as 1-D and the row index below would fail
            template_z = np.atleast_2d(np.loadtxt(args.template_z))
            len_template = template_z.shape[0]
            assert args.template_z_ind < len_template, f"template-z-ind {args.template_z_ind} must be smaller than {len_template}"
            template_z = torch.tensor(template_z[args.template_z_ind, :]).float().to(device)
            log(template_z)

        ### Get z values
        if args.z_start:
            args.z_start = np.array(args.z_start)
            args.z_end = np.array(args.z_end)
            z = np.repeat(np.arange(args.n,dtype=np.float32), zdim).reshape((args.n, zdim))
            z *= ((args.z_end - args.z_start)/(args.n-1))
            z += args.z_start
        else:
            if vanilla:
                #z = utils.load_pkl(args.zfile)
                # atleast_2d: a single-row zfile loads as 1-D, and iterating it below would
                # yield 0-d scalars instead of one z vector
                z = np.atleast_2d(np.loadtxt(args.zfile))
                z = torch.tensor(z).float().to(device)
            else:
                z = np.loadtxt(args.zfile).reshape(-1, zdim)

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        log(f'Generating {len(z)} volumes in {args.o}')
        for i,zz in enumerate(z):
            log(zz)
            if args.deform:
                #null_z = torch.zeros(zdim).to(device)
                if not torch.is_tensor(zz):  # --z-start yields numpy rows
                    zz = torch.tensor(zz).float().to(device)
                zz = torch.cat([template_z, zz], dim=-1)
            if vanilla:
                model.save_mrc(f'{args.o}/{args.prefix}'+str(i), enc=zz, Apix=render_apix, flip=args.flip)
            else:
                if args.downsample:
                    extent = lattice.extent * (args.downsample/(D-1))
                    vol = model.decoder.eval_volume(lattice.get_downsample_coords(args.downsample+1),
                                                    args.downsample+1, extent, norm, zz)
                else:
                    vol = model.decoder.eval_volume(lattice.coords, lattice.D, lattice.extent, norm, zz)
                out_mrc = '{}/{}{:03d}.mrc'.format(args.o, args.prefix, i)
                if args.flip:
                    vol = vol[::-1]
                mrc.write(out_mrc, vol.astype(np.float32), Apix=args.Apix)

    ### Single z ###
    else:
        #z = np.array(args.z)
        z = torch.randn(1, args.zdim).to(device)
        log(z)
        if vanilla:
            model.save_mrc(args.prefix, enc=z, Apix=render_apix, flip=args.flip)
            return
        if args.downsample:
            extent = lattice.extent * (args.downsample/(D-1))
            vol = model.decoder.eval_volume(lattice.get_downsample_coords(args.downsample+1),
                                            args.downsample+1, extent, norm, z)
        else:
            vol = model.decoder.eval_volume(lattice.coords, lattice.D, lattice.extent, norm, z)
        if args.flip:
            vol = vol[::-1]
        mrc.write(args.o, vol.astype(np.float32), Apix=args.Apix)

    td = dt.now()-t1
    log('Finsihed in {}'.format(td))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)

