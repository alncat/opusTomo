'''Parse image poses from RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle
import torch

from cryodrgn import utils
from cryodrgn import starfile
from cryodrgn import lie_tools
from cryodrgn.commands.filter_star import split_star
log = utils.log

def add_args(parser):
    parser.add_argument('input', help='RELION .star file')
    parser.add_argument('-D', type=int, required=True, help='Box size of reconstruction (pixels)')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--Apix', type=float, help='Pixel size (A); Required if translations are specified in Angstroms')
    parser.add_argument('-o', metavar='PKL', type=os.path.abspath, required=False, help='Output pose.pkl')
    parser.add_argument('--labels', metavar='PKL', type=os.path.abspath, required=False, help='Split the star by these cluster labels (kmeans{K}/labels.pkl), one entry per particle of THIS star; needs --outdir')
    parser.add_argument('--outdir', type=os.path.abspath, help='Directory for the per-cluster star files written by --labels')
    parser.add_argument('--poses', metavar='PKL', type=os.path.abspath, required=False, help='Load poses from given pkl')
    parser.add_argument('--out-star', metavar='STAR', type=os.path.abspath, required=False,
                        help='Write updated STAR file (e.g., with perturbed or loaded poses)')
    parser.add_argument('--perturb-rot', type=float, default=0.,
                        help='Randomly perturb each rotation by up to this many degrees')
    parser.add_argument('--perturb-trans', type=float, default=0.,
                        help='Randomly perturb each translation component by up to this many pixels')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible pose perturbation')

    return parser

def update_star_with_poses(s, euler, trans):
    df = s.df.copy()
    df['_rlnAngleRot'] = euler[:, 0]
    df['_rlnAngleTilt'] = euler[:, 1]
    df['_rlnAnglePsi'] = euler[:, 2]
    df['_rlnOriginX'] = trans[:, 0]
    df['_rlnOriginY'] = trans[:, 1]
    df['_rlnOriginZ'] = trans[:, 2]
    df.drop(columns=['_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst'],
            inplace=True, errors='ignore')
    s.df = df
    s.headers = list(df.columns)

def perturb_poses(euler, trans, max_rot_deg=0., max_shift_px=0., seed=None):
    rng = np.random.default_rng(seed)
    rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)

    if max_rot_deg > 0:
        axes = rng.normal(size=(len(euler), 3)).astype(np.float32)
        axis_norm = np.linalg.norm(axes, axis=1, keepdims=True)
        axis_norm[axis_norm == 0] = 1.
        axes /= axis_norm
        angles = rng.uniform(-max_rot_deg, max_rot_deg, size=len(euler)).astype(np.float32)
        delta_rot = lie_tools.axis_rot(torch.from_numpy(angles), torch.from_numpy(axes)).cpu().numpy()
        rot = delta_rot @ rot
        euler = lie_tools.so3_to_euler(torch.from_numpy(rot).float()).cpu().numpy()

    if max_shift_px > 0:
        trans = trans + rng.uniform(-max_shift_px, max_shift_px, size=trans.shape)

    return rot, euler, trans

def main(args):
    assert args.input.endswith('.star'), "Input file must be .star file"
    #assert args.o.endswith('.pkl'), "Output format must be .pkl"

    s = starfile.Starfile.load(args.input, relion31=args.relion31)
    N = len(s.df)
    log('{} particles'.format(N))

    # parse rotations
    keys = ('_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi')
    euler = np.empty((N,3))
    euler[:,0] = s.df['_rlnAngleRot']
    euler[:,1] = s.df['_rlnAngleTilt']
    euler[:,2] = s.df['_rlnAnglePsi']

    # parse translations
    trans = np.zeros((N,3))
    if '_rlnOriginX' in s.headers and '_rlnOriginY' in s.headers and '_rlnOriginZ' in s.headers:
        trans[:,0] = s.df['_rlnOriginX']
        trans[:,1] = s.df['_rlnOriginY']
        trans[:,2] = s.df['_rlnOriginZ']
    elif '_rlnOriginXAngst' in s.headers and '_rlnOriginYAngst' in s.headers and '_rlnOriginZAngst' in s.headers:
        assert args.Apix is not None, "Must provide --Apix argument to convert _rlnOriginXAngst and _rlnOriginYAngst translation units"
        trans[:,0] = s.df['_rlnOriginXAngst']
        trans[:,1] = s.df['_rlnOriginYAngst']
        trans[:,2] = s.df['_rlnOriginZAngst']
        trans /= args.Apix

    if args.poses:
        log(f'Load poses from {args.poses}')
        poses = utils.load_pkl(args.poses)
        load_trans = poses[1]
        load_trans *= args.D # convert from fraction to pixels
        load_eulers = poses[2]
        log(f'first euler: {load_eulers[0]}')
        log(f'first trans: {load_trans[0]}')
        euler = load_eulers.copy()
        trans = load_trans.copy()

    rot = np.asarray([utils.R_from_relion(*x) for x in euler], dtype=np.float32)
    if args.perturb_rot > 0 or args.perturb_trans > 0:
        rot, euler, trans = perturb_poses(euler, trans,
                                          max_rot_deg=args.perturb_rot,
                                          max_shift_px=args.perturb_trans,
                                          seed=args.seed)
        log(f'Applied random pose perturbation: rot <= {args.perturb_rot} deg, trans <= {args.perturb_trans} px')

    star_updated = bool(args.poses) or args.perturb_rot > 0 or args.perturb_trans > 0
    if star_updated:
        update_star_with_poses(s, euler, trans)

    if args.out_star is not None:
        out_dir = os.path.dirname(args.out_star)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        log(f'Writing updated STAR to {args.out_star}')
        s.write(args.out_star)

    log('Euler angles (Rot, Tilt, Psi):')
    log(euler[0])
    log('Converting to rotation matrix:')
    log(rot[0])
    if args.labels is not None:
        assert args.outdir is not None, "--labels needs --outdir"
        labels = np.asarray(utils.load_pkl(args.labels)).reshape(-1)
        log(f'Read labels from {args.labels}')
        assert len(labels) == N, \
            f"{args.labels} has {len(labels)} entries but {args.input} has {N} particles -- " \
            f"analyze writes labels in the ORIGINAL stack numbering, so split the star used " \
            f"for training rather than an already-filtered one"
        # splitting a star is filter_star's job; it keeps data_optics and every other block,
        # which cryodrgn.starfile's writer drops. Fall back to the in-memory writer only when
        # the poses here differ from what is on disk and there is no updated star to split.
        split_src = args.input if not star_updated else args.out_star
        if split_src is not None:
            split_star(split_src, labels, args.outdir, prefix='pre')
        else:
            log('WARNING: poses were modified but --out-star was not given; splitting the '
                'in-memory star, which drops data_optics and any other extra block')
            for i in range(labels.min(), labels.max()+1):
                out_file = args.outdir + "/pre" + str(i) + ".star"
                log(f'Writing {np.sum(labels==i)} particles in cluster {i} to {out_file}')
                s.write_subset(out_file, labels==i)

    log('Translations (pixels):')
    log(trans[0])

    # convert translations from pixels to fraction
    trans /= args.D

    # write output
    if args.o is not None:
        log(f'Writing {args.o}')
        with open(args.o,'wb') as f:
            pickle.dump((rot,trans,euler),f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
