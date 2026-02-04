'''Parse image poses from RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle

from cryodrgn import utils
from cryodrgn import starfile
log = utils.log

def add_args(parser):
    parser.add_argument('input', help='RELION .star file')
    parser.add_argument('-D', type=int, required=True, help='Box size of reconstruction (pixels)')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--Apix', type=float, help='Pixel size (A); Required if translations are specified in Angstroms')
    parser.add_argument('-o', metavar='PKL', type=os.path.abspath, required=False, help='Output pose.pkl')
    parser.add_argument('--labels', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--outdir', type=os.path.abspath, help='The directory for storing starfiles for clusters')
    parser.add_argument('--poses', metavar='PKL', type=os.path.abspath, required=False, help='Load poses from given pkl')

    return parser

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
    if '_rlnOriginX' in s.headers and '_rlnOriginY' in s.headers:
        trans[:,0] = s.df['_rlnOriginX']
        trans[:,1] = s.df['_rlnOriginY']
        trans[:,2] = s.df['_rlnOriginZ']
    elif '_rlnOriginXAngst' in s.headers and '_rlnOriginYAngst' in s.headers:
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

        df = s.df.copy()
        df['_rlnAngleRot'] = load_eulers[:, 0]
        df['_rlnAngleTilt'] = load_eulers[:, 1]
        df['_rlnAnglePsi'] = load_eulers[:, 2]
        df['_rlnOriginX'] = load_trans[:, 0]
        df['_rlnOriginY'] = load_trans[:, 1]
        df['_rlnOriginZ'] = load_trans[:, 2]
        df.drop(columns=['_rlnOriginXAngst', '_rlnOriginYAngst', '_rlnOriginZAngst'], inplace=True, errors='ignore')
        s.df = df
        s.headers = list(df.columns)

    log('Euler angles (Rot, Tilt, Psi):')
    log(euler[0])
    log('Converting to rotation matrix:')
    rot = np.asarray([utils.R_from_relion(*x) for x in euler])
    log(rot[0])
    if args.labels is not None:
        labels = utils.load_pkl(args.labels)
        log(f'Read labels from {args.labels}')
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
