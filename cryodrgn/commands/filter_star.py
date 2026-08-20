'''
Write a star file holding the subset of particles selected by an index file

Typical use: `analyze --kpc` writes ind.filter.{epoch}.pkl listing the particles of the
selected KMeans classes; this command turns those indices into a star file that can be
fed straight back into training.
'''

import argparse
import pickle

import numpy as np

from cryodrgn import utils

log = utils.log

# the particle block of a RELION 3.1 star file; older files carry a single unnamed block
PARTICLE_BLOCKS = ('particles', 'images', '')


def add_args(parser):
    parser.add_argument('star', help='Input star file (the one used for training)')
    parser.add_argument('ind', help='Index file selecting the particles to keep '
                                    '(.pkl from analyze --kpc, or a .txt of integers)')
    parser.add_argument('-o', required=True, help='Output star file (.star)')
    parser.add_argument('--block', help='Name of the data block to filter. Default: the first '
                                        'block that looks like a particle block')
    return parser


def load_ind(fname):
    '''Read an index array from a .pkl (analyze --kpc) or a .txt of integers'''
    if fname.endswith('.pkl'):
        with open(fname, 'rb') as f:
            ind = pickle.load(f)
    else:
        ind = np.loadtxt(fname, dtype=np.int64)
    ind = np.atleast_1d(np.asarray(ind)).astype(np.int64)
    assert ind.ndim == 1, f'index file {fname} must hold a 1D array, got shape {ind.shape}'
    return ind


def pick_block(blocks, block=None):
    '''Return the key of the block to filter out of a dict of data blocks'''
    if block is not None:
        if block not in blocks:
            raise ValueError(f'no data block named {block!r}; the file has {list(blocks)}')
        return block
    for name in PARTICLE_BLOCKS:
        if name in blocks:
            return name
    # fall back to the largest block, which for a RELION 3.1 file is the particle table
    # (data_optics holds one row per optics group)
    return max(blocks, key=lambda k: len(blocks[k]))


def filter_star(in_star, ind, out_star, block=None):
    '''
    Slice the particle block of in_star by ind and write out_star.

    Every other block (data_optics, data_general, ...) is copied through unchanged, so the
    output stays a valid RELION 3.1 file. Returns (n_selected, n_total).
    '''
    import starfile  # the standalone starfile package, which round-trips 3.1 optics blocks

    data = starfile.read(in_star, always_dict=True)
    key = pick_block(data, block)
    df = data[key]
    n_total = len(df)
    if ind.size and (ind.min() < 0 or ind.max() >= n_total):
        raise ValueError(f'index file selects rows {ind.min()}..{ind.max()} but block '
                         f'{key or "data_"} of {in_star} has only {n_total} rows')
    if len(np.unique(ind)) != len(ind):
        log(f'WARNING: the index file repeats {len(ind) - len(np.unique(ind))} entries; '
            f'the output star will repeat those particles')
    data[key] = df.iloc[ind].reset_index(drop=True)
    starfile.write(data, out_star)
    return len(ind), n_total


def main(args):
    assert args.o.endswith('.star'), 'Output must be a .star file'
    ind = load_ind(args.ind)
    n_sel, n_total = filter_star(args.star, ind, args.o, block=args.block)
    log(f'wrote {n_sel} of {n_total} particles to {args.o}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
