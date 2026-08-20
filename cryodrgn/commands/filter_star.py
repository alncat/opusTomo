'''
Slice a star file: keep an index subset, or split it into one star per cluster label

Two modes, both preserving data_optics and every other block of a RELION 3.1 file:

  subset  filter_star in.star ind.pkl -o out.star
          `analyze --kpc` writes ind.filter.{epoch}.pkl listing the particles of the
          selected KMeans classes; this turns those indices into a star that can be fed
          straight back into training.

  split   filter_star in.star --labels labels.pkl --outdir DIR
          one star per KMeans class, as written by analyze into kmeans{K}/labels.pkl.
          Add the index file to restrict the split to the kpc-selected particles:
          filter_star in.star ind.pkl --labels labels.pkl --outdir DIR
'''

import argparse
import os
import pickle

import numpy as np

from cryodrgn import utils

log = utils.log

# the particle block of a RELION 3.1 star file; older files carry a single unnamed block
PARTICLE_BLOCKS = ('particles', 'images', '')


def add_args(parser):
    parser.add_argument('star', help='Input star file (the one used for training)')
    parser.add_argument('ind', nargs='?', help='Index file selecting the particles to keep '
                                               '(.pkl from analyze --kpc, or a .txt of integers). '
                                               'With --labels, restricts the split to these rows')
    parser.add_argument('-o', help='Output star file (.star). Required unless --labels is given')
    parser.add_argument('--block', help='Name of the data block to slice. Default: the first '
                                        'block that looks like a particle block')

    group = parser.add_argument_group('Split into one star per cluster label')
    group.add_argument('--labels', help='Cluster labels (kmeans{K}/labels.pkl), one per row of '
                                        'the input star')
    group.add_argument('--outdir', help='Directory to write the per-label star files into')
    group.add_argument('--prefix', default='pre', help='Output name prefix, so label i goes to '
                                                       '{prefix}{i}.star (default: %(default)s)')
    group.add_argument('--skip-label', help='Comma-separated labels to skip. analyze --kpc marks '
                                            'the particles it did not select with a label equal '
                                            'to its cluster count')
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
    '''Return the key of the block to slice out of a dict of data blocks'''
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


def read_star(in_star, block=None):
    '''Return (all data blocks, key of the particle block, that block's dataframe)'''
    import starfile  # the standalone starfile package, which round-trips 3.1 optics blocks

    data = starfile.read(in_star, always_dict=True)
    key = pick_block(data, block)
    return data, key, data[key]


def write_rows(data, key, df, rows, out_star):
    '''Write out_star with the particle block replaced by df.iloc[rows], other blocks intact'''
    import starfile

    out = dict(data)
    out[key] = df.iloc[rows].reset_index(drop=True)
    starfile.write(out, out_star)


def filter_star(in_star, ind, out_star, block=None):
    '''
    Slice the particle block of in_star by ind and write out_star.

    Every other block (data_optics, data_general, ...) is copied through unchanged, so the
    output stays a valid RELION 3.1 file. Returns (n_selected, n_total).
    '''
    data, key, df = read_star(in_star, block)
    n_total = len(df)
    if ind.size and (ind.min() < 0 or ind.max() >= n_total):
        raise ValueError(f'index file selects rows {ind.min()}..{ind.max()} but block '
                         f'{key or "data_"} of {in_star} has only {n_total} rows')
    if len(np.unique(ind)) != len(ind):
        log(f'WARNING: the index file repeats {len(ind) - len(np.unique(ind))} entries; '
            f'the output star will repeat those particles')
    write_rows(data, key, df, ind, out_star)
    return len(ind), n_total


def split_star(in_star, labels, outdir, ind=None, prefix='pre', skip=(), block=None):
    '''
    Write one star per distinct value of labels, as {outdir}/{prefix}{label}.star.

    labels holds one entry per row of the input star -- analyze writes it in the original
    stack numbering even for a --kpc run, where the unselected particles carry a sentinel
    label equal to the cluster count. Pass ind (or skip) to leave those out. Returns a
    list of (label, count, path).
    '''
    data, key, df = read_star(in_star, block)
    n_total = len(df)
    if len(labels) != n_total:
        raise ValueError(f'labels has {len(labels)} entries but block {key or "data_"} of '
                         f'{in_star} has {n_total} rows -- the labels written by analyze are in '
                         f'the ORIGINAL stack numbering, so pass the star used for training '
                         f'rather than an already-filtered one')
    keep = np.ones(n_total, dtype=bool)
    if ind is not None:
        if ind.size and (ind.min() < 0 or ind.max() >= n_total):
            raise ValueError(f'index file selects rows {ind.min()}..{ind.max()} but the star has '
                             f'only {n_total} rows')
        keep[:] = False
        keep[ind] = True
    os.makedirs(outdir, exist_ok=True)
    written = []
    for label in np.unique(labels[keep]):
        if label in skip:
            log(f'skipping label {label} ({int(np.sum((labels == label) & keep))} particles)')
            continue
        rows = np.where((labels == label) & keep)[0]
        out_star = os.path.join(outdir, f'{prefix}{label}.star')
        write_rows(data, key, df, rows, out_star)
        log(f'wrote {len(rows)} particles in cluster {label} to {out_star}')
        written.append((int(label), len(rows), out_star))
    return written


def main(args):
    if args.labels is None and args.o is None:
        raise ValueError('give -o to write one subset star, or --labels/--outdir to split '
                         'the star by cluster label')
    if args.labels is not None and args.outdir is None:
        raise ValueError('--labels needs --outdir')
    ind = load_ind(args.ind) if args.ind is not None else None

    if args.labels is not None:
        labels = np.asarray(utils.load_pkl(args.labels)).reshape(-1)
        skip = [int(x) for x in args.skip_label.split(',')] if args.skip_label else []
        written = split_star(args.star, labels, args.outdir, ind=ind,
                             prefix=args.prefix, skip=skip, block=args.block)
        log(f'wrote {len(written)} cluster star files ({sum(n for _, n, _ in written)} particles '
            f'total) to {args.outdir}')

    if args.o is not None:
        assert args.o.endswith('.star'), 'Output must be a .star file'
        if ind is None:
            raise ValueError('-o needs an index file')
        n_sel, n_total = filter_star(args.star, ind, args.o, block=args.block)
        log(f'wrote {n_sel} of {n_total} particles to {args.o}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
