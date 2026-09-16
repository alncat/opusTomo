'''How much of its own motion each body keeps at its own centre, as a function of --blend-radius

multi_body_grid weights body k at a point x by

    w_k(x) = exp(-0.2 * sum_i ((x - com_k) . axes_k[i] / (radius_k[i] * blend_radius))^2)

and then normalises across bodies. At a body's own centre w_k is not 1: the neighbouring
gaussians still have appreciable value there, so the body picks up a mixture of its neighbours'
displacements instead of its own. A neighbour's rotation arrives at that distance as very nearly
a rigid translation, so what leaks in shifts the body rather than distorting it.

This prints the self weight at each body's own centre over a range of blend_radius, which is the
number to look at when choosing one.

    python analysis_scripts/blend_self_weight.py mask_params.pkl
'''

import argparse
import numpy as np
import torch


def weights_at(x, com, rad, axes, scale):
    '''Normalised blending weights at point x, for radii scaled by `scale`'''
    d = np.stack([(((x - com[k]) @ axes[k].T) / (rad[k] * scale)) ** 2
                  for k in range(len(com))]).sum(-1)
    w = np.exp(-0.2 * d) + 1e-5
    return w / w.sum()


def main(args):
    # no weights_only= here: torch 1.11, which this repo targets, does not accept it, and the
    # mask pkl is a plain dict of tensors either way
    mp = torch.load(args.masks, map_location='cpu')
    com = np.asarray(mp['com_bodies'], dtype=np.float64)
    rad = np.asarray(mp['radii_bodies'], dtype=np.float64)
    axes = np.asarray(mp['principal_axes'], dtype=np.float64)
    nb = com.shape[0]

    print(f'{args.masks}: {nb} bodies')
    print(f'centres (voxels): {np.round(com, 1).tolist()}')
    print(f'radii   (voxels): {np.round(rad, 1).tolist()}')
    sep = np.linalg.norm(com[:, None] - com[None, :], axis=-1)
    print('centre separation (voxels):')
    for k in range(nb):
        print('  ' + ' '.join(f'{sep[k, j]:7.1f}' for j in range(nb)))

    scales = [float(x) for x in args.scales.split(',')]
    head = ''.join(f'{"body " + str(k):>10s}' for k in range(nb))
    print(f'\nself weight at each body\'s own centre\n{"blend-radius":<14s}{head}')
    for sc in scales:
        row = [weights_at(com[k], com, rad, axes, sc)[k] for k in range(nb)]
        print(f'{sc:<14.2f}' + ''.join(f'{v:10.3f}' for v in row))
    print('\nwhat leaks in from the neighbours, i.e. 1 minus the above')
    print(f'{"blend-radius":<14s}{head}')
    for sc in scales:
        row = [1. - weights_at(com[k], com, rad, axes, sc)[k] for k in range(nb)]
        print(f'{sc:<14.2f}' + ''.join(f'{v:10.3f}' for v in row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('masks', help='mask parameters (.pkl) written by dsdsh prepare_multi')
    parser.add_argument('--scales', default='0.3,0.4,0.5,0.6,0.7,0.85,1.0,1.2,1.4',
                        help='comma separated blend-radius values to tabulate '
                             '(default: %(default)s)')
    main(parser.parse_args())
