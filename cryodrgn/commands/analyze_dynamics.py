'''
Rank volume series (e.g. PC traversals) by how much usable rigid-body motion they contain

Before building mask_params from a volume series with `parse_multi_pose_star --volumes`,
you have to pick a series. This reports, per body per series, whether the centres of mass
actually trace a rotation -- and whether the two bodies swing against each other (a hinge)
or merely drift together (a global motion, useless for multi-body refinement).
'''

import argparse
import os
import torch
import torch.nn.functional as F

from cryodrgn import utils
from cryodrgn import starfile
from cryodrgn import dataset
from cryodrgn import dynamics

log = utils.log


def add_args(parser):
    parser.add_argument('volumes', nargs='+', help='One or more directories of referenceN.mrc series (e.g. analyze.N/pc*)')
    parser.add_argument('--masks', required=True, help='RELION multi-body star file defining the body masks')
    parser.add_argument('--num-volumes', type=int, default=10, help='Number of referenceN.mrc per directory (default: %(default)s)')
    parser.add_argument('-o', '--outfile', help='Optional path to write the summary table as text')
    parser.add_argument('--verbose', action='store_true', help='Print per-body detail as well as the summary table')
    return parser


def load_bodies(masks_star):
    '''Body masks, their centres of mass and their parent indices, from the star file.'''
    s = starfile.Starfile.load(masks_star)
    prefix = os.path.dirname(masks_star)
    masks, coms, parents = [], [], []
    for b_i in range(len(s.df)):
        path = os.path.join(prefix, s.df['_rlnBodyMaskName'][b_i])
        assert os.path.isfile(path), f"missing body mask {path}"
        vol = dataset.VolData(path)
        masks.append(vol.get())
        coms.append(dynamics.center_of_mass(vol.get())[0])
        parents.append(int(s.df['_rlnBodyRotateRelativeTo'][b_i]) - 1)
    return torch.stack(masks, dim=0), coms, parents


def analyze_series(voldir, masks, coms, parents, num_volumes):
    '''Per-body motion statistics for one volume series.'''
    vols = []
    for i in range(num_volumes):
        path = os.path.join(voldir, f"reference{i}.mrc")
        if not os.path.isfile(path):
            return None
        vols.append(dataset.VolData(path).get())

    scale = masks.shape[-1]/vols[0].shape[-1]
    m = F.interpolate(masks.unsqueeze(0), vols[0].shape, mode='trilinear').squeeze(0)

    bodies = []
    for b_i in range(m.shape[0]):
        traj = dynamics.com_trajectory(vols, m[b_i], scale=scale)
        st = dynamics.trajectory_stats(traj)
        fit = dynamics.fit_rotation_axis(traj, pivot=coms[parents[b_i]])
        bodies.append(dict(body=b_i, parent=parents[b_i], **st, **fit))
    return bodies


def verdict(bodies):
    '''One-line judgement for a series, plus a score used only for ordering.'''
    coherence = max(b['coherence'] for b in bodies)
    swept = [abs(b['swept_angle']) for b in bodies]
    dispersion = max(b['dispersion'] for b in bodies)
    out_of_plane = max(b['out_of_plane'] for b in bodies)
    asym = max(swept)/max(min(swept), 1e-6)

    pair_angle, pair_kind = (None, None)
    if len(bodies) == 2:
        pair_angle, pair_kind = dynamics.interpret_pair(bodies[0]['axis'], bodies[1]['axis'])

    if coherence > 1.5 or min(swept) < 0.5:
        text = 'noise-dominated'
        score = 0.0
    elif pair_kind == 'global drift (parallel)':
        text = 'whole complex drifting, not internal motion'
        score = 0.1
    elif pair_kind == 'uncorrelated':
        text = 'weak / uncorrelated'
        score = 0.2
    elif dispersion > 15.0:
        text = 'moves, but the axis is not consistent'
        score = 0.3
    elif asym > 2.0:
        text = 'hinge-like but the two bodies disagree'
        score = 0.5
    else:
        text = 'CLEAN HINGE'
        # a bigger sweep is better, but only in proportion to how much the axis can be
        # trusted: discount by the pair-to-pair spread and by how far the two bodies are
        # from swinging against each other. Ranking on amplitude alone puts a large but
        # ill-determined motion above a smaller, clean one.
        trust = max(1.0 - dispersion/15.0, 0.0)*(pair_angle/180.0 if pair_angle else 1.0)
        score = 1.0 + min(swept)*max(trust, 0.05)
    return text, score, pair_angle, asym, coherence, dispersion, out_of_plane


def main(args):
    masks, coms, parents = load_bodies(args.masks)
    log(f"{masks.shape[0]} bodies, parents (0-based): {parents}")

    rows = []
    for voldir in args.volumes:
        name = os.path.basename(os.path.normpath(voldir))
        bodies = analyze_series(voldir, masks, coms, parents, args.num_volumes)
        if bodies is None:
            log(f"{name}: fewer than {args.num_volumes} referenceN.mrc, skipping")
            continue
        text, score, pair_angle, asym, coherence, dispersion, out_of_plane = verdict(bodies)
        rows.append(dict(name=name, verdict=text, score=score, pair_angle=pair_angle,
                         asym=asym, coherence=coherence, dispersion=dispersion,
                         out_of_plane=out_of_plane, bodies=bodies))
        if args.verbose:
            for b in bodies:
                log("  {} body {} (parent {}): swept {:+.2f} deg, moved {:.2f} px, "
                    "coherence {:.2f}, dispersion {:.1f} deg, out-of-plane {:.2f} px, axis {}".format(
                        name, b['body'], b['parent'], b['swept_angle'], b['displacement'],
                        b['coherence'], b['dispersion'], b['out_of_plane'],
                        [round(float(x), 3) for x in b['axis']]))

    if not rows:
        log("no usable volume series found")
        return

    rows.sort(key=lambda r: -r['score'])
    header = ("{:<10} {:>10} {:>9} {:>6} {:>11} {:>12} {:>10}  {}".format(
        "series", "swept(min)", "coherence", "asym", "dispersion", "out-of-plane", "axis-pair", "verdict"))
    lines = [header, "-"*len(header)]
    for r in rows:
        swept_min = min(abs(b['swept_angle']) for b in r['bodies'])
        pair = "{:.0f} deg".format(r['pair_angle']) if r['pair_angle'] is not None else "n/a"
        lines.append("{:<10} {:>10.2f} {:>9.2f} {:>6.1f} {:>10.1f}d {:>10.2f}px {:>10}  {}".format(
            r['name'], swept_min, r['coherence'], r['asym'], r['dispersion'],
            r['out_of_plane'], pair, r['verdict']))

    lines.append("")
    lines.append("swept(min)  smaller of the two bodies' swept angles, in degrees")
    lines.append("coherence   path/displacement of the centre of mass; ~1 is a clean arc, large means it jitters")
    lines.append("asym        ratio of the two bodies' swept angles; a real hinge moves both by similar amounts")
    lines.append("dispersion  spread of the axis estimated from different frame pairs; small means it is trustworthy")
    lines.append("out-of-plane  how far the com strays from the fitted rotation plane; large is not a single axis")
    lines.append("axis-pair   angle between the two bodies' axes; ~180 deg hinge, ~0 deg whole-complex drift")

    out = "\n".join(lines)
    print(out)
    if args.outfile:
        with open(args.outfile, 'w') as f:
            f.write(out + "\n")
        log(f"wrote {args.outfile}")

    best = rows[0]
    if best['score'] >= 1.0:
        log(f"recommended series for `parse_multi_pose_star --volumes`: {best['name']}")
    else:
        log("no series looks like a clean hinge; check the body definitions or use a "
            "traversal with larger motion before building mask_params")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
