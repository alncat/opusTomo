'''
Quantify the relationship between the composition and conformation latent spaces

Reports how much of either latent code is recoverable from the other, and whether clustering
in one code carries information about the other.  Both are scored against a permutation null.

Requires a model trained with a conformation latent space, i.e. a z.N.pkl containing both
'mu' (composition) and 'multi_mu' (conformation).

Example:
    dsd disentangle /path/to/workdir 39 --kc 10 --kf 8
'''

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cryodrgn import disentangle
from cryodrgn import utils

log = utils.log


def add_args(parser):
    parser.add_argument('workdir', type=os.path.abspath,
                        help='Directory with OPUS-ET training results')
    parser.add_argument('epoch', type=int,
                        help='Epoch number N to analyze, corresponding to z.N.pkl')
    parser.add_argument('-o', '--outdir',
                        help='Output directory (default: [workdir]/disentangle.[epoch])')
    parser.add_argument('--kc', type=int, default=10,
                        help='Number of composition classes for K-means (default: %(default)s)')
    parser.add_argument('--kf', type=int, default=8,
                        help='Number of conformation classes for K-means (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for clustering, splitting and nulls (default: %(default)s)')
    parser.add_argument('--permutations', type=int, default=disentangle.DEFAULT_PERMUTATIONS,
                        help='Permutations for the clustering null (default: %(default)s)')
    parser.add_argument('--max-particles', type=int, default=disentangle.MAX_REGRESSION_PARTICLES,
                        help='Subsample this many particles for the regressions (default: %(default)s)')

    group = parser.add_argument_group('Rigid-body readout for one conformation class')
    group.add_argument('--class', dest='klass', type=int,
                       help='Conformation class to decode into rigid-body motion. Choose from '
                            'the ranking printed by the diagnostics.')
    group.add_argument('--flip', action='store_true',
                       help='Report rotation axes in the z-flipped frame, matching volumes '
                            'written by eval_vol --flip and hence deposited models')
    group.add_argument('--volumes', action='store_true',
                       help='Also decode the 2x2 intervention volumes for --class: consensus, '
                            'composition-only, conformation-only and both. Requires a GPU.')
    group.add_argument('--Apix', type=float,
                       help='Pixel size for the decoded volumes (default: the training Apix '
                            'from config.pkl)')
    return parser


def _tonp(v):
    return v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)


def _load_any(path):
    '''Load a checkpoint or a config.

    z.N.pkl and weights.N.pkl are written by torch.save; config.pkl is a plain pickle.  Older
    torch also rejects weights_only.  Each loader is tried in turn rather than nested in
    except clauses, so a failure inside one handler cannot mask the remaining fallbacks.'''
    loaders = [
        lambda: torch.load(path, map_location='cpu', weights_only=False),
        lambda: torch.load(path, map_location='cpu'),
        lambda: utils.load_pkl(path),
    ]
    last = None
    for load in loaders:
        try:
            return load()
        except Exception as exc:
            last = exc
    raise RuntimeError(f'could not load {path}: {last}')


def load_codes(workdir, epoch):
    '''Return (composition, conformation) codes, or (composition, None) if the model has no
    conformation branch.'''
    path = os.path.join(workdir, f'z.{epoch}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    z = _load_any(path)
    if not isinstance(z, dict):
        return np.asarray(z, dtype=np.float64), None
    comp = _tonp(z['mu']).astype(np.float64)
    conf = _tonp(z['multi_mu']).astype(np.float64) if 'multi_mu' in z else None
    return comp, conf


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def plot(results, path):
    '''Two panels: predictive R^2 in both directions, and composition mixture per class.'''
    rec, cpl = results['recoverability'], results['coupling']
    kc, kf = results['kc'], results['kf']

    # font.weight carries the tick and legend text; axes.labelweight and axes.titleweight
    # cover the axis labels and titles.  Mathtext needs \mathbf separately.
    # svg.fonttype='none' keeps text as real <text> elements instead of <use> references to
    # glyph outlines, which is what makes the SVG editable after PowerPoint's Convert to Shape.
    # It means the viewer must have the font, so a common sans is requested ahead of the
    # matplotlib default.
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 15,
                         'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'axes.titlepad': 16,
                         'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={'wspace': 0.32})

    # Both panels report fractions of variance and of particles, so both are shown as
    # percentages rather than 0-1 fractions.
    ridge = [100 * max(0.0, rec['ridge_comp_to_conf']), 100 * max(0.0, rec['ridge_conf_to_comp'])]
    forest = [100 * max(0.0, rec['forest_comp_to_conf']),
              100 * max(0.0, rec['forest_conf_to_comp'])]
    x, w = np.arange(2), 0.36
    ax[0].bar(x - w / 2, ridge, w, color='#a6cee3', edgecolor='black', linewidth=0.8, label='Ridge')
    ax[0].bar(x + w / 2, forest, w, color='#1f78b4', edgecolor='black', linewidth=0.8,
              label='Random forest')
    ax[0].axhline(100 * rec['null'], ls='--', c='0.3', lw=1.4, label='Shuffled null')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(['Conformation\nfrom composition', 'Composition\nfrom conformation'])
    ax[0].set_ylim(-1, max(10, max(ridge + forest) * 1.4))
    ax[0].set_ylabel('Held-out R² (%)')
    # Literal superscript rather than mathtext: matplotlib renders $...$ through its own
    # engine as glyph outlines, which stay uneditable in the SVG whatever svg.fonttype is.
    ax[0].set_title('Predictive R²')
    ax[0].legend(frameon=False, loc='upper right')

    cmap = plt.get_cmap('tab10')
    bottom = np.zeros(kf)
    mixtures = 100 * np.nan_to_num(np.asarray(cpl['mixtures']))
    for c in range(kc):
        ax[1].bar(range(kf), mixtures[:, c], bottom=bottom, width=0.86,
                  color=cmap(c % 10), edgecolor='white', linewidth=0.3)
        bottom += mixtures[:, c]
    ax[1].set_xlim(-0.6, kf - 0.4)
    ax[1].set_ylim(0, 100)
    ax[1].set_xticks(range(kf))
    ax[1].set_xlabel('Conformation class')
    ax[1].set_ylabel('Composition-class Fraction (%)')
    ax[1].set_title('Clustering stability')
    ax[1].legend(handles=[Patch(facecolor=cmap(c % 10), edgecolor='white', label=str(c))
                          for c in reversed(range(kc))],
                 title='Composition\nclass', loc='center left', bbox_to_anchor=(1.01, 0.5),
                 frameon=False, fontsize=12, title_fontsize=13, labelspacing=0.3, handlelength=1.1)

    # No tight_layout: the composition-class legend sits outside its axes, which tight_layout
    # cannot account for.  bbox_inches='tight' at save time handles it correctly.
    # PDF and SVG are both fully vector; the PNG is for quick viewing.
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.svg', bbox_inches='tight')
    plt.close(fig)


def report(results):
    rec, cpl = results['recoverability'], results['coupling']
    log(f"{results['n_particles']} particles, composition {results['dim_composition']}-D, "
        f"conformation {results['dim_conformation']}-D")
    log('')
    log(f"Predictive R^2 (held-out, {rec['n_particles']} particles, "
        f"shuffled null {rec['null']:.4f}):")
    log(f"  conformation from composition   ridge {rec['ridge_comp_to_conf']:7.4f}"
        f"   forest {rec['forest_comp_to_conf']:7.4f}")
    log(f"  composition from conformation   ridge {rec['ridge_conf_to_comp']:7.4f}"
        f"   forest {rec['forest_conf_to_comp']:7.4f}")
    log('')
    log(f"Clustering: AMI {cpl['ami']:.4f}   ARI {cpl['ari']:.4f}   "
        f"mean TV {np.mean(cpl['tv']):.4f} (null {cpl['tv_null_mean']:.4f})")
    log('')
    log('Conformation classes ranked by composition coupling:')
    log(f"  {'class':>5} {'size':>7} {'TV':>7} {'x null':>7}  dominant composition class")
    for row in results['ranking']:
        log(f"  {row['conformation_class']:>5} {row['size']:>7} {row['tv']:>7.3f} "
            f"{row['tv_over_null']:>7.1f}  class {row['dominant_composition_class']:>2} "
            f"at {100 * row['dominant_fraction']:5.1f}% "
            f"(global {100 * row['dominant_fraction_global']:4.1f}%, "
            f"{row['enrichment']:.1f}x)")


def report_rigid_body(rb, klass):
    log('')
    log(f'Rigid-body readout for conformation class {klass} '
        f"({rb['num_bodies']} bodies"
        + (', z-flipped frame)' if rb['flip'] else ', model frame)'))
    for row in rb['per_body']:
        log(f"  body {row['body']}   global {row['global_angle_deg']:6.2f} deg"
            f"   class {row['class_angle_deg']:6.2f} deg")
    if 'note' in rb:
        log(f"  {rb['note']}")
        return
    com = rb.get('com_bodies')
    ref = (f"body0->body1 ({np.round(com[0], 1).tolist()} -> {np.round(com[1], 1).tolist()})"
           if com is not None and len(com) >= 2 else 'body0->body1')
    for key, label in [('inter_body_global', 'inter-body, global mean'),
                       ('inter_body_class', 'inter-body, this class'),
                       ('class_vs_global', 'class relative to global')]:
        d = rb[key]
        line = f"  {label:<26} {d['angle_deg']:6.2f} deg"
        if d.get('axis') is not None:
            axis = np.round(d['axis'], 3)
            line += f"   axis [{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}]"
        else:
            line += f"   {d.get('note', '')}"
        log(line)
        if d.get('axis') is not None and 'angle_to_body_axis_deg' in d:
            log(f"  {'':<26} {d['angle_to_body_axis_deg']:.1f} deg from {ref}, {d['sense']}")


def rigid_body_readout(args, conf, labels_conf, outdir):
    from cryodrgn import rigid_body

    if args.klass not in np.unique(labels_conf):
        raise ValueError(f'conformation class {args.klass} is empty or out of range '
                         f'(0..{args.kf - 1})')
    path = os.path.join(args.workdir, f'weights.{args.epoch}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found, needed to replay the conformation decoder')
    ckpt = _load_any(path)
    state = ckpt['decoder_state_dict'] if 'decoder_state_dict' in ckpt else ckpt

    rb = rigid_body.analyse(conf[labels_conf == args.klass].mean(0), conf.mean(0), state,
                            flip=args.flip)
    rb['conformation_class'] = int(args.klass)
    rb['class_size'] = int((labels_conf == args.klass).sum())
    report_rigid_body(rb, args.klass)

    subdir = os.path.join(outdir, f'class_{args.klass}')
    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, 'rigid_body.json'), 'w') as f:
        json.dump(_jsonable(rb), f, indent=2)
    log(f'  wrote {os.path.join(subdir, "rigid_body.json")}')


# The four rows of the intervention z-file.  Holding one code at its global mean while the
# other takes its class mean isolates what each branch contributes.
INTERVENTION = [('consensus', 'global', 'global'),
                ('composition', 'class', 'global'),
                ('conformation', 'global', 'class'),
                ('both', 'class', 'class')]


def intervention_volumes(args, comp, conf, labels_conf, outdir):
    '''Decode the 2x2 and measure which branch moved the density.'''
    import argparse as _argparse
    from cryodrgn import mrc
    from cryodrgn.commands import eval_vol

    weights = os.path.join(args.workdir, f'weights.{args.epoch}.pkl')
    config = os.path.join(args.workdir, 'config.pkl')
    for p in (weights, config):
        if not os.path.exists(p):
            raise FileNotFoundError(f'{p} not found, needed to decode volumes')
    apix = args.Apix
    if apix is None:
        apix = _load_any(config).get('model_args', {}).get('Apix')
        if apix is None:
            raise ValueError('no Apix in config.pkl; pass --Apix explicitly')
        log(f'Using the training pixel size, {apix} A/pix. Override with --Apix.')

    sel = labels_conf == args.klass
    means = {'global': (comp.mean(0), conf.mean(0)),
             'class': (comp[sel].mean(0), conf[sel].mean(0))}
    rows = np.array([np.concatenate([means[c][0], means[f][1]]) for _, c, f in INTERVENTION])

    subdir = os.path.join(outdir, f'class_{args.klass}')
    os.makedirs(subdir, exist_ok=True)
    zfile = os.path.join(subdir, 'intervention.txt')
    np.savetxt(zfile, rows)
    log(f'  wrote {zfile} ({rows.shape[0]} rows of {rows.shape[1]}-D enc = [composition, conformation])')

    # Build eval_vol's namespace from its own parser so its defaults cannot drift from ours.
    parser = eval_vol.add_args(_argparse.ArgumentParser())
    argv = ['--load', weights, '-c', config, '-o', subdir, '--zfile', zfile,
            '--Apix', str(apix), '--num-bodies', '0']
    if args.flip:
        argv.append('--flip')
    ev = parser.parse_args(argv)
    eval_vol.check_inputs(ev)
    eval_vol.main(ev)

    vols = {}
    for i, (name, _, _) in enumerate(INTERVENTION):
        src = os.path.join(subdir, f'reference{i}.mrc')
        dst = os.path.join(subdir, f'{name}.mrc')
        os.replace(src, dst)
        vols[name] = np.asarray(mrc.parse_mrc(dst)[0], dtype=np.float64)

    d_comp = vols['composition'] - vols['consensus']
    d_conf = vols['conformation'] - vols['consensus']
    d_both = vols['both'] - vols['consensus']
    mrc.write(os.path.join(subdir, 'dcomp.mrc'), d_comp.astype(np.float32), Apix=apix)

    def corr(a, b):
        '''None when one side has no variance, which is the expected outcome for the
        conformation term without --masks: correlation against a constant is undefined.'''
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    def fmt(v):
        return 'n/a (no change to correlate against)' if v is None else f'{v:+.4f}'
    summary = {
        'norm_composition_only': float(np.linalg.norm(d_comp)),
        'norm_conformation_only': float(np.linalg.norm(d_conf)),
        'norm_both': float(np.linalg.norm(d_both)),
        'corr_both_with_composition': corr(d_both, d_comp),
        'corr_both_with_conformation': corr(d_both, d_conf),
        'Apix': float(apix),
    }
    log('')
    log('  Change in density relative to the consensus volume:')
    log(f"    composition code only    ||delta|| = {summary['norm_composition_only']:.4f}")
    log(f"    conformation code only   ||delta|| = {summary['norm_conformation_only']:.4f}")
    log(f"    both                     ||delta|| = {summary['norm_both']:.4f}")
    log(f"    correlation of both-minus-consensus with the composition change  "
        f"{fmt(summary['corr_both_with_composition'])}")
    log(f"    correlation of both-minus-consensus with the conformation change "
        f"{fmt(summary['corr_both_with_conformation'])}")
    log('    A near-zero conformation term is the expected result, not a failure: without')
    log('    --masks the body transforms are not applied, so the conformation code drives')
    log('    motion that a statically saved volume cannot show.')

    with open(os.path.join(subdir, 'intervention.json'), 'w') as f:
        json.dump(_jsonable(summary), f, indent=2)
    log(f"  wrote {os.path.join(subdir, 'intervention.json')} and dcomp.mrc")


def main(args):
    comp, conf = load_codes(args.workdir, args.epoch)
    if conf is None:
        log(f'z.{args.epoch}.pkl contains no conformation code (no "multi_mu" entry).')
        log('This model was trained with the composition latent space only, so there is no '
            'second space to compare against.  Nothing to do.')
        return

    outdir = args.outdir or os.path.join(args.workdir, f'disentangle.{args.epoch}')
    os.makedirs(outdir, exist_ok=True)
    log(f'Writing to {outdir}')

    results = disentangle.run_diagnostics(comp, conf, kc=args.kc, kf=args.kf, seed=args.seed,
                                          n_permutations=args.permutations,
                                          max_particles=args.max_particles)
    report(results)

    labels_comp = results.pop('labels_composition')
    labels_conf = results.pop('labels_conformation')
    np.savetxt(os.path.join(outdir, 'labels_composition.txt'), labels_comp, fmt='%d')
    np.savetxt(os.path.join(outdir, 'labels_conformation.txt'), labels_conf, fmt='%d')

    with open(os.path.join(outdir, 'disentangle.json'), 'w') as f:
        json.dump(_jsonable(results), f, indent=2)
    plot(results, os.path.join(outdir, 'disentangle'))
    log('')
    log(f"Wrote disentangle.json, disentangle.png/pdf/svg and the two label files to {outdir}")

    if args.klass is not None:
        rigid_body_readout(args, conf, labels_conf, outdir)
        if args.volumes:
            intervention_volumes(args, comp, conf, labels_conf, outdir)
    elif args.volumes:
        log('')
        log('--volumes needs --class: choose a conformation class from the ranking above.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
