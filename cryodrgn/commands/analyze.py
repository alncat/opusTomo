'''
Visualize latent space and generate volumes
'''

import argparse
import numpy as np
import sys, os
import pickle
import json
import shutil
import healpy as hp
import inspect
from datetime import datetime as dt
import scipy
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt
# every figure below is saved at this resolution, cropped to its contents so axis
# labels are never clipped at the image edge
plt.rcParams.update({'savefig.dpi': 200, 'savefig.bbox': 'tight'})
import seaborn as sns
import torch

import cryodrgn
from cryodrgn import analysis
from cryodrgn import utils
from cryodrgn.pose import PoseTracker
from cryodrgn.commands.filter_star import filter_star

log = utils.log

def add_args(parser):
    parser.add_argument('workdir', type=os.path.abspath, help='Directory with cryoDRGN results')
    parser.add_argument('epoch', type=int, help='Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)')
    parser.add_argument('--device', type=int, help='Optionally specify CUDA device')
    parser.add_argument('-o','--outdir', help='Output directory for analysis results (default: [workdir]/analyze.[epoch])')
    parser.add_argument('--skip-vol', action='store_true', help='Skip generation of volumes')
    parser.add_argument('--skip-umap', action='store_true', help='Skip running UMAP')
    parser.add_argument('--vanilla', action='store_true', help='Load an OPUS-ET (real-space) result: read z from z.N.pkl and poses from --pose, rather than a cryoDRGN z.pkl')
    parser.add_argument('--D', type=int, help='Box size of the images (pixels), used to load the pose tracker')
    parser.add_argument('--pose', help='Pose pickle (pose.N.pkl) to load particle groups from (used with --vanilla)')

    group = parser.add_argument_group('Extra arguments for volume generation')
    group.add_argument('--Apix', type=float, default=1, help='Pixel size to add to .mrc header (default: %(default)s A/pix)')
    group.add_argument('--flip', action='store_true', help='Flip handedness of output volume')
    group.add_argument('-d','--downsample', type=int, help='Downsample volumes to this box size (pixels)')
    group.add_argument('--pc', type=int, default=2, help='Number of principal component traversals to generate (default: %(default)s)')
    group.add_argument('--ksample', type=int, default=20, help='Number of kmeans samples to generate (default: %(default)s)')
    group.add_argument('--kpc', type=str, default=None, help='Re-cluster a subset of an existing KMeans run, as K:classes (e.g. 20:2-5,8): K is the cluster count of the previous analyze, classes selects clusters (x-y is an inclusive range) (default: %(default)s)')
    group.add_argument('--joint', action='store_true', help='Also write centers_joint.txt: composition centers with the matching conformation (multi_mu) latents concatenated (needs a multi-body result)')
    return parser

def analyze_z1(z, outdir, vg):
    '''Plotting and volume generation for 1D z'''
    assert z.shape[1] == 1
    z = z.reshape(-1)
    N = len(z)

    plt.figure()
    plt.scatter(np.arange(N), z, alpha=.1, s=2, rasterized=True)
    plt.xlabel('particle')
    plt.ylabel('z')
    plt.savefig(f'{outdir}/z.png')
    plt.close()

    plt.figure()
    sns.histplot(z, kde=True)
    plt.xlabel('z')
    plt.savefig(f'{outdir}/z_hist.png')
    plt.close()

    ztraj = np.linspace(*np.percentile(z,(5,95)), 10) # or np.percentile(z, np.linspace(5,95,10)) ?
    vg.gen_volumes(outdir, ztraj)

def save_pca(outdir, pc, pca):
    '''Deposit the PCA projection and fitted model so downstream custom analysis
    (arbitrary-PC particle selection, custom traversals, seeing which latent dims
    drive each PC) needs no re-fitting -- run_pca otherwise discards both.
      pc.pkl   : (N, zdim) projection of z onto the principal components
      pca.pkl  : fitted sklearn PCA (components_, mean_, explained_variance_ratio_);
                 drop-in for get_pc_traj / pca.transform / pca.inverse_transform
      explained_variance_ratio.txt : human-readable per-PC variance share'''
    os.makedirs(outdir, exist_ok=True)
    utils.save_pkl(pc, f'{outdir}/pc.pkl')
    utils.save_pkl(pca, f'{outdir}/pca.pkl')
    np.savetxt(f'{outdir}/explained_variance_ratio.txt', pca.explained_variance_ratio_)

def analyze_zN(z, outdir, vg, pose_tilt, skip_umap=False, num_pcs=2, num_ksamples=20, joint=False, multi_z=None):
    '''pose_tilt: per-particle Euler tilt angle in degrees (0-180), or None. Used to color
    umap.png, so a latent direction that follows particle orientation (and thus the
    missing wedge) shows up as a color gradient.'''
    zdim = z.shape[1]
    if joint and multi_z is None:
        raise ValueError("joint=True needs multi_z: centers_joint is built by concatenating the "
                         "conformation latents onto the composition centers")

    # Principal component analysis
    print(z[:4, :])
    log('Perfoming principal component analysis...')
    pc, pca = analysis.run_pca(z)
    log(f'PCA transformed data shape {pc.shape}')
    save_pca(outdir, pc, pca)
    plot_pca_variance(pca, f'{outdir}/pca_variance.png')
    # axis labels carry each PC's share of the variance
    pc_labels = [f'PC{i+1} ({100*r:.1f}%)' for i, r in enumerate(pca.explained_variance_ratio_)]
    print(pc[:4, :])
    log('Generating volumes...')
    pc_trajs = []
    for i in range(num_pcs):
        start, end = np.percentile(pc[:,i],(1.0,99.0))
        log(f'traversing pc {i} from {start} to {end}')
        z_pc = analysis.get_pc_traj(pca, z.shape[1], 10, i+1, start, end)
        pc_trajs.append(z_pc)
        if not os.path.exists(f'{outdir}/pc{i+1}'):
            os.mkdir(f'{outdir}/pc{i+1}')
        vg.gen_volumes(f'{outdir}/pc{i+1}', z_pc)
        np.savetxt(f'{outdir}/pc{i+1}/z_pc.txt', z_pc)

    # kmeans clustering in pca domain
    #log('K-means clustering in PCA...')
    #K = num_ksamples
    #z_pc = np.concatenate([np.zeros_like(pc[:, :1]), pc[:, 1:]], axis=-1)
    #kmeans_labels, centers = analysis.cluster_kmeans(z_pc, K)
    #centers, centers_ind_pc = analysis.get_nearest_point(z_pc, centers)
    #centers = centers_pc = pca.inverse_transform(centers)
    #if not os.path.exists(f'{outdir}/kmeansPC{K}'):
    #    os.mkdir(f'{outdir}/kmeansPC{K}')
    #utils.save_pkl(kmeans_labels, f'{outdir}/kmeansPC{K}/labels.pkl')
    #utils.save_pkl(centers, f'{outdir}/kmeansPC{K}/centers.pkl')
    #np.savetxt(f'{outdir}/kmeansPC{K}/centers.txt', centers)
    #np.savetxt(f'{outdir}/kmeansPC{K}/centers_ind.txt', centers_ind_pc, fmt='%d')
    #log('Generating volumes...')
    #vg.gen_volumes(f'{outdir}/kmeansPC{K}', centers)
    #log(f'{np.bincount(kmeans_labels)}')

    # kmeans clustering
    log('K-means clustering...')
    K = num_ksamples
    kmeans_labels, centers = analysis.cluster_kmeans(z, K)
    _, centers_ind = analysis.get_nearest_point(z, centers)
    #concat the corresponding conformation latent codes to centers
    if multi_z is not None:
        assert z.shape[0] == multi_z.shape[0], f"z shape {z.shape} must be equal to multi_z shape {multi_z.shape} at first dimension"
        multi_z_center = multi_z[centers_ind]
        centers_joint = np.concatenate([centers, multi_z_center], axis=-1)
    if not os.path.exists(f'{outdir}/kmeans{K}') or not skip_umap:
        if not os.path.exists(f'{outdir}/kmeans{K}'):
            os.mkdir(f'{outdir}/kmeans{K}')
        utils.save_pkl(kmeans_labels, f'{outdir}/kmeans{K}/labels.pkl')
        utils.save_pkl(centers, f'{outdir}/kmeans{K}/centers.pkl')
        np.savetxt(f'{outdir}/kmeans{K}/centers.txt', centers)
        np.savetxt(f'{outdir}/kmeans{K}/centers_ind.txt', centers_ind, fmt='%d')
        if joint:
            np.savetxt(f'{outdir}/kmeans{K}/centers_joint.txt', centers_joint)
    elif skip_umap:
        log('loading kmeans clustering')
        kmeans_labels = utils.load_pkl(f'{outdir}/kmeans{K}/labels.pkl')
        centers = utils.load_pkl(f'{outdir}/kmeans{K}/centers.pkl')
        centers_ind = np.loadtxt(f'{outdir}/kmeans{K}/centers_ind.txt', dtype=np.int64)
        if joint:
            np.savetxt(f'{outdir}/kmeans{K}/centers_joint.txt', centers_joint)
    log('Generating volumes...')
    vg.gen_volumes(f'{outdir}/kmeans{K}', centers)
    log(f'{np.bincount(kmeans_labels)}')
    plot_cluster_sizes(kmeans_labels, K, f'{outdir}/kmeans{K}/cluster_sizes.png')

    # UMAP -- slow step
    if zdim > 2:
        log('Running UMAP...')
        umap_file = f'{outdir}/umap.pkl'
        if os.path.exists(umap_file) and skip_umap:
            umap_emb = utils.load_pkl(umap_file)
            skip_umap = False
        else:
            umap_emb = analysis.run_umap(z)
            utils.save_pkl(umap_emb, umap_file)
            skip_umap = False

    #log('Running TSNE...')
    #tsne_file = f'{outdir}/tsne.pkl'
    #if os.path.exists(tsne_file):
    #    tsne_emb = utils.load_pkl(tsne_file)
    #else:
    #    tsne_emb = analysis.run_tsne(z)
    #    utils.save_pkl(tsne_emb, tsne_file)

    # Make some plots. Each savefig is followed by plt.close() (the current figure
    # is always the one just saved), so figures do not pile up across the run.
    log('Generating plots...')
    g = sns.jointplot(x=pc[:,0], y=pc[:,1], alpha=.1, s=2)
    g.set_axis_labels(pc_labels[0], pc_labels[1])
    plt.savefig(f'{outdir}/z_pca.png')
    plt.close()

    g = sns.jointplot(x=pc[:,0], y=pc[:,1], kind='hex')
    g.set_axis_labels(pc_labels[0], pc_labels[1])
    plt.savefig(f'{outdir}/z_pca_hexbin.png')
    plt.close()

    if zdim > 2 and not skip_umap:
        # square plotting box with the same span on both axes
        xmin, xmax = np.min(umap_emb[:, 0]), np.max(umap_emb[:, 0])
        ymin, ymax = np.min(umap_emb[:, 1]), np.max(umap_emb[:, 1])
        span = max(xmax - xmin, ymax - ymin)
        xmax, ymax = xmin + span, ymin + span
        if pose_tilt is not None:
            # fixed 0-180 range so runs can be compared by color
            analysis.scatter_color(umap_emb[:,0], umap_emb[:,1], pose_tilt, cmap='viridis',
                                   s=2, alpha=.3, vmin=0, vmax=180, label='Euler tilt (deg)')
        else:
            plt.subplots()
            plt.scatter(umap_emb[:,0], umap_emb[:,1], s=2, alpha=.3, rasterized=True)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig(f'{outdir}/umap.png')
        plt.close()

        g = sns.jointplot(x=umap_emb[:,0], y=umap_emb[:,1], kind='hex')
        g.set_axis_labels('UMAP1','UMAP2')
        plt.savefig(f'{outdir}/umap_hexbin.png')
        plt.close()

    analysis.scatter_annotate(pc[:,0], pc[:,1], centers_ind=centers_ind, annotate=True)
    plt.xlabel(pc_labels[0])
    plt.ylabel(pc_labels[1])
    plt.savefig(f'{outdir}/kmeans{K}/z_pca.png')
    plt.close()

    g = analysis.scatter_annotate_hex(pc[:,0], pc[:,1], centers_ind=centers_ind, annotate=True)
    g.set_axis_labels(pc_labels[0], pc_labels[1])
    plt.savefig(f'{outdir}/kmeans{K}/z_pca_hex.png')
    plt.close()

    if zdim > 2 and not skip_umap:
        analysis.scatter_annotate(umap_emb[:,0], umap_emb[:,1], centers_ind=centers_ind, annotate=True,
                                  xlim=(xmin, xmax), ylim=(ymin, ymax),
                                  alpha=.15, s=0.5)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig(f'{outdir}/kmeans{K}/umap.png')
        plt.close()

        # same view, each particle colored by the k-means cluster it belongs to
        analysis.plot_by_cluster(umap_emb[:,0], umap_emb[:,1], K, kmeans_labels,
                                 centers_ind=centers_ind, annotate=True, s=1, alpha=.3)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig(f'{outdir}/kmeans{K}/umap_clusters.png')
        plt.close()

        g = analysis.scatter_annotate_hex(umap_emb[:,0], umap_emb[:,1], centers_ind=centers_ind, annotate=True)
        g.set_axis_labels('UMAP1','UMAP2')
        plt.savefig(f'{outdir}/kmeans{K}/umap_hex.png')
        plt.close()

    for i in range(num_pcs):
        if zdim > 2 and not skip_umap:
            # clip the color range to the same 1-99 percentile span the PC traversal uses,
            # so a few outliers do not squeeze everyone else into one color
            vmin, vmax = np.percentile(pc[:,i], (1.0, 99.0))
            fig, ax = analysis.scatter_color(umap_emb[:,0], umap_emb[:,1], pc[:,i], label=pc_labels[i],
                                             s=1, alpha=.3, vmin=vmin, vmax=vmax)
            # mark the particle nearest each traversal point; numbers match vol_000..vol_009
            # in pc{i+1}/, and the line shows the order the traversal visits them
            _, traj_ind = analysis.get_nearest_point(z, pc_trajs[i])
            np.savetxt(f'{outdir}/pc{i+1}/z_pc_nearest_ind.txt', traj_ind, fmt='%d')
            ax.plot(umap_emb[traj_ind,0], umap_emb[traj_ind,1], '-', color='k', lw=0.8, zorder=2)
            analysis.annotate_centers(ax, umap_emb[traj_ind])
            ax.set_aspect('equal')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.savefig(f'{outdir}/pc{i+1}/umap.png')
            plt.close()
    return kmeans_labels, umap_emb

def plot_pca_variance(pca, out_png):
    '''Scree plot: each PC's share of the latent variance plus the running total,
    to judge how much of the latent space the PC1/PC2 plots actually show.'''
    ratio = 100 * np.asarray(pca.explained_variance_ratio_)
    n = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(n, ratio, color='C0', label='per PC')
    ax.plot(n, np.cumsum(ratio), '-o', color='C1', markersize=3, label='cumulative')
    ax.set_xticks(n)
    ax.set_ylim(0, 100)
    ax.set_xlabel('principal component')
    ax.set_ylabel('explained variance (%)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_png)
    plt.close(fig)

def plot_cluster_sizes(labels, K, out_png):
    '''Particles per k-means cluster, in the same colors as kmeans{K}/umap_clusters.png.'''
    counts = np.bincount(labels, minlength=K)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * K), 4))
    bars = ax.bar(np.arange(K), counts, color=analysis._get_colors(K))
    ax.bar_label(bars, labels=[str(c) for c in counts], fontsize=7, padding=1)
    ax.set_xticks(np.arange(K))
    ax.set_xlabel('k-means cluster')
    ax.set_ylabel('particles')
    ax.set_title(f'{len(labels)} particles in {K} clusters', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(out_png)
    plt.close(fig)

def plot_losses(run_log, out_png):
    '''Plot per-epoch training/validation loss parsed from run.log.
    Uses its own figure (closed afterwards) so the curves do not leak into the
    implicit figure 1 that later plots reuse.'''
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_max = 0
    for prefix in ("training", "validation"):
        losses = analysis.parse_loss_vanilla(run_log, prefix)
        if len(losses) == 0:
            continue
        n_max = max(n_max, len(losses))
        epochs = np.arange(1, len(losses) + 1)
        # markers help when there are only a few epochs, but clutter long runs
        ax.plot(epochs, losses, label=prefix, linewidth=1.5,
                marker='o' if len(losses) <= 30 else None, markersize=3)
    # at most ~10 integer ticks regardless of how many epochs were run
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    if n_max > 0:
        ax.set_xlim(0.5, n_max + 0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

class VolumeGenerator:
    '''Helper class to call analysis.gen_volumes'''
    def __init__(self, weights, config, vol_args={}, skip_vol=False):
        self.weights = weights
        self.config = config
        self.vol_args = vol_args
        self.skip_vol = skip_vol

    def gen_volumes(self, outdir, z_values):
        if self.skip_vol: return
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        zfile = f'{outdir}/z_values.txt'
        np.savetxt(zfile, z_values)
        analysis.gen_volumes(self.weights, self.config, zfile, outdir, **self.vol_args)

def main(args):
    t1 = dt.now()
    E = args.epoch
    log(f"analyzing {E}")
    workdir = args.workdir
    zfile = f'{workdir}/z.{E}.pkl'
    poses = args.pose
    weights = f'{workdir}/weights.{E}.pkl'
    config = f'{workdir}/config.pkl'
    outdir = f'{workdir}/analyze.{E}'
    outdir_multi = f'{workdir}/defanalyze.{E}'
    if E == -1:
        zfile = f'{workdir}/z.pkl'
        weights = f'{workdir}/weights.pkl'
        outdir = f'{workdir}/analyze'

    if args.outdir:
        outdir = args.outdir
    log(f'Saving results to {outdir}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # defined for all branches so the run manifest (and the non-vanilla path) stay robust;
    # the vanilla branch below reassigns both from the loaded latents / pose tracker.
    multi_z = None
    groups = None
    pose_tilt = None

    if args.vanilla:
        if os.path.isfile(f"{workdir}/run.log"):
            plot_losses(f"{workdir}/run.log", f"{workdir}/train_losses.png")
        load_kwargs = {}
        if "weights_only" in inspect.signature(torch.load).parameters:
            # PyTorch is new enough to support this arg (e.g. >= 2.x)
            load_kwargs["weights_only"] = False  # or True, if you want safe weights-only loading
        load_z = torch.load(zfile, **load_kwargs)
        z = load_z["mu"].cpu().numpy()
        log("loading {}, z shape {}".format(zfile, z.shape))
        Nimg = z.shape[0]
        zdim = z.shape[1]
        # args.pose may be None (it is optional): guard before os.path.isfile, which raises an
        # opaque TypeError on None. Same guard style as the --kpc subset export below.
        if args.pose is not None and os.path.isfile(args.pose):
            posetracker = PoseTracker.load(poses, Nimg, args.D, None, None,
                                       deform=True, deform_emb_size=zdim, latents=zfile, batch_size=4)# hp_order=2)
            groups = posetracker.euler_groups
            utils.save_pkl(groups, f"{workdir}/groups.pkl")
            # eulers are stored as hopf angles; column 1 equals the ZYZ Euler tilt (0-180 deg)
            pose_tilt = posetracker.eulers[:, 1].cpu().numpy()
            log("loading {}".format(poses))
        else:
            groups = None
            if args.pose is None:
                log('WARNING: no --pose given; UMAP plots will not be colored by Euler tilt '
                    'and groups.pkl will not be written')
            else:
                log(f'WARNING: pose file {args.pose} not found; continuing without particle groups')
        # loading z codes for deformation too
        if "multi_mu" in load_z:
            multi_z = load_z["multi_mu"].cpu().numpy()
        else:
            multi_z = None
        if args.joint and multi_z is None:
            raise ValueError(f"--joint needs conformation latents, but {zfile} has no 'multi_mu'. "
                             "It predates multi-body support; re-run analyze without --joint.")
    else:
        z = utils.load_pkl(zfile)
    zdim = z.shape[1]

    vol_args = dict(Apix=args.Apix, downsample=args.downsample, flip=args.flip, cuda=args.device)
    vg = VolumeGenerator(weights, config, vol_args, skip_vol=args.skip_vol)

    if zdim == 1:
        analyze_z1(z, outdir, vg)
    else:
        if args.kpc is None:
            analyze_zN(z, outdir, vg, pose_tilt, skip_umap=args.skip_umap, num_pcs=args.pc, num_ksamples=args.ksample, joint=args.joint, multi_z=multi_z)
        # perform pca on selected cluster
        if args.kpc is not None:
            umap_file = f'{outdir}/umap.pkl'
            if os.path.exists(umap_file):
                umap_emb = utils.load_pkl(umap_file)
            elif zdim <= 2:
                raise FileNotFoundError(
                    f"--kpc needs a UMAP embedding at {umap_file}, but analyze only writes one when "
                    f"zdim > 2 (here zdim={zdim}). Re-run this epoch with zdim > 2, or drop --kpc.")
            else:
                raise FileNotFoundError(
                    f"--kpc needs a UMAP embedding at {umap_file}, which is missing. Run a plain "
                    f"`analyze {E}` (without --kpc) first to produce it.")
            # kmeans clustering
            if ':' not in args.kpc:
                raise ValueError(f"--kpc must be K:classes, e.g. 20:2-5,8 -- the part before ':' is the "
                                 f"number of clusters of the previous analysis (got {args.kpc!r})")
            K = args.kpc.split(':')[0]
            log(f'select K-means clustering for class {args.kpc} in {K}...')

            if os.path.exists(f'{outdir}/kmeans{K}'):
                log('loading kmeans clustering')
                kmeans_labels = utils.load_pkl(f'{outdir}/kmeans{K}/labels.pkl')
                centers = utils.load_pkl(f'{outdir}/kmeans{K}/centers.pkl')
                centers_ind = np.loadtxt(f'{outdir}/kmeans{K}/centers_ind.txt', dtype=np.int64)
            else:
                raise FileNotFoundError(f"Previous KMeans not found: {outdir}/kmeans{K}")
            #keep only latents in selected classes
            selected_cls = args.kpc.split(':')[1].split(',')
            selected_pcs = []
            for x in selected_cls:
                #if '-' in string, then expand it
                if '-' in x:
                    ks = x.split('-')
                    assert int(ks[0]) < int(ks[1]), f"starting class number {ks[0]} must be smaller than ending class number {ks[1]}"
                    selected_pcs += list(range(int(ks[0]), int(ks[1])+1))
                else:
                    selected_pcs.append(int(x))
            ptcl_inds = np.arange(len(kmeans_labels))
            selected_ptcl_inds = ptcl_inds[np.isin(kmeans_labels, selected_pcs)]
            z_k = z[np.isin(kmeans_labels, selected_pcs)]
            umap_emb = umap_emb[np.isin(kmeans_labels, selected_pcs)]
            #keep only latents in selected classes
            if multi_z is not None:
                multi_z = multi_z[np.isin(kmeans_labels, selected_pcs)]
                if pose_tilt is not None:
                    pose_tilt = pose_tilt[np.isin(kmeans_labels, selected_pcs)]
            log(f'Perfoming principal component analysis for class {args.kpc}...')
            pc, pca = analysis.run_pca(z_k)
            outdir = f'{workdir}/analyze.filter.{E}'
            save_pca(outdir, pc, pca)
            for i in range(args.pc):
                start, end = np.percentile(pc[:,i],(1,99))
                log(f'traversing pc {i} from {start} to {end}')
                z_pc = analysis.get_pc_traj(pca, z_k.shape[1], 10, i+1, start, end)
                pc_dir = Path(outdir) / f"pc{i+1}"
                pc_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(f'{outdir}/pc{i+1}/z_pc.txt', z_pc)
            # kmeans clustering
            log(f'K-means clustering for class {args.kpc}...')
            K = args.ksample
            kmeans_labels_filtered, centers = analysis.cluster_kmeans(z_k, args.ksample)
            _, centers_ind = analysis.get_nearest_point(z_k, centers)
            #centers_ind indexes z_k, i.e. the selected subset. Keep it that way for multi_z
            #and umap_emb, which are filtered the same way, but map it back to the original
            #particle stack for centers_ind.txt -- that file is documented as particle indices,
            #and labels.pkl below is written in the original numbering too.
            centers_ind_orig = selected_ptcl_inds[centers_ind]
            #recover the original labels
            kmeans_labels[:] = args.ksample
            for i_k in range(args.ksample):
                kmeans_labels[selected_ptcl_inds[kmeans_labels_filtered == i_k]] = i_k
            if multi_z is not None:
                assert z_k.shape[0] == multi_z.shape[0], f"z shape {z.shape} must be equal to multi_z shape {multi_z.shape} at first dimension"
                multi_z_center = multi_z[centers_ind]
                centers_joint = np.concatenate([centers, multi_z_center], axis=-1)
            if not os.path.exists(f'{outdir}/kmeans{K}'):
                os.mkdir(f'{outdir}/kmeans{K}')

            utils.save_pkl(kmeans_labels, f'{outdir}/kmeans{K}/labels.pkl')
            utils.save_pkl(centers, f'{outdir}/kmeans{K}/centers.pkl')
            np.savetxt(f'{outdir}/kmeans{K}/centers.txt', centers)
            np.savetxt(f'{outdir}/kmeans{K}/centers_ind.txt', centers_ind_orig, fmt='%d')
            if args.joint:
                np.savetxt(f'{outdir}/kmeans{K}/centers_joint.txt', centers_joint)
            # export the kpc-selected particles so they can be re-trained on their own:
            #  - ind.filter.{E}.pkl: pass as --ind alongside the ORIGINAL star and pose; training
            #    applies this same index filter to the stack, poses and ctf consistently. Preferred.
            #  - pose.filter.{E}.pkl: a standalone subset pose tuple (re-indexed 0..M-1) matching a
            #    stack filtered by the same indices, for callers that want a self-contained pose.
            # selected_ptcl_inds is ascending, so both stay aligned with an ascending stack filter.
            utils.save_pkl(selected_ptcl_inds, f'{outdir}/ind.filter.{E}.pkl')
            log(f'saved {len(selected_ptcl_inds)} selected particle indices to {outdir}/ind.filter.{E}.pkl '
                f'(re-train with the original star and pose plus --ind this file)')
            # ... and the matching star, sliced from the star recorded in config.pkl. The tomo
            # datasets index the star row-for-row, so selected_ptcl_inds are star rows directly.
            star_in = None
            try:
                with open(config, 'rb') as f:
                    star_in = pickle.load(f)['dataset_args']['particles']
            except Exception as e:
                log(f'WARNING: could not read the input star from {config} ({e}); '
                    f'skipping the filtered star')
            if star_in is not None:
                star_out = f'{outdir}/particles.filter.{E}.star'
                if not os.path.isfile(star_in):
                    log(f'WARNING: the star recorded in config.pkl is missing ({star_in}); '
                        f'run `dsd filter_star <star> {outdir}/ind.filter.{E}.pkl -o {star_out}` '
                        f'once it is available')
                else:
                    try:
                        n_sel, n_total = filter_star(star_in, selected_ptcl_inds, star_out)
                        log(f'saved {n_sel} of {n_total} particles to {star_out}')
                    except Exception as e:
                        log(f'WARNING: could not write {star_out} ({e})')

            if args.pose is not None and os.path.isfile(args.pose):
                sub_poses = pickle.load(open(args.pose, 'rb'))
                if not isinstance(sub_poses, tuple):
                    sub_poses = (sub_poses,)
                npose = len(sub_poses[0])
                if npose != len(kmeans_labels):
                    log(f'WARNING: pose {args.pose} has {npose} entries but the analysis has '
                        f'{len(kmeans_labels)} particles; skipping subset pose (would be misaligned)')
                else:
                    sub_poses = tuple(np.asarray(p)[selected_ptcl_inds] for p in sub_poses)
                    pickle.dump(sub_poses, open(f'{outdir}/pose.filter.{E}.pkl', 'wb'))
                    log(f'saved subset pose ({len(selected_ptcl_inds)} particles) to {outdir}/pose.filter.{E}.pkl')
            xmax = np.max(umap_emb[:, 0])
            xmin = np.min(umap_emb[:, 0])
            ymax = np.max(umap_emb[:, 1])
            ymin = np.min(umap_emb[:, 1])
            analysis.scatter_annotate(umap_emb[:,0], umap_emb[:,1], centers_ind=centers_ind, annotate=True,
                                  xlim=(xmin, xmax), ylim=(ymin, ymax),
                                  alpha=.15, s=1.)
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.savefig(f'{outdir}/kmeans{K}/umap.png')
            plt.close()

        if multi_z is not None:
            if args.kpc is not None:
                outdir_multi = f'{workdir}/defanalyze.filter.{E}'
            if not os.path.exists(outdir_multi):
                os.mkdir(outdir_multi)
            analyze_zN(multi_z, outdir_multi, vg, pose_tilt, skip_umap=args.skip_umap, num_pcs=min(args.pc, multi_z.shape[1]), num_ksamples=args.ksample)

    # run manifest -- makes the output dir self-describing so downstream custom
    # analysis knows which latents/pose/config produced it without re-deriving.
    manifest = {
        'timestamp': dt.now().isoformat(),
        'cryodrgn_version': cryodrgn.__version__,
        'epoch': E,
        'workdir': workdir,
        'zfile': zfile,
        'weights': weights,
        'config': config,
        'pose': poses,
        'outdir': outdir,
        'vanilla': bool(args.vanilla),
        'zdim': int(zdim),
        'num_particles': int(z.shape[0]),
        'has_multi_z': multi_z is not None,
        'multi_zdim': int(multi_z.shape[1]) if multi_z is not None else None,
        'num_pcs': args.pc,
        'num_ksamples': args.ksample,
        'kpc': args.kpc,
        'joint': bool(args.joint),
        'Apix': args.Apix,
        'downsample': args.downsample,
    }
    with open(f'{outdir}/analyze_config.json', 'w') as f:
        # default=str keeps the manifest writable if a path-like or numpy scalar
        # reaches one of the fields.
        json.dump(manifest, f, indent=2, default=str)
    log(f'wrote run manifest to {outdir}/analyze_config.json')

    # copy over template if file doesn't exist
    out_ipynb = f'{outdir}/cryoDRGN_viz.ipynb'
    if not os.path.exists(out_ipynb):
        log(f'Creating jupyter notebook...')
        ipynb = f'{cryodrgn._ROOT}/templates/cryoDRGN_viz_template.ipynb'
        shutil.copyfile(ipynb, out_ipynb)
    else:
        log(f'{out_ipynb} already exists. Skipping')
    log(out_ipynb)

    # copy over template if file doesn't exist
    out_ipynb = f'{outdir}/cryoDRGN_filtering.ipynb'
    if not os.path.exists(out_ipynb):
        log(f'Creating jupyter notebook...')
        ipynb = f'{cryodrgn._ROOT}/templates/cryoDRGN_filtering_template.ipynb'
        shutil.copyfile(ipynb, out_ipynb)
    else:
        log(f'{out_ipynb} already exists. Skipping')
    log(out_ipynb)

    log(f'Finished in {dt.now()-t1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
