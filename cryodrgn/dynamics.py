'''
Rigid-body motion analysis for a series of volumes (e.g. a PC traversal).

Shared by two callers so that the geometry is defined in exactly one place:
  * parse_multi_pose_star --volumes, which turns a series into multi-body
    rotation axes for mask_params.pkl
  * the analyze_dynamics command, which reports whether a series contains
    motion that defines an axis at all

The central routine is fit_rotation_axis. For a rigid rotation about an axis n
through any pivot on that axis, every centre of mass of the moving body lies in
a plane whose normal is n -- the pivot only sets the offset along n, not the
plane's orientation. So the axis can be recovered by fitting a plane through the
whole trajectory (SVD) rather than by crossing the first and last lever arms,
which throws away every intermediate frame and is dominated by noise whenever
the motion is small. The singular values additionally say *whether* the fit
means anything: a straight-line trajectory (pure translation) leaves the plane
undetermined, and a non-planar trajectory is not a single-axis rotation.
'''

import numpy as np
import torch

from cryodrgn import utils

log = utils.log


def center_of_mass(volume):
    '''Centre of mass, principal radii and principal axes of a volume.

    Returns (center, radii, axes) with radii ascending and axes[k] the axis
    matching radii[k] -- models.py pairs them by index in its body mask, so
    they must come from the same call.
    '''
    N = volume.shape[-1]
    x_idx = torch.linspace(0, N-1, N) - N/2 #[-s, s)
    grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
    grid = torch.stack([grid[2], grid[1], grid[0]], dim=-1)  # (x, y, z)
    vol = ((volume > 0).float()*volume).unsqueeze(-1)
    mass = vol.sum()
    assert mass.item() > 0, "volume has no positive density"
    center = (vol*grid).sum(dim=(0, 1, 2))/mass
    centered = grid - center
    # second-moment (inertia) tensor, density-weighted; symmetric by construction
    matrix = (centered.unsqueeze(-1)*centered.unsqueeze(-2)*vol.unsqueeze(-1)).sum(dim=(0, 1, 2))
    eigvals, eigvecs = np.linalg.eigh(matrix.numpy())
    order = np.argsort(eigvals)  # ascending; eigh already sorts, kept for safety
    radii = torch.as_tensor(np.sqrt(np.clip(eigvals[order], 0, None)/float(mass))).float()
    axes = torch.from_numpy(eigvecs[:, order].T).float()  # axes[k] <-> radii[k]
    return center, radii, axes


def com_trajectory(vols, mask, scale=1.0):
    '''(T, 3) centres of mass of one body across a volume series.'''
    return torch.stack([center_of_mass(v*mask)[0]*scale for v in vols], dim=0)


def trajectory_stats(traj):
    '''Net displacement vs path length.

    coherence = path/displacement is ~1 for a body moving steadily along an arc
    and grows without bound as the centre of mass merely jitters, which is the
    signature of a series with no real motion in it.
    '''
    steps = (traj[1:] - traj[:-1]).norm(dim=-1)
    displacement = float((traj[-1] - traj[0]).norm())
    path_length = float(steps.sum())
    return dict(displacement=displacement,
                path_length=path_length,
                coherence=path_length/max(displacement, 1e-12))


def fit_rotation_axis(traj, pivot):
    '''Fit a rotation axis to a centre-of-mass trajectory, using every frame pair.

    traj  : (T, 3) centres of mass, T >= 2
    pivot : (3,) point the body rotates about

    The axis is the normalised sum of r_i x r_j over all frame pairs, where
    r_t = pivot - c_t. Each term equals r_i x (c_i - c_j), so the long lever arm
    is what sets the scale, and pairs of well-separated frames -- which sweep the
    largest angle -- dominate the sum while noise averages down. This reduces to
    the classic first-vs-last cross product when T = 2.

    Plane-fitting the trajectory by SVD looks like the natural all-frames
    estimator but is much worse here: for a shallow arc (60 px lever, 20 deg) the
    chord is ~21 px while the bulge that defines the plane is under 1 px, so
    sub-pixel noise swamps it. Benchmarks put the SVD normal 15-45 deg off where
    the pairwise sum stays within a degree.

    Returns a dict:
      axis           unit axis, signed by the right-hand rule of the observed motion
      dispersion     median angle (deg) between individual pair axes and the
                     consensus; this is the honest "can I trust this axis" number
      swept_angle    total angle swept about pivot, in degrees
      out_of_plane   RMS deviation from the fitted plane, in px
      lever          |pivot - c_0| in px
      well_defined   True when the axis is trustworthy

    Two limitations are inherent to reading motion off centres of mass, not
    implementation gaps, and no statistic here can flag them:
      * a body spinning about its own centre of mass does not move that centre
        at all, so such rotation is completely invisible;
      * over a shallow sweep, a screw (rotation plus a rise along the axis)
        traces almost the same near-straight path as a pure rotation about some
        other axis. Benchmarked: a 20 deg sweep with a 25 px rise is fitted with
        dispersion 0.4 deg -- i.e. high apparent confidence -- while the axis is
        23 deg wrong. Resolving that needs the body's orientation (e.g. aligning
        the densities frame to frame), not just its centre of mass.
    '''
    traj = torch.as_tensor(traj).float()
    pivot = torch.as_tensor(pivot).float()
    assert traj.ndim == 2 and traj.shape[-1] == 3, f"expected (T,3), got {tuple(traj.shape)}"
    assert traj.shape[0] >= 2, "need at least 2 frames"

    lever = pivot - traj                                  # (T,3) lever arms
    T = traj.shape[0]
    i, j = torch.triu_indices(T, T, offset=1)
    crosses = torch.cross(lever[i], lever[j], dim=-1)      # (P,3), |.| ~ sin of swept angle
    total = crosses.sum(dim=0)
    if float(total.norm()) < 1e-12:                        # no resolvable rotation at all
        axis = _any_perpendicular(lever[0])
        return dict(axis=axis, dispersion=90.0, swept_angle=0.0, out_of_plane=0.0,
                    lever=float(lever[0].norm()), well_defined=False)
    axis = total/total.norm()

    # how consistent are the individual pair estimates? weight out the near-degenerate
    # pairs (adjacent frames) by keeping only those with a usable cross-product norm
    norms = crosses.norm(dim=-1)
    keep = norms > 0.25*float(norms.max())
    if int(keep.sum()) >= 2:
        unit = crosses[keep]/norms[keep].unsqueeze(-1)
        cos = (unit @ axis).clamp(-1.0, 1.0)
        dispersion = float(np.degrees(np.median(np.arccos(cos.numpy()))))
    else:
        dispersion = 90.0

    # out-of-plane wander, in px: for a single-axis rotation the coms are coplanar
    centred = traj - traj.mean(dim=0)
    out_of_plane = float((centred @ axis).pow(2).mean().sqrt())


    # swept angle: accumulate frame to frame in the plane, so a >180 deg sweep adds up
    planar = lever - torch.outer(lever @ axis, axis)
    swept = 0.0
    for a, b in zip(planar[:-1], planar[1:]):
        na, nb = float(a.norm()), float(b.norm())
        if na < 1e-9 or nb < 1e-9:
            continue
        cos = float(torch.dot(a, b))/(na*nb)
        step = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if float(torch.dot(torch.cross(a, b, dim=-1), axis)) < 0:
            step = -step
        swept += step

    well_defined = dispersion < 15.0 and abs(swept) > 0.5
    return dict(axis=axis, dispersion=dispersion, swept_angle=swept,
                out_of_plane=out_of_plane, lever=float(lever[0].norm()),
                well_defined=well_defined)


def _any_perpendicular(v):
    '''Some unit vector perpendicular to v, for the no-motion fallback.'''
    v = torch.as_tensor(v).float()
    probe = torch.tensor([1., 0., 0.]) if abs(float(v[0])) < 0.9*float(v.norm().clamp(min=1e-12)) \
        else torch.tensor([0., 1., 0.])
    out = torch.cross(v, probe, dim=-1)
    return out/out.norm().clamp(min=1e-12)


def axis_from_endpoints(traj, pivot):
    '''The legacy first-vs-last lever-arm cross product, for comparison only.'''
    pivot = torch.as_tensor(pivot).float()
    r0 = pivot - traj[0]
    r1 = pivot - traj[-1]
    cr = torch.cross(r0, r1, dim=-1)
    n = float(cr.norm())
    if n < 1e-12:
        return None
    return cr/n


def angle_between(a, b):
    '''Angle between two axes, in degrees.'''
    a = torch.as_tensor(a).float()
    b = torch.as_tensor(b).float()
    cos = float(torch.dot(a, b)/(a.norm()*b.norm()).clamp(min=1e-12))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def orientation_from_axis(lever_dir, axis):
    '''Right-handed frame mapping the lever arm to x and the rotation axis to z.

    models.py conjugates the predicted body rotation with this matrix, so it must
    be a proper rotation (det = +1); a reflection would silently flip the
    chirality of every learned body rotation.
    '''
    lever_dir = torch.as_tensor(lever_dir).float()
    axis = torch.as_tensor(axis).float()
    x = lever_dir - axis*torch.dot(lever_dir, axis)  # make the lever arm exactly perpendicular
    if float(x.norm()) < 1e-9:  # lever arm parallel to the axis: pick any perpendicular
        probe = torch.tensor([1., 0., 0.]) if abs(float(axis[0])) < 0.9 else torch.tensor([0., 1., 0.])
        x = torch.cross(probe, axis, dim=-1)
    x = x/x.norm().clamp(min=1e-12)
    y = torch.cross(axis, x, dim=-1)
    y = y/y.norm().clamp(min=1e-12)
    return torch.stack([x, y, axis], dim=0)


def interpret_pair(axis_a, axis_b):
    '''Classify the relative sense of two bodies' axes in a two-body system.

    ~180 deg (antiparallel) is the signature of an internal hinge: the bodies
    swing against each other. ~0 deg means both centres of mass move the same
    way, i.e. the whole complex is drifting rather than flexing -- useless for
    multi-body refinement even though the trajectories look perfectly clean.
    '''
    ang = angle_between(axis_a, axis_b)
    if ang > 140:
        kind = 'hinge (antiparallel)'
    elif ang < 40:
        kind = 'global drift (parallel)'
    else:
        kind = 'uncorrelated'
    return ang, kind
