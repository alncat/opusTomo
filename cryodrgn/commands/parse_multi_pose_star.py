'''Parse image poses from RELION .star file'''

import argparse
import numpy as np
import sys, os
import pickle

from cryodrgn import utils
from cryodrgn import lie_tools
from cryodrgn import starfile
from cryodrgn import dataset
from cryodrgn import dynamics
from cryodrgn.utils import ALIGN_CORNERS
import torch.nn.functional as F
import torch

log = utils.log

# the geometry lives in cryodrgn.dynamics so that this command and the
# analyze_dynamics command derive axes with the same code
center_of_mass = dynamics.center_of_mass

def add_args(parser):
    parser.add_argument('input', help='RELION .star file')
    parser.add_argument('-D', type=int, required=True, help='Box size of reconstruction (pixels)')
    parser.add_argument('--relion31', action='store_true', help='Flag for relion3.1 star format')
    parser.add_argument('--Apix', type=float, help='Pixel size (A); Required if translations are specified in Angstroms')
    parser.add_argument('-o', metavar='PKL', type=os.path.abspath, required=False, help='Output pose.pkl')
    parser.add_argument('--labels', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--masks', metavar='PKL', type=os.path.abspath, required=True, help='masks for multi-body')
    parser.add_argument('--volumes', metavar='PKL', type=os.path.abspath, required=False, help='Output label.pkl')
    parser.add_argument('--num-volumes', type=int, default=10, help='Number of referenceN.mrc volumes to read from --volumes (default: %(default)s)')
    parser.add_argument('--bodies', type=int, required=True, help='Number of bodies')
    parser.add_argument('--origin-rel', type=int, default=1, help='0-based index of the anchor/origin body used for orientation alignment (default: %(default)s)')
    parser.add_argument('--outmasks', default="mask_params", help="the name of pkl file storing masks related parameters")
    parser.add_argument('--outdir', type=os.path.abspath, help='Directory to write the mask-params pkl into (default: the directory of --masks)')
    return parser

def main(args):
    assert args.input.endswith('.star'), "Input file must be .star file"
    #assert args.o.endswith('.pkl'), "Output format must be .pkl"

    s = starfile.Starfile.load_multibody(args.input, relion31=args.relion31)
    N = len(s.df)
    log('{} particles'.format(N))

    # parse rotations
    keys = ('_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi')
    euler = np.empty((N,3))
    euler[:,0] = s.df['_rlnAngleRot']
    euler[:,1] = s.df['_rlnAngleTilt']
    euler[:,2] = s.df['_rlnAnglePsi']
    log('Euler angles (Rot, Tilt, Psi):')
    log(euler[0])
    log('Converting to rotation matrix:')
    rot = np.asarray([utils.R_from_relion(*x) for x in euler])
    log(rot[0])

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

    log('Translations (pixels):')
    log(trans[0])

    # convert translations from pixels to fraction
    trans /= args.D

    #process multibody
    log(f"there are {args.bodies} bodies")
    if s.multibodies is not None and len(s.multibodies) != 0:
        assert len(s.multibodies) == args.bodies
        body_eulers = []
        body_trans = []
        for b_i in range(args.bodies):
            body = s.multibodies[b_i]
            keys = ('_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi')
            euler_body = np.empty((N,1,3))
            assert len(body) == N
            for i in range(3):
                euler_body[:,0,i] = body[keys[i]]
            log('Euler angles (Rot, Tilt, Psi):')
            log(euler_body[0])
            body_eulers.append(euler_body)
            trans_body = np.zeros((N,1,3))
            body_header = s.multibody_headers[b_i]
            if '_rlnOriginX' in body_header and '_rlnOriginY' in body_header:
                trans_body[:,0,0] = body['_rlnOriginX']
                trans_body[:,0,1] = body['_rlnOriginY']
            elif '_rlnOriginXAngst' in body_header and '_rlnOriginYAngst' in body_header:
                assert args.Apix is not None, "Must provide --Apix argument to convert _rlnOriginXAngst and _rlnOriginYAngst translation units"
                trans_body[:,0,0] = body['_rlnOriginXAngst']
                trans_body[:,0,1] = body['_rlnOriginYAngst']
                trans_body /= args.Apix

            log('Translations (pixels):')
            log(trans_body[0])
            trans_body /= args.D
            body_trans.append(trans_body)
    else:
        body_eulers = []
        body_trans = []
        for b_i in range(args.bodies):
            euler_body = np.zeros((N,1,3))
            euler_body[:,0,1] = 90.
            trans_body = np.zeros((N,1,3))
            body_eulers.append(euler_body)
            body_trans.append(trans_body)

    if len(body_eulers):
        body_eulers = np.concatenate(body_eulers, axis=1)
        body_trans = np.concatenate(body_trans, axis=1)
        print(body_eulers.shape, body_trans.shape)

    # write output
    if args.o is not None:
        log(f'Writing {args.o}')
        with open(args.o,'wb') as f:
            if len(body_eulers):
                pickle.dump((rot,trans,euler,body_eulers,body_trans),f)
            else:
                pickle.dump((rot,trans,euler),f)

    log(f'Loading reference volume from {args.masks}')
    s_mask = starfile.Starfile.load(args.masks)
    prefix = os.path.dirname(args.masks)
    print(s_mask.headers, prefix)
    #assert len(s_mask.df) == len(s.multibodies)
    in_relatives = []
    com_bodies = []
    radii = []
    masks = []
    axes = []
    for b_i in range(len(s_mask.df)):
        mask_name = prefix + "/" + s_mask.df['_rlnBodyMaskName'][b_i]
        in_relatives.append(int(s_mask.df['_rlnBodyRotateRelativeTo'][b_i]) - 1)
        print(mask_name)
        ref_vol = dataset.VolData(mask_name)
        masks.append(ref_vol.get())
        # cryodrgn.dynamics.center_of_mass rather than VolData.center_of_mass: the latter rounds
        # the centre to the nearest voxel (~0.4 px here, and it also biases the radii/axes, which
        # are measured relative to it), weights the inertia tensor by density^3, and uses eig
        # instead of eigh.
        c, r, eigvecs = dynamics.center_of_mass(ref_vol.get())
        com_bodies.append(c)
        radii.append(r)
        axes.append(eigvecs)

    masks = torch.stack(masks, dim=0)
    masks = (masks > 0)*masks

    # models.py normalises every length by vol_size = D-1 (models.py:1119) and then multiplies by
    # scale = render_size/(crop_vol_size-1)*2, which works out to "one unit = one voxel of the
    # reconstruction grid". The body masks are measured on their own grid, so convert whenever the
    # mask box differs from the reconstruction box. Verified against training: this dataset stores
    # |com| = 43.37 mask voxels and the decoder logged com/vol_size*scale = 0.5455, i.e. it placed
    # the pivots 43.37 instead of 41.97 reconstruction voxels out -- 3.8 A too far.
    mask_box = masks.shape[-1]
    to_vol = args.D/mask_box
    if mask_box != args.D:
        log(f"WARNING: body masks are {mask_box}^3 but the reconstruction box (-D) is {args.D}; "
            f"rescaling body geometry by {to_vol:.4f}. Check that --masks belongs to this dataset "
            f"(masks drawn on the reconstruction grid need no correction).")
    com_bodies = [c*to_vol for c in com_bodies]
    radii = [r*to_vol for r in radii]
    vol_coms = None
    rot_radii = None
    if args.volumes:
        #read in dynamics volumes
        assert args.num_volumes >= 2, \
            f"--num-volumes must be at least 2 (the first and last volume define the motion), got {args.num_volumes}"
        vols = []
        for b_i in range(args.num_volumes):
            mask_name = args.volumes + "/reference" + str(b_i) + ".mrc"
            assert os.path.isfile(mask_name), \
                f"missing volume {mask_name} (expected {args.num_volumes} referenceN.mrc files; set --num-volumes)"
            print(mask_name)
            ref_vol = dataset.VolData(mask_name)
            vols.append(ref_vol.get())
            if b_i == 0:
                # Register the masks to the rendered volume. The rendered box is NOT a rescaled
                # reconstruction box: models.py samples it on grid_coarse spanning [-1,1] with
                # scale = render_size/(crop_vol_size-1)*2, so one rendered voxel equals one
                # reconstruction voxel and the box is the CENTRE CROP of the render box (the
                # training log confirms crop fraction (160-1)/(180-1) = 0.888268). Interpolating
                # the mask straight onto the rendered box, as this used to do, ignores that crop
                # and shrinks the mask by 1/window_r -- about 12.5% here, i.e. 7-8 voxels of
                # misregistration at the molecule's edge.
                vol_box = vols[0].shape[-1]
                assert vol_box <= args.D, \
                    f"rendered volume box {vol_box} is larger than the reconstruction box -D {args.D}"
                m = F.interpolate(masks.unsqueeze(0), size=(args.D,)*3, mode='trilinear',
                                  align_corners=ALIGN_CORNERS)      # mask grid -> reconstruction grid
                off = (args.D - vol_box)//2
                masks = m[..., off:off+vol_box, off:off+vol_box, off:off+vol_box].squeeze(0)
                log("registered masks {}^3 -> {}^3 (resample to -D) -> {}^3 (centre crop, "
                    "matching the rendered box)".format(mask_box, args.D, vol_box))
                print(masks.sum(dim=(1,2,3)))
        # the "consensus" frame used for coms/principal axes: middle of the series.
        # (len-1)//2 reproduces the previously hardcoded index 4 for the default 10 volumes.
        mid_vol = (len(vols) - 1)//2

        trajs = []
        vol_coms = []
        vol_radii = []
        principal_axes = []
        for m_i in range(masks.shape[0]):
            # the whole trajectory, not just the endpoints: the axis is fitted from every frame
            traj = dynamics.com_trajectory(vols, masks[m_i])   # 渲染体素 == 重构体素
            trajs.append(traj)
            st = dynamics.trajectory_stats(traj)
            log("body {}: com moves {:.2f} px net over a {:.2f} px path (coherence {:.2f})".format(
                m_i, st["displacement"], st["path_length"], st["coherence"]))
            # radii and principal axes must come from the SAME center_of_mass call: models.py
            # pairs radius[k] with principal_axes[k] in the anisotropic Gaussian body mask, and
            # each call sorts its own eigenvalues. Taking radii from the mask loop (as before)
            # while taking axes from the volume mismatched that pairing.
            vol_com, vol_r, p_axes = center_of_mass(vols[mid_vol]*masks[m_i])
            vol_coms.append(vol_com)
            vol_radii.append(vol_r)
            principal_axes.append(p_axes)

        rot_axes = []
        orientations = []
        rot_radii = []
        # honor --origin-rel here too; the default (non-volumes) path already does. The old
        # auto-guess is still reported so a mismatch is visible.
        auto_origin_rel = np.bincount(in_relatives).argmax()
        origin_rel = args.origin_rel
        print(f"origin_rel: {origin_rel} (--origin-rel; most common parent body would be {auto_origin_rel})")
        for m_i in range(masks.shape[0]):
            pivot = com_bodies[in_relatives[m_i]]
            # Fit the axis to the whole trajectory (plane fit) rather than crossing the first and
            # last lever arms, which uses two frames and is noise-dominated for small motions.
            fit = dynamics.fit_rotation_axis(trajs[m_i], pivot=pivot)
            rot_axis = fit["axis"]
            log("body {}: axis {} | swept {:+.2f} deg | dispersion {:.1f} deg | "
                "out-of-plane {:.2f} px | lever {:.1f} px".format(
                    m_i, [round(float(x), 3) for x in rot_axis], fit["swept_angle"],
                    fit["dispersion"], fit["out_of_plane"], fit["lever"]))
            if not fit["well_defined"]:
                log("WARNING: body {} has no reliable rotation axis about body {} "
                    "(swept {:.2f} deg, dispersion {:.1f} deg across frame pairs). The motion is "
                    "too small or too incoherent to define an axis; note a body spinning about "
                    "its own centre of mass is invisible to this method entirely. Check the "
                    "volume series -- `dsd analyze_dynamics` ranks them.".format(
                        m_i, in_relatives[m_i], fit["swept_angle"], fit["dispersion"]))
            lever = F.normalize(pivot - trajs[m_i][0], dim=0)
            rot_axes.append(rot_axis)
            # right-handed frame (det=+1) mapping the lever arm to x and the axis to z; a
            # reflection here would flip the chirality of every learned body rotation and
            # disagree with the default path's align_with_z.
            orientations.append(dynamics.orientation_from_axis(lever, rot_axis))
            if m_i == origin_rel:
                rot_radii.append(vol_coms[m_i] - vol_coms[m_i])
            else:
                rot_radii.append(vol_coms[m_i] - vol_coms[in_relatives[m_i]])
            #print(rot_axis, mat)
            #print(mat@rot_axis)
            #print(mat@mat.T)

        if len(rot_axes) == 2:
            ang, kind = dynamics.interpret_pair(rot_axes[0], rot_axes[1])
            log("the two bodies' axes are {:.1f} deg apart: {}".format(ang, kind))
            if kind != 'hinge (antiparallel)':
                log("WARNING: for a two-body system the axes should be roughly antiparallel. "
                    "Parallel axes mean the whole complex is drifting rather than flexing, and "
                    "an uncorrelated angle means the motion is noise -- either way this series "
                    "will not give a meaningful multi-body decomposition.")
        rot_axes = torch.stack(rot_axes, dim=0)
        orientations = torch.stack(orientations, dim=0)
        vols = torch.stack(vols, dim=0)
        rot_radii = torch.stack(rot_radii, dim=0)
        vol_coms = torch.stack(vol_coms, dim=0)
        vol_radii = torch.stack(vol_radii, dim=0)
        principal_axes = torch.stack(principal_axes, dim=0)
        print("translate orientations: ", orientations)#, principal_axes)

    consensus_mask = masks.mean(dim=0)
    #weights = F.softmax(masks*4, dim=0)
    #print(weights.shape)

    com_bodies = torch.stack(com_bodies, dim=0)
    if vol_coms is None:
        vol_coms = com_bodies
    radii_bodies = torch.stack(radii, dim=0)
    rotate_directions = []
    rotate_directions_ori = []
    orient_bodies = []
    relats = []
    print("in_relatives: ", in_relatives)
    #print("com_bodies: ", com_bodies - vol_coms, "radii_bodies: ", radii_bodies)
    origin_rel = args.origin_rel #np.bincount(in_relatives).argmax()
    print("origin_rel:", origin_rel)
    for b_i in range(len(s_mask.df)):
        rotate_directions.append(com_bodies[in_relatives[b_i]] - com_bodies[b_i])
        rotate_directions_ori.append(com_bodies[b_i] - com_bodies[in_relatives[b_i]])
        rotate_directions[-1] = F.normalize(rotate_directions[-1], dim=0)
        if b_i != origin_rel:
            orient_bodies.append(utils.align_with_z(-rotate_directions[-1]))
        else:
            orient_bodies.append(utils.align_with_z(rotate_directions[-1]))
        print(rotate_directions[-1].shape, orient_bodies[-1] @ rotate_directions[-1])
        relats.append(com_bodies[in_relatives[b_i]])
        #reset rotation axis for center
        #if b_i == origin_rel:
        #    rotate_directions_ori[b_i] = com_bodies[b_i] - com_bodies[b_i]
        #normalize direction
    A_rot90 = lie_tools.yrot(torch.tensor(-90))
    rotate_directions = torch.stack(rotate_directions, dim=0)
    rotate_directions_ori = torch.stack(rotate_directions_ori, dim=0)
    if rot_radii is None:
        rot_radii = rotate_directions
    #print((orientations@rotate_directions_ori.unsqueeze(-1)).squeeze(), rot_axes, orientations)
    #print((orientations@rot_radii.unsqueeze(-1)).squeeze())
    #print(orientations@torch.transpose(principal_axes, -1, -2))
    print("rotate_directions from volumes: ", rot_radii)
    orient_bodies = torch.stack(orient_bodies, dim=0)
    relats = torch.stack(relats, dim=0)
    axes = torch.stack(axes, dim=0)
    #print("A_rot90: ", A_rot90)
    #print("relats: ", relats)
    print("rotate_directions: ", rotate_directions_ori)
    print("orient_bodies: ", orient_bodies)
    # --outdir was previously accepted and silently ignored; honor it, falling back to the
    # mask starfile's directory as before.
    out_prefix = args.outdir if args.outdir else prefix
    if args.outdir and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    output_name = out_prefix + f"/{args.outmasks}.pkl"
    log(f'Writing {output_name}')
    if not args.volumes:
        print("principal_axes: ", axes)
        print("com_bodies: ", com_bodies)
        torch.save({"in_relatives": relats, "com_bodies": com_bodies,
                "orient_bodies": orient_bodies, "rotate_directions": rotate_directions_ori, "radii_bodies": radii_bodies, "principal_axes": axes}, \
    #            #"weights": weights, "consensus_mask": consensus_mask},
               output_name)
    else:
        # radii and principal_axes both volume-derived so they stay index-paired (see models.py's
        # exp(-0.2*sum(x_k^2/radius_k^2)) body mask); previously radii came from the mask instead.
        torch.save({"in_relatives": relats, "com_bodies": vol_coms, "radii_bodies": vol_radii,
                "orient_bodies": orientations, "rotate_directions": rot_radii, "principal_axes": principal_axes},  \
                #"weights": weights, "consensus_mask": consensus_mask},
               output_name)
    # shift of each rigid body in experimental data is selfRound(my_old_offset - Aori*com) + ibody_offset
    # backproject into reference model should be, Aresi(x-com) + com - Inv(Aori)*ibody_offset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
