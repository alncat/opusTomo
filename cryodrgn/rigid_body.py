'''
Decode the conformation latent code into interpretable rigid-body motion.

The conformation decoder predicts, for each predefined body, a rotation and translation that
are applied at projection time.  Nothing about that motion is visible in a statically saved
volume unless the multibody path is used, so the only way to read the motion off a trained
model is to replay the affine head on a conformation code and compose the transforms exactly
as the decoder does.

The composition reproduced here is the one used in VanillaDecoder.save:

    R_body_i = orient_bodies_i^T @ quat_to_SO3([16, affine(z)_i,:3]) @ orient_bodies_i

for the first num_bodies transforms.  The head emits num_bodies+1 of them; the extra global
transform is dropped, matching the decoder.  Note this is a conjugation, not a left-multiply
by orient_bodies -- getting that wrong changes the angle by degrees.

Handedness: these transforms live in the model's native frame, which is the mirror image of
the frame the volumes are written in when eval_vol is given --flip.  A mirror is improper, so
it inverts rotation handedness: a rotation axis is a pseudovector and transforms as
(ax, ay, az) -> (-ax, -ay, az), while an ordinary position difference transforms as
(x, y, z) -> (x, y, -z).  Pass flip=True to report in the flipped frame, which is the one that
matches deposited models.
'''

import re

import numpy as np
import torch

QUAT_W = 16.0  # the decoder builds its quaternion as [16, v] before normalising

# The handedness label is the sign of the axis projected onto the line joining two body
# centres.  Near perpendicular that sign is decided by numerical noise, so below this
# projection the sense is reported as undetermined rather than stated with false confidence.
MIN_AXIS_PROJECTION = 0.2


def load_affine_head(state_dict):
    '''Return (affine_fn, num_bodies) for a trained decoder state dict.

    affine_fn maps a conformation code of shape (d,) to an array of shape (num_bodies+1, 6),
    replaying the head's Linear/LeakyReLU(0.2) stack.'''
    def layer_index(key):
        m = re.search(r'affine_out\.(\d+)\.', key)
        if m is None:
            raise ValueError(f'cannot parse the affine_out layer index from {key}')
        return int(m.group(1))

    weight_keys = sorted(
        [k for k in state_dict if 'affine_out' in k and k.endswith('weight')], key=layer_index)
    if not weight_keys:
        raise ValueError('no affine_out layers in the checkpoint: this model has no '
                         'conformation decoder')
    if 'principal_axes' not in state_dict:
        raise ValueError('no principal_axes in the checkpoint: this model was trained without '
                         'body masks, so there are no rigid bodies to report')
    num_bodies = int(state_dict['principal_axes'].shape[0])

    def affine(conf):
        h = torch.as_tensor(np.asarray(conf)).double().view(1, -1)
        for i, wk in enumerate(weight_keys):
            bk = wk[:-6] + 'bias'
            b = state_dict[bk].double() if bk in state_dict else 0.0
            h = h @ state_dict[wk].double().t() + b
            if i < len(weight_keys) - 1:
                h = torch.where(h > 0, h, 0.2 * h)  # LeakyReLU(0.2)
        return h.view(num_bodies + 1, 6).numpy()

    return affine, num_bodies


def quat_to_matrix(vec, w=QUAT_W):
    '''Rotation matrix from the decoder's [w, v] quaternion convention.'''
    q = np.asarray([w, *np.asarray(vec, dtype=np.float64)])
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def body_rotations(conf, affine, orient, num_bodies):
    '''Per-body rotation matrices for one conformation code, in the model's native frame.'''
    params = affine(conf)
    orient_t = np.transpose(orient, (0, 2, 1))
    return np.stack([orient_t[i] @ quat_to_matrix(params[i, :3]) @ orient[i]
                     for i in range(num_bodies)])


def rotation_angle(r):
    '''Rotation angle in degrees.'''
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def rotation_axis(r):
    '''Unit rotation axis, right-handed, from the skew-symmetric part.

    Returns None for a rotation too small for the axis to be determined: the skew part scales
    with sin(angle), so at very small angles its direction is numerical noise.'''
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def flip_axis(axis):
    '''Map a rotation axis (a pseudovector) into the z-flipped frame.'''
    return None if axis is None else np.array([-axis[0], -axis[1], axis[2]])


def flip_vector(v):
    '''Map an ordinary vector into the z-flipped frame.'''
    return np.array([v[0], v[1], -v[2]])


def _axis_report(r, com_vector=None, flip=False):
    angle = rotation_angle(r)
    axis = rotation_axis(r)
    out = {'angle_deg': angle}  # type: dict
    if axis is None:
        out['axis'] = None
        out['note'] = 'rotation too small for the axis to be determined'
        return out
    if flip:
        axis = flip_axis(axis)
    out['axis'] = axis.tolist()
    if com_vector is not None:
        u = com_vector / np.linalg.norm(com_vector)
        if flip:
            u = flip_vector(u)
            u = u / np.linalg.norm(u)
        dot = float(axis @ u)
        out['dot_with_body_axis'] = dot
        out['angle_to_body_axis_deg'] = float(np.degrees(np.arccos(np.clip(abs(dot), -1, 1))))
        if abs(dot) < MIN_AXIS_PROJECTION:
            out['sense'] = ('undetermined: the axis is nearly perpendicular to body0->body1, '
                            'so the handedness about it is not meaningful')
        else:
            out['sense'] = ('right-handed about body0->body1' if dot > 0
                            else 'left-handed about body0->body1')
    return out


def analyse(conf_class, conf_global, state_dict, flip=False):
    '''Rigid-body readout for one conformation class against the global mean.

    Reports each body's rotation, the inter-body relative rotation in both states, and the
    change from global to class, which is the quantity a reader interprets as the motion.'''
    affine, num_bodies = load_affine_head(state_dict)
    orient = state_dict['orient_bodies'].double().numpy()
    com = state_dict['com_bodies'].double().numpy() if 'com_bodies' in state_dict else None
    com_vector = (com[1] - com[0]) if (com is not None and len(com) >= 2) else None

    r_class = body_rotations(conf_class, affine, orient, num_bodies)
    r_global = body_rotations(conf_global, affine, orient, num_bodies)

    out = {
        'num_bodies': num_bodies,
        'flip': bool(flip),
        'per_body': [
            {'body': i,
             'global_angle_deg': rotation_angle(r_global[i]),
             'class_angle_deg': rotation_angle(r_class[i])}
            for i in range(num_bodies)],
    }
    if com is not None:
        out['com_bodies'] = com.tolist()

    if num_bodies >= 2:
        # Body 0 expressed in body 1's frame, i.e. body 1 held fixed.
        rel_class = r_class[0] @ r_class[1].T
        rel_global = r_global[0] @ r_global[1].T
        out['inter_body_global'] = _axis_report(rel_global, com_vector, flip)
        out['inter_body_class'] = _axis_report(rel_class, com_vector, flip)
        out['class_vs_global'] = _axis_report(rel_class @ rel_global.T, com_vector, flip)
    else:
        out['note'] = ('only one body: inter-body motion is undefined and the per-body '
                       'rotations are relative to the zero conformation code')
    return out
