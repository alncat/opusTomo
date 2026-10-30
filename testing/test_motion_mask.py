import argparse
import importlib
import pickle
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cryodrgn.models import ALIGN_CORNERS, HetOnlyVAE


def make_model(**kwargs):
    # Small parameter-volume decoder exercises the VAE -> decoder forwarding
    # without allocating the convolutional template network.
    return HetOnlyVAE(
        SimpleNamespace(D=17), 0, 8, 0, 8, 289, zdim=2,
        encode_mode='fixed', enc_type='vanilla', downfrac=1.,
        window_r=1., templateres=16, **kwargs,
    )


@pytest.mark.parametrize('mode,expected_x', [('follow', 9), ('fixed', 8)])
def test_mask_body_motion_and_global_pose(mode, expected_x):
    decoder = make_model(motion_mask=mode).decoder
    decoder.scale = decoder.transformer.scale
    ref_mask = torch.zeros(1, 1, 16, 16, 16)
    ref_mask[0, 0, 6, 7, 11] = 1.
    decoder.register_buffer('ref_mask', ref_mask)
    # A 90-degree global rotation and one-voxel global translation should
    # place the fixed mask at (z,y,x)=(6,10,8). Body motion adds one x voxel.
    rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rot = rot.reshape(1, 1, 1, 3, 3).requires_grad_()
    step = 2. / 15
    body_trans = torch.tensor([[0., step, 0.], [0., step, 0.]], requires_grad=True)
    affine_grid, _, _ = decoder.transformer.multi_body_grid(
        rot, torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[-.2, 0., 0.], [.2, 0., 0.]]),
        torch.zeros(1, 3), body_trans,
        radius=torch.full((2, 3), .2), axes=torch.eye(3).repeat(2, 1, 1),
    )
    trans = torch.tensor([[step * decoder.vol_size / decoder.scale, 0., 0.]],
                         requires_grad=True)
    mask = decoder.sample_motion_mask(affine_grid, rot, trans)
    expected = torch.zeros_like(mask)
    expected[0, 0, 6, 10, expected_x] = 1.
    torch.testing.assert_close(mask, expected, atol=3e-6, rtol=0.)
    assert not mask.requires_grad
    if mode == 'follow':
        # Preserve the previous sampling operation exactly by default.
        old = F.grid_sample(ref_mask, (affine_grid + trans / decoder.vol_size * decoder.scale).detach(),
                            align_corners=ALIGN_CORNERS)
        assert torch.equal(mask, old)
    else:
        unchanged = decoder.sample_motion_mask(affine_grid + .3, rot, trans)
        assert torch.equal(mask, unchanged)


def test_motion_mask_default_and_validation():
    assert make_model().decoder.motion_mask == 'follow'
    with pytest.raises(ValueError, match='motion_mask'):
        make_model(motion_mask='unknown')


@pytest.mark.parametrize('command', ['train_tomo', 'train_tomo_dist', 'train_tomo_hvd', 'train_multi'])
@pytest.mark.parametrize('mode', [None, 'follow', 'fixed'])
def test_cli_saves_motion_mask_config(command, mode, tmp_path):
    module = importlib.import_module('cryodrgn.commands.' + command)
    parser = argparse.ArgumentParser()
    module.add_args(parser)
    argv = ['particles.star', '-o', str(tmp_path), '--poses', 'poses.pkl']
    if command == 'train_tomo_dist':
        argv += ['--split', 'split.pkl']
    if mode is not None:
        argv += ['--motion-mask', mode]
    args = parser.parse_args(argv)
    args.real_data = True  # normally assigned during dataset setup
    model = make_model(motion_mask=args.motion_mask)
    cfg_path = tmp_path / 'config.pkl'
    module.save_config(args, SimpleNamespace(norm=(0., 1.)),
                       SimpleNamespace(D=17, extent=.5, ignore_DC=True), model, cfg_path)
    with cfg_path.open('rb') as stream:
        cfg = pickle.load(stream)
    assert cfg['model_args']['motion_mask'] == (mode or 'follow')
    assert model.decoder.motion_mask == cfg['model_args']['motion_mask']
