import pytest
import torch

from cryodrgn.models import SpatialTransformer


def test_narrow_blend_keeps_near_body_motion_when_gaussians_are_tiny():
    transformer = SpatialTransformer(32, render_size=32)
    identity = torch.eye(3)
    grid, _, _ = transformer.multi_body_grid(
        rot_ori=identity.reshape(1, 1, 1, 3, 3),
        rot_resi=identity.repeat(2, 1, 1),
        coms=torch.tensor([[-0.625, 0., 0.], [0.625, 0., 0.]]),
        trans=torch.zeros(1, 3),
        tbody=torch.tensor([[0., 0., 0.], [0.1, 0., 0.]]),
        radius=torch.full((2, 3), 1.5 / 32),
        axes=identity.repeat(2, 1, 1),
    )

    # This point is much nearer the moving body. Its motion should remain 0.1
    # even though both unnormalized Gaussian weights are below 1e-5.
    offset = grid[0, 16, 16, 19, 0] - transformer.grid[0, 16, 16, 19, 0]
    assert offset.item() == pytest.approx(0.1, abs=0.01)


def test_resume_rejects_a_changed_blend_radius():
    from cryodrgn.models import validate_blend_radius_on_resume

    decoder = torch.nn.Module()
    decoder.register_buffer("radius", torch.full((2, 3), 0.5))
    with pytest.raises(ValueError, match="--blend-radius"):
        validate_blend_radius_on_resume(decoder, {"radius": torch.full((2, 3), 0.25)})


def test_resume_accepts_the_checkpoint_blend_radius():
    from cryodrgn.models import validate_blend_radius_on_resume

    decoder = torch.nn.Module()
    decoder.register_buffer("radius", torch.full((2, 3), 0.5000001))
    validate_blend_radius_on_resume(decoder, {"radius": torch.full((2, 3), 0.5)})
    assert torch.equal(decoder.radius, torch.full((2, 3), 0.5))
