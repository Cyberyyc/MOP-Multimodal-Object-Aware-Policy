import io
import numpy as np
import pytest
import torch
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from mop.geometry import lift_tracks_to_3d, object_position_from_tracks
from mop.vision import SpatialSoftmax, G2MPEncoder
from mop.policy import MOPPolicy


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(2)


def test_spatial_keypoints_retain_pixel_locations_and_gradients():
    features = torch.zeros(1, 1, 3, 3, requires_grad=True)
    peaked = features + torch.tensor([[[[0., 0., 0.], [0., 0., 0.], [0., 0., 30.]]]])
    result = SpatialSoftmax()(peaked)
    torch.testing.assert_close(result, torch.tensor([[[1., 1.]]]), atol=1e-5, rtol=0)
    result.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_full_modality_dropout_preserves_geometry_and_state():
    encoder = G2MPEncoder(5, feature_dim=32, state_feature_dim=16, modality_dropout=1.)
    encoder.train()
    cloud, state = torch.randn(2, 20, 3), torch.randn(2, 5)
    output_a = encoder(cloud, torch.rand(2, 3, 64, 64), state)
    output_b = encoder(cloud, torch.rand(2, 3, 64, 64), state)
    torch.testing.assert_close(output_a, output_b)


def make_policy_and_batch():
    policy = MOPPolicy(state_dim=5, action_dim=4, feature_dim=32, state_feature_dim=16,
                       transformer_width=32, latent_dim=8, transformer_layers=1,
                       down_dims=(32, 64, 128), inference_steps=2)
    batch = {'obs': {'point_cloud': torch.randn(2, 2, 20, 3),
                     'agent_pos': torch.randn(2, 2, 5),
                     'rgb': torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)},
             'action': torch.randn(2, 16, 4), 'object_position': torch.randn(2, 16, 3)}
    normalizer = LinearNormalizer()
    normalizer.fit({'point_cloud': batch['obs']['point_cloud'],
                    'agent_pos': batch['obs']['agent_pos'], 'action': batch['action']})
    policy.set_normalizer(normalizer)
    return policy, batch


def test_joint_objective_backpropagates_through_all_three_modules():
    policy, batch = make_policy_and_batch()
    loss, metrics = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in [policy.encoder.visual_projection.weight, policy.encoder.alpha,
                      policy.opp.output.weight, policy.opp.posterior_parameters.weight,
                      policy.conditioner.beta, policy.model.final_conv[-1].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert metrics['kl'] >= 0 and metrics['diffusion'] >= 0


def test_padding_is_masked_and_future_labels_are_not_required_at_inference():
    policy, batch = make_policy_and_batch()
    batch['object_valid'] = torch.ones(2, 16, dtype=torch.bool)
    batch['object_valid'][:, -4:] = False
    batch['object_position'][:, -4:] = float('nan')
    batch['action_valid'] = torch.ones(2, 16, dtype=torch.bool)
    batch['action_valid'][:, -3:] = False
    batch['action'][:, -3:] = float('nan')
    loss, _ = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    policy.eval()
    result = policy.predict_action(batch['obs'], generator=torch.Generator().manual_seed(11))
    assert result['action'].shape == (2, 8, 4)
    assert result['action_pred'].shape == (2, 16, 4)
    assert result['object_position'].shape == (2, 16, 3)
    assert all(torch.isfinite(value).all() for value in result.values())


def test_checkpoint_restores_normalization_and_seeded_inference():
    policy, batch = make_policy_and_batch()
    policy.eval()
    stream = io.BytesIO()
    torch.save(policy.state_dict(), stream)
    stream.seek(0)
    restored, _ = make_policy_and_batch()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.eval()
    a = policy.predict_action(batch['obs'], generator=torch.Generator().manual_seed(42))
    b = restored.predict_action(batch['obs'], generator=torch.Generator().manual_seed(42))
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_rgbd_tracking_geometry_and_visibility():
    pixels = np.array([[[1., 2.], [3., 0.], [99., 1.], [np.nan, 0.]]])
    depth = np.full((1, 4, 4), 2.)
    transform = np.eye(4)
    transform[:3, 3] = [10, 0, 0]
    xyz, valid = lift_tracks_to_3d(pixels, depth, np.eye(3), camera_to_world=transform)
    np.testing.assert_allclose(xyz[0, 0], [12, 4, 2])
    np.testing.assert_array_equal(valid, [[True, True, False, False]])
    center, visible = object_position_from_tracks(xyz, valid)
    np.testing.assert_allclose(center, [[14, 2, 2]])
    np.testing.assert_array_equal(visible, [True])


def test_unobserved_object_trajectory_is_rejected():
    policy, batch = make_policy_and_batch()
    batch['object_valid'] = torch.zeros(2, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match='at least one valid'):
        policy.compute_loss(batch)
