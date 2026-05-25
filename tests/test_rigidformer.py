import torch

import pytest
param = pytest.mark.parametrize

@param('fps', (False, True))
def test_rigidformer(
    fps
):
    from rigidformer.rigidformer import Rigidformer, RigidformerRolloutWrapper

    object_pos = torch.randn(2, 2, 256, 3)
    object_pos_prev = torch.randn(2, 2, 256, 3)
    object_pos_next = torch.randn(2, 2, 256, 3)
    vertex_properties = torch.randn(2, 2, 3)

    anchor_indices = torch.randint(0, 256, (2, 2, 4))

    from einops.layers.torch import Reduce

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(
        512,
        hierarchical_encoder = Reduce('b no n d -> b no d', 'mean') # mock before building out pointnet++ and platonic transformer
    )

    kwargs = dict()
    if not fps:
        kwargs.update(anchor_indices = anchor_indices)

    loss, loss_breakdown = rigidformer(
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_pos = object_pos,
        object_pos_prev = object_pos_prev,
        object_pos_next = object_pos_next,
        **kwargs
    )

    loss.backward()

    rollout_wrapper = RigidformerRolloutWrapper(rigidformer)

    object_positions = rollout_wrapper(
        num_steps = 4,
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_positions = [object_pos_prev, object_pos],
        **kwargs
    )

    assert len(object_positions) == 6

    last_position = object_positions[-1]

    assert last_position.shape == (2, 2, 256, 3)
