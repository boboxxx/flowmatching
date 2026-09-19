import torch

from flowharq.model import FlowHARQJSCC
from ugp.channel import (
    ChannelContext,
    ChannelObservation,
    TimeVaryingRayleigh,
    temporal_correlation,
)


def test_temporal_correlation_static_vehicle_is_one():
    speed = torch.tensor([0.0, 0.0])
    age = torch.tensor([0.0, 5.0])
    assert torch.allclose(temporal_correlation(speed, age), torch.ones(2), atol=1e-4)


def test_channel_shape_and_gradients():
    tokens = torch.randn(2, 16, 8, requires_grad=True)
    power = torch.ones(2, 16)
    speed = torch.tensor([60.0, 120.0])
    age = torch.tensor([1.0, 2.0])
    context = ChannelContext(torch.tensor([5.0, 10.0]), speed, age, temporal_correlation(speed, age))
    output = TimeVaryingRayleigh()(tokens, power, context)
    assert output.shape == tokens.shape
    output.square().mean().backward()
    assert torch.isfinite(tokens.grad).all()


def test_receiver_state_and_temporally_correlated_second_round():
    torch.manual_seed(7)
    tokens = torch.randn(2, 4, 8)
    power = torch.ones(2, 4)
    speed = torch.tensor([0.0, 120.0])
    age = torch.ones(2)
    context = ChannelContext(torch.tensor([5.0, 5.0]), speed, age, temporal_correlation(speed, age))
    channel = TimeVaryingRayleigh()
    first = channel.forward_with_state(tokens, power, context)
    second = channel.forward_with_state(tokens, power, context, previous_gain=first.gain)
    assert first.tokens.shape == tokens.shape
    assert first.receiver_features(context).shape == (2, 6)
    assert torch.isfinite(first.receiver_features(context)).all()
    # A static link is nearly unchanged; a high-speed link evolves across the HARQ gap.
    assert (second.gain[0] - first.gain[0]).abs().item() < 0.05
    assert (second.gain[1] - first.gain[1]).abs().item() > 0.05


def test_mrc_uses_instantaneous_channel_power():
    first = ChannelObservation(
        tokens=torch.ones(1, 2, 4),
        gain=torch.ones(1, 1, 1, dtype=torch.complex64),
        gain_sq=torch.ones(1, 1, 1),
        effective_snr_db=torch.zeros(1),
        equalized_noise_variance=torch.ones(1, 1, 1),
    )
    second = ChannelObservation(
        tokens=torch.full((1, 2, 4), 3.0),
        gain=torch.full((1, 1, 1), 2.0 + 0.0j),
        gain_sq=torch.full((1, 1, 1), 4.0),
        effective_snr_db=torch.zeros(1),
        equalized_noise_variance=torch.ones(1, 1, 1),
    )
    combined = FlowHARQJSCC.combine(first, second)
    assert torch.allclose(combined, torch.full_like(combined, 2.6))
