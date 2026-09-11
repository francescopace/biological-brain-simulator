"""Regression tests for benchmark protocol integrity."""

import numpy as np
import pytest
import torch

import examples.iris_benchmark as iris_benchmark
from examples.grid_nav_benchmark import build_brain as build_grid_brain
from examples.grid_nav_benchmark import apply_reinforcement, choose_action, position_encode
from examples.iris_benchmark import normalize_features
from examples.mnist_benchmark import downsample_images, l1_equalization_target
from src.brain import Brain
from src.region import RegionType


def test_grid_policy_prefers_motor_spikes_without_oracle_decoder():
    action, source = choose_action(
        counts=np.array([0, 3, 1, 0]),
        mean_voltage=np.array([-60.0, -70.0, -50.0, -65.0]),
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    assert action == 1
    assert source == "policy"


def test_grid_policy_uses_voltage_only_when_silent():
    action, source = choose_action(
        counts=np.zeros(4, dtype=int),
        mean_voltage=np.array([-60.0, -70.0, -50.0, -65.0]),
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    assert action == 2
    assert source == "voltage"


def test_grid_uses_minimal_one_hot_state_action_baseline():
    encoded = position_encode((2, 3))
    assert encoded.sum() == pytest.approx(1.0)
    assert encoded[13] == pytest.approx(1.0)

    brain = build_grid_brain(seed=42)
    assert set(brain.regions) == {"input", "motor"}
    assert len(brain.projections) == 1
    assert brain.projections[0].source_name == "input"
    assert brain.projections[0].target_name == "motor"
    assert brain.homeostasis.scaling_enabled is False
    assert brain.memory.enabled is False
    assert brain.oscillators.enabled is False

    proj = brain.get_projection("input", "motor")
    pairs = set(zip(
        proj.syn_pre[:proj.n_synapses].tolist(),
        proj.syn_post[:proj.n_synapses].tolist(),
    ))
    assert pairs == {(state, action) for state in range(25) for action in range(4)}


@pytest.mark.parametrize("benchmark", ["grid", "iris"])
@pytest.mark.parametrize("positive", [True, False])
def test_reinforcement_ignores_new_spikes_from_unselected_action(benchmark, positive):
    # Synthetic two-neuron pathways exercise the actual reward helpers,
    # without data loading, rollouts or benchmark training.
    brain = Brain(seed=42)
    for name in ("input", "motor"):
        region = brain.add_region(name, RegionType.SENSORY, n_neurons=0, max_neurons=2)
        for _ in range(2):
            region.add_neuron()
    brain.connect_regions("input", "motor", density=1.0)
    brain.freeze_structural_plasticity()
    brain.freeze_homeostatic_scaling()
    brain.disable_memory()
    brain.disable_oscillations()
    proj = brain.get_projection("input", "motor")
    ns = proj.n_synapses
    proj.syn_weight[:ns] = 0.5
    proj.syn_eligibility[:ns] = 0.2
    brain.regions["input"].last_spike_time[:2] = 0.0
    # The unselected motor fires during the reward window, after masking.
    brain.regions["motor"].v[1] = 30.0
    before = proj.syn_weight[:ns].clone()
    target = Brain.projection_target("input", "motor")

    if benchmark == "grid":
        apply_reinforcement(brain, [target], 0.1, positive=positive, action=0)
    else:
        iris_benchmark._reinforce_motor(brain, 0, 0.1, positive=positive)

    selected = proj.syn_post[:ns] == 0
    delta = proj.syn_weight[:ns] - before
    assert brain.regions["motor"].total_spikes[1].item() > 0
    assert torch.all(delta[selected] > 0 if positive else delta[selected] < 0)
    assert torch.equal(proj.syn_weight[:ns][~selected], before[~selected])
    assert torch.all(proj.syn_eligibility[:ns][~selected] == 0)
    assert brain.dopamine(target) == 0.0


def test_iris_normalization_uses_supplied_training_bounds():
    train = np.array([[0.0, 10.0], [2.0, 20.0]])
    test = np.array([[3.0, 5.0]])
    normalized = normalize_features(
        test,
        lo=train.min(axis=0),
        hi=train.max(axis=0),
    )
    np.testing.assert_allclose(normalized, [[1.5, -0.5]])


def test_iris_reports_spike_silence_separately_from_voltage_fallback(monkeypatch):
    monkeypatch.setattr(
        iris_benchmark,
        "present",
        lambda brain, x: (
            np.zeros(3, dtype=int),
            np.array([-60.0, -55.0, -65.0]),
        ),
    )
    monkeypatch.setattr(iris_benchmark, "reset_between_samples", lambda brain: None)

    spike_pred, fallback_pred, counts = iris_benchmark.test_one_sample(
        object(), np.zeros(iris_benchmark.N_INPUT), test_repeats=1,
    )

    assert spike_pred == -1
    assert fallback_pred == 1
    assert counts.sum() == 0


def test_mnist_equalization_can_reuse_training_target():
    train = np.zeros((2, 28 * 28), dtype=np.float64)
    train[0, :20] = 1.0
    train[1, :40] = 1.0
    test = np.zeros((1, 28 * 28), dtype=np.float64)
    test[0, :10] = 1.0

    target = l1_equalization_target(train)
    transformed = downsample_images(test, target_l1=target)

    assert target == pytest.approx(30.0)
    # Clipping is part of the encoder contract and remains bounded.
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0
