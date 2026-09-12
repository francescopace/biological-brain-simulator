"""Equivalence and operation-count checks for benchmark fast paths."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_state_diagnosis import run_variant
from src.region import Region


@pytest.fixture
def small_mnist(monkeypatch):
    for name, value in {
        "N_INPUT": 6, "N_CORTEX_EXC": 8, "N_CORTEX_INH": 8,
        "INPUT_TO_CORTEX_DENSITY": 0.6,
        "ASSIGN_PRESENT_STEPS": 25, "TEST_PRESENT_STEPS": 30,
        "REST_STEPS": 7,
    }.items():
        monkeypatch.setattr(mn, name, value)
    brain = mn.build_brain(seed=123)
    for region in brain.regions.values():
        region.theta[:region.n_neurons] = 0.3
    return brain


def assert_same_simulation(left, right):
    assert left.time == right.time
    assert left.step_count == right.step_count
    assert torch.equal(left._rng.get_state(), right._rng.get_state())
    assert torch.equal(left.encoder._rng.get_state(), right.encoder._rng.get_state())
    targets_left = [*left.regions.values(), *left.projections]
    targets_right = [*right.regions.values(), *right.projections]
    for a, b in zip(targets_left, targets_right):
        for key, value in vars(a).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(b, key)), key
        if hasattr(a, "_rng"):
            assert torch.equal(a._rng.get_state(), b._rng.get_state())


@pytest.mark.parametrize("repeats", [1, 2, 3, 5])
@pytest.mark.parametrize("score_weights", [(0.7, 0.3), (1.0, 0.0), (0.0, 1.0)])
def test_fast_inference_matches_independent_reference_exactly(
    small_mnist, monkeypatch, repeats, score_weights,
):
    monkeypatch.setattr(mn, "TEST_REPEATS", repeats)
    monkeypatch.setattr(mn, "SPIKE_SCORE_WEIGHT", score_weights[0])
    monkeypatch.setattr(mn, "VOLTAGE_SCORE_WEIGHT", score_weights[1])
    X = np.random.default_rng(5).random((5, 6))
    y = np.array([0, 1, 0, 1, 0])
    source = copy.deepcopy(small_mnist)
    outputs = []
    steps = []
    for fast in (False, True):
        monkeypatch.setattr(mn, "FAST_INDEPENDENT_INFERENCE", fast)
        snapshots = []
        snapshot = mn._inference_brain

        def capture(brain):
            frozen = snapshot(brain)
            snapshots.append(frozen)
            return frozen

        with patch.object(mn, "_inference_brain", capture):
            readout = mn.build_readout(small_mnist, X, y, (0, 1))
        frozen = capture(small_mnist)
        predictions = [mn.predict_sample(frozen, x, *readout[2:], (0, 1)) for x in X]
        outputs.append((readout, predictions))
        steps.append(sum(b.step_count - small_mnist.step_count for b in snapshots))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        np.testing.assert_array_equal(a, b)
    for (label_a, score_a), (label_b, score_b) in zip(outputs[0][1], outputs[1][1]):
        assert label_a == label_b
        np.testing.assert_array_equal(score_a, score_b)
    assert steps == [5 * (2 * (25 + 7) + repeats * (30 + 7)), 5 * (25 + 30)]
    assert_same_simulation(small_mnist, source)


@pytest.mark.parametrize("flag", ["noise", "theta", "scaling", "memory", "reward",
                                  "growth", "plasticity", "metaplasticity", "oscillations", "legacy"])
def test_shortcuts_require_frozen_deterministic_independent_state(small_mnist, monkeypatch, flag):
    brain = mn._inference_brain(small_mnist)
    assert mn.can_reuse_inference(brain)
    if flag == "noise":
        brain.encoder.noise_level = 0.01
    elif flag == "theta":
        brain.enable_adaptive_thresholds()
    elif flag == "scaling":
        brain.enable_homeostatic_scaling()
    elif flag == "memory":
        brain.enable_memory()
    elif flag == "reward":
        brain.enable_reward_modulated_plasticity()
    elif flag == "growth":
        brain.growth.growth_interval = 500
    elif flag == "plasticity":
        brain.projections[0].plasticity_enabled = True
    elif flag == "metaplasticity":
        brain.metaplasticity_enabled = True
    elif flag == "oscillations":
        brain.enable_oscillations()
    else:
        monkeypatch.setattr(mn, "INDEPENDENT_INFERENCE", False)
    assert not mn.can_reuse_inference(brain)


def test_nonpositive_repeat_count_is_rejected(small_mnist, monkeypatch):
    monkeypatch.setattr(mn, "TEST_REPEATS", 0)
    with pytest.raises(ValueError, match="TEST_REPEATS must be positive"):
        mn.predict_sample(mn._inference_brain(small_mnist), np.ones(6),
                          np.ones((2, 8)), np.ones((2, 8)), (0, 1))


def test_legacy_protocol_keeps_both_passes_repeats_and_rests(small_mnist, monkeypatch):
    monkeypatch.setattr(mn, "INDEPENDENT_INFERENCE", False)
    monkeypatch.setattr(mn, "TEST_REPEATS", 3)
    X = np.ones((2, 6))
    snapshots = []
    original = mn._inference_brain

    def capture(brain):
        frozen = original(brain)
        snapshots.append(frozen)
        return frozen

    monkeypatch.setattr(mn, "_inference_brain", capture)
    readout = mn.build_readout(small_mnist, X, np.array([0, 1]), (0, 1))
    mn.evaluate(small_mnist, X, np.array([0, 1]), *readout[2:], (0, 1))
    assert snapshots[0].step_count == 2 * 2 * (25 + 7)
    assert snapshots[1].step_count == 2 * 3 * (30 + 7)


def test_training_fast_path_matches_original_loop_including_rng(small_mnist, monkeypatch):
    reference, optimized = copy.deepcopy(small_mnist), copy.deepcopy(small_mnist)
    X = np.random.default_rng(0).random((4, 6))
    target = mn.compute_norm_target(reference)
    # Original loop: convert numpy input on each step, collect unused responses.
    for epoch in range(2):
        for idx in np.random.default_rng(42 + epoch).permutation(len(X)):
            cortex = reference.regions["cortex"]
            exc_idx = mn.excitatory_cortex_indices(reference)
            voltage_sum = torch.zeros(len(exc_idx), device=cortex.v.device)
            for _ in range(30):
                reference.stimulate("input", X[idx])
                reference.step()
                voltage_sum += cortex.v[exc_idx]
                mn.apply_feedforward_stdp(reference)
            mn.normalize_feedforward_weights(reference, target)
            mn.reset_brain_state(reference)
    # The training-only path must not even look up readout neuron indices.
    original_indices = mn.excitatory_cortex_indices
    calls = []

    def indices(brain):
        calls.append(1)
        return original_indices(brain)

    monkeypatch.setattr(mn, "excitatory_cortex_indices", indices)
    assert mn.train_unsupervised(optimized, X, epochs=2, train_present_steps=30, log_every=0) == target
    assert len(calls) == 1  # Only compute_norm_target.
    assert_same_simulation(reference, optimized)
    assert not torch.equal(optimized.projections[0].syn_weight, small_mnist.projections[0].syn_weight)


def test_disabled_memory_skips_activity_reduction_but_enabled_memory_captures(small_mnist, monkeypatch):
    brain = small_mnist
    reads = []

    def activity(region):
        reads.append(region.name)
        return 0.2

    monkeypatch.setattr(Region, "mean_activity", property(activity))
    brain.step()
    assert reads == []
    brain.enable_memory()
    for region in brain.regions.values():
        region.activity[:region.n_neurons] = 1.0
    brain.step()
    assert reads == ["input", "cortex"]
    assert len(brain.memory.traces) == 2


def test_standard_diagnostic_accounts_for_reused_repeats(small_mnist, monkeypatch):
    monkeypatch.setattr(mn, "CLASSES", (0, 1))
    monkeypatch.setattr(mn, "TEST_REPEATS", 3)
    X, y = np.ones((2, 6)), np.array([0, 1])
    outputs = []
    for fast in (False, True):
        monkeypatch.setattr(mn, "FAST_INDEPENDENT_INFERENCE", fast)
        outputs.append(run_variant(small_mnist, "standard", X, y, X, y, reverse=True))
    assert outputs[0]["simulated_repeats"] == 3
    assert outputs[1]["simulated_repeats"] == 1
    for key in ("accuracy", "predictions", "spike_only_accuracy", "voltage_only_accuracy",
                "silent_samples", "mean_spikes_per_sample", "order_disagreements"):
        assert outputs[0][key] == outputs[1][key]
