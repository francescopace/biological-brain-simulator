"""Saved-state interventions isolate weights and theta without training."""

import copy

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_state_factorial import factorial_variant, response_summary


@pytest.fixture
def donors(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4,
                        "INPUT_TO_CORTEX_DENSITY": 1.0}.items():
        monkeypatch.setattr(mn, name, value)
    initial = mn.build_brain(seed=3)
    weights, theta = copy.deepcopy(initial), copy.deepcopy(initial)
    weights.get_projection("input", "cortex").syn_weight *= 0.5
    # Donor transients and non-selected factors must not leak into the variant.
    weights.regions["cortex"].theta.fill_(12.0)
    theta.get_projection("input", "cortex").syn_weight *= 0.1
    for region in theta.regions.values():
        region.theta.fill_(0.25)
        region.v.fill_(-12.0)
    return initial, weights, theta


def test_variant_changes_only_selected_weights_and_thresholds(donors):
    initial, weights, theta = donors
    before = [simulation_digest(b) for b in donors]
    model = factorial_variant(*donors)
    expected = copy.deepcopy(initial)
    expected.get_projection("input", "cortex").syn_weight.copy_(
        weights.get_projection("input", "cortex").syn_weight)
    for name, region in expected.regions.items():
        region.theta.copy_(theta.regions[name].theta)
    assert simulation_digest(model) == simulation_digest(expected)
    assert [simulation_digest(b) for b in donors] == before
    model.regions["cortex"].theta[0] += 1
    assert [simulation_digest(b) for b in donors] == before


@pytest.mark.parametrize("attr", ["syn_pre", "syn_delay", "syn_alive", "syn_modulation"])
def test_variant_rejects_unpaired_topology(donors, attr):
    initial, weights, theta = donors
    array = getattr(weights.get_projection("input", "cortex"), attr)
    array[0] = not array[0] if attr == "syn_alive" else array[0] + 1
    with pytest.raises(ValueError, match="Factorial donors differ"):
        factorial_variant(initial, weights, theta)


def test_variant_rejects_changed_intrinsic_dynamics_and_internal_weights(donors):
    initial, weights, theta = donors
    weights.regions["cortex"].b[0] += 1
    with pytest.raises(ValueError, match="cortex.b"):
        factorial_variant(*donors)
    weights.regions["cortex"].b.copy_(initial.regions["cortex"].b)
    weights.regions["cortex"].syn_weight[0] += 1
    with pytest.raises(ValueError, match="Only input-to-cortex"):
        factorial_variant(*donors)


def test_initial_diagonal_preserves_full_state(donors):
    initial, *_ = donors
    assert simulation_digest(factorial_variant(initial, initial, initial)) == simulation_digest(initial)


def test_response_summary_counts_population_and_silence():
    responses = mn.ReadoutResponses(np.arange(3), np.array([[0, 0, 0], [2, 0, 1]]), np.zeros((2, 3)))
    assert response_summary(responses) == {"silent_samples": 1, "mean_spikes_per_sample": 1.5,
                                           "mean_active_neurons_per_sample": 1.0,
                                           "active_neurons_across_samples": 2}
