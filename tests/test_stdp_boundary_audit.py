"""The diagnostic censors timestamps only on its private counterfactual weights."""

import numpy as np
import pytest
import torch

from examples.mnist_boundary_audit import audit, censored_update
import examples.mnist_benchmark as mn
from examples.mnist_optimization_check import simulation_digest
from src.brain import Brain
from src.neuron import NeuronType
from src.region import RegionType


def pair():
    brain = Brain(seed=3)
    for name in ("input", "cortex"):
        region = brain.add_region(name, RegionType.SENSORY, 0, max_neurons=1)
        region.add_neuron(NeuronType.EXCITATORY)
    brain.connect_regions("input", "cortex", density=1.)
    brain.disable_oscillations()
    brain.disable_memory()
    brain.disable_reward_modulated_plasticity()
    brain.freeze_structural_plasticity()
    brain.freeze_homeostatic_scaling()
    brain.projections[0].syn_weight[0] = 1.
    brain.time = 10.
    return brain


@pytest.mark.parametrize("arm", ["ltp", "ltd"])
@pytest.mark.parametrize("timestamp,expected_censored", [(5., True), (8., True), (9., False)])
def test_censoring_uses_the_image_boundary_without_mutating_state(arm, timestamp, expected_censored):
    brain = pair()
    src, dst = brain.regions["input"], brain.regions["cortex"]
    if arm == "ltp":
        src.last_spike_time[0], dst.last_spike_time[0] = timestamp, 10.
        dst.fired[0] = True
    else:
        src.last_spike_time[0], dst.last_spike_time[0] = 10., timestamp
        src.fired[0] = True
    weights = brain.projections[0].syn_weight[:1].clone()
    original = simulation_digest(brain)
    private = censored_update(brain, weights, image_start=8.)
    assert simulation_digest(brain) == original
    assert torch.equal(weights, torch.ones_like(weights))
    mn.apply_feedforward_stdp(brain)
    actual = brain.projections[0].syn_weight[:1]
    assert not torch.equal(actual, weights)
    assert (float(actual[0]) > 1.) == (arm == "ltp")
    assert torch.equal(private, weights if expected_censored else actual)


def test_audit_preserves_its_source_network_and_reports_zero_for_initial_empty_history():
    brain = pair()
    before = simulation_digest(brain)
    rows = audit(brain, np.ones((2, 1)), train_steps=30, rest_steps=5)
    assert simulation_digest(brain) == before
    assert len(rows) == 2
    assert rows[0]["pre_image_history_difference_l1"] == 0.
    assert rows[0]["changed_weight_step_pairs"] == 0
    assert all(0 <= r["steps_with_history_effect"] <= 30 for r in rows)
