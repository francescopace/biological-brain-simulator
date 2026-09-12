"""Synthetic tests for evaluation-only checkpoint interventions."""

import copy

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_state_diagnosis import (
    MODES, audit_sleep_weights, intervention, network_digest, reset_transients, run_variant,
)
from src.brain import Brain
from src.neuron import NeuronType
from src.region import RegionType
from src.persistence import save_brain


@pytest.fixture
def diagnostic_brain():
    brain = Brain(seed=42)
    for name in ("input", "cortex"):
        region = brain.add_region(name, RegionType.SENSORY, n_neurons=0, max_neurons=3)
        for _ in range(3):
            region.add_neuron(NeuronType.EXCITATORY)
        region.add_one_synapse(0, 1)
        region.theta[:3] = 0.5
        region.v[:3] = -50.0
        region.current[:3] = 10.0
        region.spike_buffer.fill_(1.0)
    brain.connect_regions("input", "cortex", density=1.0)
    brain.disable_oscillations()
    return brain


def test_transient_reset_preserves_weights_topology_theta_and_clock(diagnostic_brain):
    brain = diagnostic_brain
    before = network_digest(brain)
    brain.time, brain.step_count = 17.0, 17
    for target in [*brain.regions.values(), *brain.projections]:
        target.syn_resource[:target.n_synapses] = 0.2
        target.syn_facilitation[:target.n_synapses] = 0.5
    reset_transients(brain)
    assert network_digest(brain) == before
    assert (brain.time, brain.step_count) == (17.0, 17)
    for region in brain.regions.values():
        assert torch.all(region.theta[:3] == 0.5)
        assert torch.all(region.v[:3] == -65.0)
        assert torch.equal(region.u[:3], region.b[:3] * -65.0)
        assert not torch.any(region.spike_buffer)
        assert not torch.any(region.current)
    for target in [*brain.regions.values(), *brain.projections]:
        assert torch.all(target.syn_resource[:target.n_synapses] == 1.0)
        assert torch.all(target.syn_facilitation[:target.n_synapses] == 0.0)


@pytest.mark.parametrize("mode", MODES)
def test_interventions_are_copy_only_and_restore_hooks(diagnostic_brain, mode):
    brain = diagnostic_brain
    before = copy.deepcopy(brain)
    snapshot_hook, present_hook = mn._inference_brain, mn.present_sample
    with intervention(mode, rest_steps=3):
        snapshot = mn._inference_brain(brain)
        assert snapshot is not brain
        assert not snapshot.homeostasis.theta_enabled
        assert not snapshot.homeostasis.scaling_enabled
        assert not snapshot.reward_stdp.enabled
        assert not snapshot.memory.enabled
        assert network_digest(snapshot) == network_digest(brain)
        with pytest.raises(ValueError, match="must not train"):
            mn.present_sample(snapshot, np.zeros(3), 1, learn=True)
    assert mn._inference_brain is snapshot_hook
    assert mn.present_sample is present_hook
    assert brain.time == before.time
    for name, region in brain.regions.items():
        for attr in ("v", "u", "theta", "current", "spike_buffer"):
            assert torch.equal(getattr(region, attr), getattr(before.regions[name], attr))


def test_intervention_restores_hooks_after_exception(diagnostic_brain):
    hook = mn._inference_brain
    with pytest.raises(RuntimeError, match="interrupted"):
        with intervention("baseline"):
            mn._inference_brain(diagnostic_brain)
            raise RuntimeError("interrupted")
    assert mn._inference_brain is hook


def test_independent_evaluation_is_order_invariant(diagnostic_brain, monkeypatch):
    monkeypatch.setattr(mn, "ASSIGN_PRESENT_STEPS", 5)
    monkeypatch.setattr(mn, "TEST_PRESENT_STEPS", 5)
    monkeypatch.setattr(mn, "TEST_REPEATS", 2)
    monkeypatch.setattr(mn, "REST_STEPS", 1)
    X = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    y = np.array([0, 1, 2])
    result = run_variant(diagnostic_brain, "independent_samples", X, y, X, y, reverse=True)
    assert result["order_disagreements"] == 0
    assert result["reverse_accuracy"] == result["accuracy"]
    assert result["weights_and_topology_unchanged"]


def test_sleep_audit_isolates_scaling_without_changing_checkpoint(diagnostic_brain, tmp_path):
    brain = diagnostic_brain
    brain.disable_reward_modulated_plasticity()
    brain.freeze_structural_plasticity()
    brain.homeostasis.check_interval = 1
    path = tmp_path / "checkpoint"
    save_brain(brain, path)
    result = audit_sleep_weights(path, rest_steps=3)
    assert result["checkpoint_unchanged"]
    free, fixed = result["cases"]
    assert any(item["changed"] for item in free["weight_changes"])
    assert all(item["changed"] == 0 for item in fixed["weight_changes"])
