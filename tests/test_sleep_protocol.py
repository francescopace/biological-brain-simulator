"""Fast synthetic checks for the controlled sleep/replay protocol."""

import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import examples.sleep_benchmark as sleep
from src.brain import Brain
from src.memory import MemoryTrace
from src.neuron import NeuronType
from src.region import RegionType


def sleep_brain():
    brain = Brain(seed=42)
    for name in ("input", "cortex"):
        region = brain.add_region(name, RegionType.SENSORY, n_neurons=0, max_neurons=6)
        for _ in range(3):
            region.add_neuron(NeuronType.EXCITATORY)
        region.add_one_synapse(0, 1)
        region.theta[:3] = 0.2
    brain.connect_regions("input", "cortex", density=1.0)
    brain.disable_oscillations()
    brain.homeostasis.check_interval = 1
    brain.growth.growth_interval = 1
    brain.growth.neurogenesis_threshold = -1.0
    brain.memory.consolidation_interval = 1
    brain.memory.replay_strength = 80.0
    for name in brain.regions:
        brain.memory.traces.append(MemoryTrace(
            neuron_indices=torch.arange(3), activity_snapshot=torch.ones(3),
            region_name=name, creation_time=0.0,
        ))
    return brain


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("stdp", [False, True])
def test_sleep_only_updates_feedforward_weights_when_stdp_enabled(replay, stdp):
    brain = sleep_brain()
    original = copy.deepcopy(brain)
    result = sleep.sleep_phase(brain, 12, replay, enable_stdp=stdp)
    assert result["stdp_steps"] == (12 if stdp else 0)
    assert bool(result["replayed_traces"]) == replay
    for name, region in brain.regions.items():
        assert region.n_neurons == original.regions[name].n_neurons
        assert torch.equal(region.theta, original.regions[name].theta)
        assert result["weight_changes"][f"region:{name}"]["changed"] == 0
    updates = result["weight_changes"]["proj:input->cortex"]["changed"]
    if not stdp:
        assert updates == 0
    if replay and stdp:
        assert updates > 0  # Actual replay-driven spikes reach manual STDP.
    assert len(brain.memory.traces) == 2
    assert all(trace.creation_time == 0.0 for trace in brain.memory.traces)
    assert brain.homeostasis.scaling_enabled
    assert brain.homeostasis.theta_enabled
    assert brain.metaplasticity_enabled
    assert brain.growth.growth_interval == 1
    assert brain.memory.enabled
    assert brain.reward_stdp.enabled


def test_sleep_restores_controls_on_exception(monkeypatch):
    brain = sleep_brain()
    brain.homeostasis.theta_enabled = False
    brain.memory.enabled = False
    original = (True, False, True, 1, False, True)
    monkeypatch.setattr(brain, "step", Mock(side_effect=RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        sleep.sleep_phase(brain, 1, enable_stdp=False)
    assert (
        brain.homeostasis.scaling_enabled, brain.homeostasis.theta_enabled,
        brain.metaplasticity_enabled, brain.growth.growth_interval,
        brain.memory.enabled, brain.reward_stdp.enabled,
    ) == original


def test_four_arms_reuse_one_decoder_and_identical_source(monkeypatch):
    brain = sleep_brain()
    before = copy.deepcopy(brain)
    spikes, voltage = np.ones((2, 3)), np.ones((2, 3))
    readout = Mock(return_value=(None, None, spikes, voltage))
    evaluation = Mock(return_value=(0.5, None))
    monkeypatch.setattr(sleep, "build_readout", readout)
    monkeypatch.setattr(sleep, "evaluate", evaluation)
    X, y = np.ones((2, 3)), np.array([0, 1])
    results = sleep.compare_sleep_conditions(brain, X, y, X, y, 3)
    assert readout.call_count == 1
    assert evaluation.call_count == 5
    assert {(r["replay"], r["stdp"]) for r in results["conditions"]} == {
        (True, True), (True, False), (False, True), (False, False),
    }
    evaluated_models = []
    for call in evaluation.call_args_list:
        assert call.args[3] is spikes
        assert call.args[4] is voltage
        evaluated_models.append(call.args[0])
    assert len({id(model) for model in evaluated_models}) == 5
    assert brain.time == before.time
    assert brain.memory.traces[0].replay_count == 0
    assert torch.equal(brain.projections[0].syn_weight, before.projections[0].syn_weight)
