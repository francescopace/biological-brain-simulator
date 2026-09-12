"""STDP amplitude sweeps must isolate plasticity and preserve checkpoint semantics."""

import copy
from dataclasses import replace

import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, weight_summary
from examples.mnist_optimization_check import simulation_digest


@pytest.fixture
def small_builder(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4,
                        "INPUT_TO_CORTEX_DENSITY": 1.}.items():
        monkeypatch.setattr(mn, name, value)


def test_explicit_default_preserves_initial_state_and_rng(small_builder):
    assert simulation_digest(mn.build_brain(seed=5)) == simulation_digest(mn.build_brain(seed=5, stdp_scale=.2))


@pytest.mark.parametrize("exc_only", [False, True])
def test_scale_changes_only_exc_feedforward_stdp_amplitudes(small_builder, exc_only):
    original = mn.build_brain(seed=5, feedforward_exc_only=exc_only)
    changed = mn.build_brain(seed=5, feedforward_exc_only=exc_only, stdp_scale=2.)
    left, right = (b.get_projection("input", "cortex") for b in (original, changed))
    ns = right.n_synapses
    exc = changed.regions["cortex"].neuron_type[right.syn_post[:ns].long()] == mn.NeuronType.EXCITATORY.value
    assert torch.all(right.syn_A_plus[:ns][exc] == torch.tensor(.01 * 2.))
    assert torch.all(right.syn_A_minus[:ns][exc] == torch.tensor(.0105 * 2.))
    assert torch.count_nonzero(right.syn_A_plus[:ns][~exc]) == 0
    assert torch.count_nonzero(right.syn_A_minus[:ns][~exc]) == 0
    right.syn_A_plus.copy_(left.syn_A_plus)
    right.syn_A_minus.copy_(left.syn_A_minus)
    assert simulation_digest(original) == simulation_digest(changed)


@pytest.mark.parametrize("scale", [-1., float("nan"), float("inf")])
def test_invalid_amplitude_is_rejected(scale):
    with pytest.raises(ValueError, match="STDP scale"):
        mn.build_brain(stdp_scale=scale)
    with pytest.raises(ValueError, match="STDP scale"):
        replace(LearningConfig(), stdp_scale=scale).validate()


def test_zero_amplitude_keeps_plasticity_enabled_but_cannot_move_valid_weights(small_builder):
    brain = mn.build_brain(seed=5, stdp_scale=0.)
    projection = brain.get_projection("input", "cortex")
    before = projection.syn_weight.clone()
    assert projection.plasticity_enabled
    for region in brain.regions.values():
        region.fired[:region.n_neurons] = True
        region.last_spike_time[:region.n_neurons] = 0.
    brain.time = 1.
    mn.apply_feedforward_stdp(brain)
    assert torch.equal(projection.syn_weight, before)


def test_bound_summary_ignores_dead_synapses_and_preserves_state(small_builder):
    brain = mn.build_brain(seed=5)
    projection = brain.get_projection("input", "cortex")
    reference = copy.deepcopy(projection)
    projection.syn_weight[0] = projection.syn_min_weight[0]
    projection.syn_weight[1] = projection.syn_max_weight[1]
    projection.syn_alive[2] = False
    reference.syn_alive[2] = False
    before = simulation_digest(brain)
    result = weight_summary(projection, reference)
    assert result["live_synapses"] == projection.n_synapses - 1
    assert result["at_min"] == result["at_max"] == 1
    assert result["changed_weights"] == 2
    assert result["relative_l1_change_from_reference"] > 0
    assert simulation_digest(brain) == before
    reference.syn_pre[0] += 1
    with pytest.raises(ValueError, match="identical live topology"):
        weight_summary(projection, reference)


def test_weight_summary_handles_empty_projection(small_builder):
    projection = mn.build_brain(seed=5).get_projection("input", "cortex")
    projection.n_synapses = 0
    result = weight_summary(projection, projection)
    assert result["live_synapses"] == result["at_min"] == result["at_max"] == 0
    assert result["min"] is result["max"] is result["relative_l1_change_from_reference"] is None
