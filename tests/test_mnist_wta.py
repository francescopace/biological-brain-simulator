"""Check actual inhibitory activation, not just the existence of WTA edges."""

import copy

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_wta_check import coupling_variant, image_activity, paired_impulse
from src.neuron import FiringPattern, NeuronType
from src.persistence import load_brain, save_brain
from src.region import Region, RegionType


@pytest.mark.parametrize("cap", [.1, .05])
@pytest.mark.parametrize("weight,expected", [(8., []), (24., [4]), (32., [3]), (64., [2])])
def test_single_exc_spike_recruits_partner_at_calibrated_coupling(cap, weight, expected):
    result = paired_impulse(weight, integration_max_step=cap)
    assert result["source_spike_steps"] == [1]
    assert result["partner_spike_steps"] == expected
    assert result["stored_weight"] == weight
    assert result["stored_max_weight"] == max(10., weight)


@pytest.mark.parametrize("cap", [.1, .05])
@pytest.mark.parametrize("resource,expected", [(.9, [2]), (.5, [3])])
def test_calibrated_pair_still_responds_with_depleted_resources(cap, resource, expected):
    assert paired_impulse(64., resource=resource, integration_max_step=cap)["partner_spike_steps"] == expected


@pytest.mark.parametrize("cap", [.1, .05])
def test_recruited_inhibitory_partner_suppresses_a_competing_spike(cap):
    counts = {}
    for lateral in (False, True):
        cortex = Region("cortex", RegionType.ASSOCIATION, max_neurons=4, integration_max_step=cap)
        for _ in range(2):
            cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
        for _ in range(2):
            cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
        mn.wire_cortex_microcircuit(cortex, exc_to_inh_weight=64.)
        if not lateral:
            cortex.syn_alive[:cortex.n_synapses][cortex.syn_weight[:cortex.n_synapses] < 0] = False
        cortex.v[0] = 35.
        for step in range(1, 16):
            if step == 2:
                cortex.current[1] = 20.
            cortex.step(float(step), step)
        counts[lateral] = cortex.total_spikes[:4].tolist()
    assert counts[False] == [1, 1, 1, 1]
    assert counts[True] == [1, 0, 1, 0]


@pytest.fixture
def tiny_brain(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4,
                        "INPUT_TO_CORTEX_DENSITY": 1.}.items():
        monkeypatch.setattr(mn, name, value)
    return mn.build_brain(seed=101)


def test_explicit_default_preserves_state_topology_and_rng(tiny_brain):
    assert simulation_digest(tiny_brain) == simulation_digest(mn.build_brain(seed=101, exc_to_inh_weight=8.))


def test_stronger_builder_changes_only_matched_weights_and_bounds(tiny_brain):
    before = simulation_digest(tiny_brain)
    variant = coupling_variant(tiny_brain, 64.)
    built = mn.build_brain(seed=101, exc_to_inh_weight=64.)
    assert simulation_digest(variant) == simulation_digest(built)
    assert simulation_digest(tiny_brain) == before
    cortex = built.regions["cortex"]
    assert torch.equal(cortex.syn_weight[:4], torch.full((4,), 64.))
    assert torch.equal(cortex.syn_max_weight[:4], torch.full((4,), 64.))
    assert torch.equal(cortex.syn_weight[4:], tiny_brain.regions["cortex"].syn_weight[4:])


def test_explicit_coupling_survives_save_and_load(tiny_brain, tmp_path):
    brain = coupling_variant(tiny_brain, 64.)
    save_brain(brain, tmp_path / "brain")
    loaded = load_brain(tmp_path / "brain")
    assert simulation_digest(loaded) == simulation_digest(brain)


def test_existing_internal_synapses_are_not_modified_by_wiring():
    cortex = Region("cortex", RegionType.ASSOCIATION, max_neurons=2)
    cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
    cortex.add_one_synapse(0, 1, 1.)
    mn.wire_cortex_microcircuit(cortex, exc_to_inh_weight=64.)
    assert cortex.syn_weight[:2].tolist() == [1., 64.]
    assert cortex.syn_max_weight[:2].tolist() == [10., 64.]


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_coupling_is_rejected_before_mutating_region(weight):
    cortex = Region("empty", RegionType.ASSOCIATION)
    with pytest.raises(ValueError, match="coupling"):
        mn.wire_cortex_microcircuit(cortex, exc_to_inh_weight=weight)
    assert cortex.n_synapses == 0
    with pytest.raises(ValueError, match="coupling"):
        LearningConfig(exc_to_inh_weight=weight).validate()


def test_activity_audit_does_not_mutate_source(tiny_brain):
    source = coupling_variant(tiny_brain, 64.)
    before = simulation_digest(source)
    rows = image_activity(source, np.ones((2, 4)), 50)
    assert rows[0] == rows[1]
    assert simulation_digest(source) == before


def test_variant_rejects_nonmatched_exc_to_inh_connections(tiny_brain):
    altered = copy.deepcopy(tiny_brain)
    altered.regions["cortex"].syn_post[0] = 5
    with pytest.raises(ValueError, match="matched-pair"):
        coupling_variant(altered, 64.)
