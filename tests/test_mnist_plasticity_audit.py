"""Separate-arm diagnostics must reproduce the actual plasticity trajectory."""

import copy

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_plasticity_audit import audit, replay_arms
from src.synaptic_events import SynapseEventIndex


@pytest.fixture
def brain(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4,
                        "INPUT_TO_CORTEX_DENSITY": 1.}.items():
        monkeypatch.setattr(mn, name, value)
    return mn.build_brain(seed=5, exc_to_inh_weight=64., stdp_scale=2.)


@pytest.mark.parametrize("at_bound", [False, True])
@pytest.mark.parametrize("pre_fired,post_fired", [(False, False), (False, True), (True, False), (True, True)])
def test_arm_replay_is_exact_and_preserves_the_source(brain, at_bound, pre_fired, post_fired):
    projection = brain.get_projection("input", "cortex")
    ns = projection.n_synapses
    if at_bound:
        projection.syn_weight[:ns:2] = projection.syn_min_weight[:ns:2]
        projection.syn_weight[1:ns:2] = projection.syn_max_weight[1:ns:2]
    projection.syn_alive[0] = False
    brain.time = 10.
    for name, fired in (("input", pre_fired), ("cortex", post_fired)):
        region = brain.regions[name]
        region.fired[:region.n_neurons] = fired
        region.last_spike_time[:region.n_neurons] = 10. if fired else 9.
    before = projection.syn_weight[:ns].clone()
    source = simulation_digest(brain)
    ltp, combined = replay_arms(brain, before)
    assert simulation_digest(brain) == source
    assert torch.equal(before, projection.syn_weight[:ns])
    assert torch.all(ltp >= before)
    assert torch.all(combined <= ltp)
    mn.apply_feedforward_stdp(brain)
    assert torch.equal(combined, projection.syn_weight[:ns])


def test_replay_cross_checks_sparse_event_indices(brain, monkeypatch):
    monkeypatch.setattr(SynapseEventIndex, "min_synapses", 1)
    projection = brain.get_projection("input", "cortex")
    brain.time = 10.
    for region in brain.regions.values():
        region.fired[:region.n_neurons] = False
        region.last_spike_time[:region.n_neurons] = 9.
    brain.regions["cortex"].fired[0] = True
    before = projection.syn_weight[:projection.n_synapses].clone()
    _, combined = replay_arms(brain, before)
    mn.apply_feedforward_stdp(brain)
    assert projection._post_events._endpoints is not None
    assert torch.equal(combined, projection.syn_weight[:projection.n_synapses])


def test_disabled_projection_has_no_arm_updates(brain):
    projection = brain.get_projection("input", "cortex")
    projection.plasticity_enabled = False
    for region in brain.regions.values():
        region.fired[:region.n_neurons] = True
        region.last_spike_time[:region.n_neurons] = brain.time
    before = projection.syn_weight[:projection.n_synapses].clone()
    assert all(torch.equal(before, output) for output in replay_arms(brain, before))


@pytest.mark.parametrize("per_neuron", [False, True])
def test_diagnostic_preserves_the_complete_training_trajectory(brain, per_neuron):
    brain.regions["cortex"].v[0] = 35.  # Exercise real updates even in this tiny circuit.
    images = np.random.default_rng(3).uniform(size=(3, 4))
    source = simulation_digest(brain)
    rows, final = audit(brain, images, train_steps=20, rest_steps=5, per_neuron=per_neuron)
    assert simulation_digest(brain) == source
    reference = copy.deepcopy(brain)
    target = mn.compute_norm_target(reference, per_neuron=per_neuron)
    for image in images:
        mn.present_sample(reference, image, 20, learn=True, collect_responses=False)
        mn.normalize_feedforward_weights(reference, target)
        mn.reset_brain_state(reference, rest_steps=5)
    assert final == simulation_digest(reference)
    assert len(rows) == 3
    assert sum(r["ltp_l1"] + r["ltd_l1"] for r in rows) > 0
