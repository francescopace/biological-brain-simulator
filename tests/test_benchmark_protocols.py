"""Protocol regressions using tiny networks or mocked training only."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import examples.degradation_benchmark as degradation
import examples.forgetting_benchmark as forgetting
import examples.mnist_benchmark as mnist
import examples.mnist_diagnosis as diagnosis
from src.brain import Brain
from src.device import DEVICE
from src.neuron import FiringPattern, NeuronType
from src.region import Region, RegionType


def tiny_brain():
    brain = Brain(seed=42)
    sensory = brain.add_region(
        "input", RegionType.SENSORY, n_neurons=0, max_neurons=2,
    )
    cortex = brain.add_region(
        "cortex", RegionType.ASSOCIATION, n_neurons=0, max_neurons=8,
    )
    for _ in range(2):
        sensory.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    for neuron_type in (NeuronType.EXCITATORY, NeuronType.INHIBITORY,
                        NeuronType.EXCITATORY):
        cortex.add_neuron(neuron_type, FiringPattern.REGULAR_SPIKING)
    brain.connect_regions("input", "cortex", density=1.0)
    brain.disable_oscillations()
    return brain


def test_degradation_handles_zero_damage_and_kills_only_excitatory_neurons():
    brain = tiny_brain()
    cortex = brain.regions["cortex"]
    original_alive = cortex.neuron_alive.clone()
    degradation.snn_damage(brain, 0.0)
    assert torch.equal(cortex.neuron_alive, original_alive)

    degradation.snn_damage(brain, 1.0)
    assert cortex.neuron_alive[:3].tolist() == [False, True, False]
    projection = brain.projections[0]
    n = projection.n_synapses
    assert torch.equal(
        projection.syn_alive[:n], projection.syn_post[:n] == 1,
    )


def test_mnist_reset_resolves_patched_rest_at_call_time(monkeypatch):
    brain = SimpleNamespace(regions={}, reset_traces=Mock())
    quiet = Mock()
    monkeypatch.setattr(mnist, "quiet_steps", quiet)
    original_rest = mnist.REST_STEPS
    with diagnosis.patched_benchmark_globals(replace(diagnosis.FAST_BASE, rest_steps=3)):
        mnist.reset_brain_state(brain)
        quiet.assert_called_with(brain, 3)
        mnist.reset_brain_state(brain, rest_steps=0)
        quiet.assert_called_with(brain, 0)
    mnist.reset_brain_state(brain)
    quiet.assert_called_with(brain, original_rest)


def test_competition_sweep_runs_distinct_representable_inhibition(monkeypatch):
    effective_weights = []

    def fake_training(config, label):
        cortex = Region("cortex", RegionType.ASSOCIATION, max_neurons=4)
        for neuron_type in (NeuronType.EXCITATORY, NeuronType.EXCITATORY,
                            NeuronType.INHIBITORY, NeuronType.INHIBITORY):
            cortex.add_neuron(neuron_type, FiringPattern.REGULAR_SPIKING)
        with diagnosis.patched_benchmark_globals(config):
            mnist.wire_cortex_microcircuit(cortex)
        weights = cortex.syn_weight[:cortex.n_synapses]
        actual_weight = float(-weights[weights < 0][0])
        assert actual_weight == pytest.approx(config.inh_lateral_weight)
        effective_weights.append(actual_weight)
        return "unused", None, None, None, None

    monkeypatch.setattr(diagnosis, "train_protocol", fake_training)
    monkeypatch.setattr(diagnosis, "evaluate_protocol", lambda *args: {
        "accuracy": 0.0, "label_entropy": 0.0, "max_class_share": 0.0,
    })
    diagnosis.run_competition_sweep()
    assert len(effective_weights) > 1
    assert len(set(effective_weights)) == len(effective_weights)


def test_readout_and_evaluation_freeze_learning_and_topology_on_copies(monkeypatch):
    brain = tiny_brain()
    brain.growth.growth_interval = 1
    brain.growth.neurogenesis_threshold = -1.0
    brain.growth.max_new_neurons_per_cycle = 1
    brain.homeostasis.check_interval = 1
    for region in brain.regions.values():
        region.activity[:region.n_neurons] = 10.0
    projection = brain.projections[0]
    projection.syn_eligibility[:projection.n_synapses] = 1.0
    brain.reward(1.0, brain.projection_target("input", "cortex"))
    original = copy.deepcopy(brain)
    snapshots = []
    make_snapshot = mnist._inference_brain

    def capture_snapshot(model):
        snapshot = make_snapshot(model)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(mnist, "_inference_brain", capture_snapshot)
    monkeypatch.setattr(mnist, "ASSIGN_PRESENT_STEPS", 2)
    monkeypatch.setattr(mnist, "TEST_PRESENT_STEPS", 2)
    monkeypatch.setattr(mnist, "TEST_REPEATS", 1)
    monkeypatch.setattr(mnist, "REST_STEPS", 1)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1])
    indices, labels, spikes, voltages = mnist.build_readout(brain, X, y, (0, 1))
    _, predictions = mnist.evaluate(brain, X, y, spikes, voltages, (0, 1))
    np.testing.assert_array_equal(indices, [0, 2])
    assert labels.shape == (2,)
    assert spikes.shape == voltages.shape == (2, 2)
    assert predictions.shape == (2,)
    for snapshot in snapshots:
        assert snapshot is not brain
        assert snapshot.regions["cortex"].n_neurons == 3
        assert not snapshot.growth.history
        assert snapshot.metaplasticity_enabled is False
        assert snapshot.homeostasis.theta_enabled is False
        assert snapshot.homeostasis.scaling_enabled is False
        assert snapshot.memory.enabled is False
        assert snapshot.reward_stdp.enabled is False
        assert torch.equal(snapshot.projections[0].syn_weight, projection.syn_weight)
    assert brain.time == original.time
    assert brain.growth.growth_interval == 1
    assert brain.homeostasis.theta_enabled is True
    assert brain.homeostasis.scaling_enabled is True
    assert brain.metaplasticity_enabled is True
    assert brain.reward_stdp.dopamine == original.reward_stdp.dopamine
    for name, region in brain.regions.items():
        saved = original.regions[name]
        assert region.n_neurons == saved.n_neurons
        for attr in ("neuron_alive", "v", "theta", "activity", "syn_weight"):
            assert torch.equal(getattr(region, attr), getattr(saved, attr))
    assert torch.equal(brain._rng.get_state(), original._rng.get_state())


def test_forgetting_reuses_pre_b_decoder_without_refitting_a(monkeypatch):
    model = SimpleNamespace(samples_seen=0)
    readout_calls = []
    evaluations = []

    def present(model, *args, **kwargs):
        model.samples_seen += 1

    def readout(model, X, y, classes):
        readout_calls.append((model.samples_seen, classes))
        templates = np.full((len(classes), 2), model.samples_seen, dtype=float)
        return None, None, templates, templates.copy()

    def evaluate(model, X, y, spikes, voltages, classes):
        evaluations.append((model.samples_seen, spikes, voltages, classes))
        return len(evaluations) / 10, None

    monkeypatch.setattr(forgetting, "build_brain", lambda **kwargs: model)
    monkeypatch.setattr(forgetting, "compute_norm_target", lambda model: 1.0)
    monkeypatch.setattr(forgetting, "present_sample", present)
    monkeypatch.setattr(forgetting, "normalize_feedforward_weights", lambda *args: None)
    monkeypatch.setattr(forgetting, "reset_brain_state", lambda *args: None)
    monkeypatch.setattr(forgetting, "build_readout_subset", lambda X, y, **kwargs: (X, y))
    monkeypatch.setattr(forgetting, "build_readout", readout)
    monkeypatch.setattr(forgetting, "evaluate", evaluate)
    X_a, X_b = np.zeros((2, 2)), np.ones((2, 2))
    y_a, y_b = np.array([0, 1]), np.array([5, 6])
    result = forgetting.snn_sequential(X_a, y_a, X_b, y_b, X_a, y_a, X_b, y_b)
    assert result == pytest.approx((0.1, 0.2, 0.3))
    assert readout_calls == [(2, forgetting.TASK_A_CLASSES), (4, forgetting.TASK_B_CLASSES)]
    assert evaluations[0][1] is evaluations[2][1]
    assert evaluations[0][2] is evaluations[2][2]
    assert evaluations[2][0] == 4


def test_forgetting_mlp_uses_same_task_candidates_as_snn():
    X = np.zeros((1, 1))
    W1 = torch.zeros((1, 1), device=DEVICE)
    b1 = torch.zeros(1, device=DEVICE)
    W2 = torch.zeros((1, 10), device=DEVICE)
    b2 = torch.zeros(10, device=DEVICE)
    b2[5] = 100.0
    b2[1] = 1.0
    accuracy, *_ = forgetting.mlp_train_eval(
        X, np.array([1]), X, np.array([1]), forgetting.TASK_A_CLASSES,
        W1=W1, b1=b1, W2=W2, b2=b2, epochs=0,
    )
    assert accuracy == 1.0
