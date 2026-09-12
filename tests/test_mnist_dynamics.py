"""Circuit filtering and numerical interventions must not silently rewire controls."""

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_dynamics_check import run_case, variant
from examples.mnist_learning_check import LearningConfig, prepare_dataset
from examples.mnist_optimization_check import simulation_digest
from src.brain import Brain
from src.neuron import NeuronType
from src.region import RegionType


def mixed_brain():
    brain = Brain(seed=12)
    for name in ("input", "cortex"):
        region = brain.add_region(name, RegionType.ASSOCIATION, n_neurons=0, max_neurons=4)
        for cell_type in (NeuronType.INHIBITORY, NeuronType.EXCITATORY,
                          NeuronType.INHIBITORY, NeuronType.EXCITATORY):
            region.add_neuron(cell_type)
    return brain


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("cell_type", [NeuronType.EXCITATORY, NeuronType.INHIBITORY])
def test_connection_filter_selects_actual_noncontiguous_live_target_indices(bidirectional, cell_type):
    brain = mixed_brain()
    brain.regions["cortex"].neuron_alive[3] = False
    brain.connect_regions("input", "cortex", density=1., bidirectional=bidirectional,
                          target_neuron_type=cell_type)
    assert len(brain.projections) == 1 + bidirectional
    for proj in brain.projections:
        target = brain.regions[proj.target_name]
        posts = proj.syn_post[:proj.n_synapses].long()
        assert proj.n_synapses > 0
        assert (target.neuron_type[posts] == cell_type).all()
        assert target.neuron_alive[posts].all()
        expected_posts = torch.where((target.neuron_type[:4] == cell_type) & target.neuron_alive[:4])[0]
        assert torch.equal(torch.unique(posts), expected_posts)


def test_empty_filtered_target_still_connects_reverse_direction():
    brain = mixed_brain()
    brain.regions["cortex"].neuron_type[:4] = NeuronType.EXCITATORY
    count = brain.connect_regions("input", "cortex", density=1., bidirectional=True,
                                  target_neuron_type=NeuronType.INHIBITORY)
    assert brain.projections[0].n_synapses == 0
    assert brain.projections[1].n_synapses == 8
    assert count == 8


def test_default_connection_generation_preserves_old_rng_and_weights():
    brain = mixed_brain()
    generator = torch.Generator(device=brain._rng.device)
    generator.set_state(brain._rng.get_state())
    conn = torch.rand((2, 4), device=brain.regions["input"].v.device, generator=generator) < .7
    pre, post = torch.where(conn)
    weights = -torch.log(torch.rand(len(pre), device=conn.device, generator=generator)) * .3
    brain.connect_regions("input", "cortex", density=.7)
    proj = brain.projections[0]
    assert torch.equal(proj.syn_pre[:len(pre)], torch.tensor([1, 3], device=conn.device)[pre].int())
    assert torch.equal(proj.syn_post[:len(post)], post.int())
    assert torch.equal(proj.syn_weight[:len(pre)], weights.clamp(0., 10.))


def test_mnist_default_allocates_only_excitatory_feedforward_edges(monkeypatch):
    for name, value in {"N_INPUT": 3, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4,
                        "INPUT_TO_CORTEX_DENSITY": 1.}.items():
        monkeypatch.setattr(mn, name, value)
    new = mn.build_brain(seed=7)
    old = mn.build_brain(seed=7, integration_method="legacy_euler", feedforward_exc_only=False)
    proj = new.get_projection("input", "cortex")
    assert proj.n_synapses == 12
    assert old.get_projection("input", "cortex").n_synapses == 24
    assert (new.regions["cortex"].neuron_type[proj.syn_post[:12].long()] == NeuronType.EXCITATORY).all()
    assert all(r.integration_method == "heun" for r in new.regions.values())
    assert all(r.integration_method == "legacy_euler" for r in old.regions.values())


def test_factorial_intervention_changes_only_requested_mode_and_inhibitory_liveness():
    source = mixed_brain()
    source.connect_regions("input", "cortex", density=1.)
    before = simulation_digest(source)
    changed = variant(source, "heun", True, .05)
    assert simulation_digest(source) == before
    for original, target in zip([*source.regions.values(), *source.projections],
                                [*changed.regions.values(), *changed.projections]):
        for key, value in vars(original).items():
            if isinstance(value, torch.Tensor):
                if original in source.projections and key == "syn_alive":
                    n = original.n_synapses
                    expected = value.clone()
                    expected[:n] &= source.regions["cortex"].neuron_type[original.syn_post[:n].long()] == 0
                    assert torch.equal(expected, target.syn_alive)
                else:
                    assert torch.equal(value, getattr(target, key)), key
    assert all(r.integration_max_step == .05 for r in changed.regions.values())


def test_small_factorial_case_preserves_source_and_caches_responses(monkeypatch, tmp_path):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 5, "N_CORTEX_INH": 5,
                        "INPUT_TO_CORTEX_DENSITY": 1., "CLASSES": (0, 1)}.items():
        monkeypatch.setattr(mn, name, value)
    source = mn.build_brain(seed=4, integration_method="legacy_euler", feedforward_exc_only=False)
    config = LearningConfig(train_per_class=2, readout_per_class=2, validation_per_class=2,
                            inference_steps=20)
    raw = np.random.default_rng(4).integers(0, 256, size=(20, 4))
    data = prepare_dataset(raw, np.arange(20) % 2, config, classes=(0, 1), train_boundary=20)
    before = simulation_digest(source)
    result = run_case(source, data, config, tmp_path, method="heun", exc_only=True,
                      max_step=.1, probe_samples=1)
    assert result["live_feedforward"] == 20
    assert (tmp_path / "responses.npz").exists()
    assert len(result["numerical_probes"]) == 1
    assert simulation_digest(source) == before
