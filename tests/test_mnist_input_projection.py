"""Feedforward topology calibration must be explicit and resume-safe."""

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, prepare_dataset, train_condition
from examples.mnist_optimization_check import simulation_digest


@pytest.fixture
def small_builder(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 8, "N_CORTEX_INH": 8}.items():
        monkeypatch.setattr(mn, name, value)


def test_explicit_defaults_preserve_the_complete_initial_state_and_rng(small_builder):
    assert simulation_digest(mn.build_brain(seed=5)) == simulation_digest(
        mn.build_brain(seed=5, input_density=.15, input_weight_boost=4.))


def test_omitted_parameters_still_resolve_benchmark_globals_at_call_time(small_builder, monkeypatch):
    monkeypatch.setattr(mn, "INPUT_TO_CORTEX_DENSITY", 1.)
    monkeypatch.setattr(mn, "INPUT_WEIGHT_BOOST", .6)
    assert simulation_digest(mn.build_brain(seed=5)) == simulation_digest(
        mn.build_brain(seed=5, input_density=1., input_weight_boost=.6))


@pytest.mark.parametrize("exc_only", [False, True])
def test_dense_projection_contains_every_allowed_pair_exactly_once(small_builder, exc_only):
    brain = mn.build_brain(seed=5, input_density=1., input_weight_boost=.6,
                           stdp_scale=.03, feedforward_exc_only=exc_only)
    projection = brain.get_projection("input", "cortex")
    n_target = 8 if exc_only else 16
    ns = projection.n_synapses
    assert ns == 4 * n_target
    edges = projection.syn_pre[:ns].long() * n_target + projection.syn_post[:ns].long()
    assert torch.equal(torch.sort(edges).values.cpu(), torch.arange(4 * n_target))
    assert bool(projection.syn_alive[:ns].all())
    types = brain.regions["cortex"].neuron_type[projection.syn_post[:ns].long()]
    exc = types == mn.NeuronType.EXCITATORY.value
    assert torch.all(projection.syn_A_plus[:ns][exc] == torch.tensor(.01 * .03))
    assert torch.all(projection.syn_A_minus[:ns][exc] == torch.tensor(.0105 * .03))
    assert torch.count_nonzero(projection.syn_A_plus[:ns][~exc]) == 0
    assert torch.count_nonzero(projection.syn_A_minus[:ns][~exc]) == 0


def test_boost_changes_only_bounded_feedforward_weights(small_builder):
    original = mn.build_brain(seed=5, input_density=1., input_weight_boost=1.)
    changed = mn.build_brain(seed=5, input_density=1., input_weight_boost=.6)
    left, right = [model.get_projection("input", "cortex") for model in (original, changed)]
    ns = left.n_synapses
    expected = torch.clamp(left.syn_weight[:ns] * .6, left.syn_min_weight[:ns], left.syn_max_weight[:ns])
    assert torch.equal(right.syn_weight[:ns], expected)
    right.syn_weight.copy_(left.syn_weight)
    assert simulation_digest(original) == simulation_digest(changed)


def test_density_does_not_change_neuron_parameters_or_wta_wiring(small_builder):
    sparse, dense = [mn.build_brain(seed=5, input_density=density) for density in (.15, 1.)]
    assert sparse.get_projection("input", "cortex").n_synapses < 32
    for name in sparse.regions:
        for field, value in vars(sparse.regions[name]).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(dense.regions[name], field)), (name, field)
    assert torch.equal(sparse.encoder._rng.get_state(), dense.encoder._rng.get_state())


def test_zero_density_and_zero_boost_are_explicit_ablation_options(small_builder):
    empty = mn.build_brain(seed=5, input_density=0.)
    assert empty.get_projection("input", "cortex").n_synapses == 0
    zero = mn.build_brain(seed=5, input_density=1., input_weight_boost=0.)
    assert zero.get_projection("input", "cortex").n_synapses == 32
    assert torch.count_nonzero(zero.get_projection("input", "cortex").syn_weight) == 0


@pytest.mark.parametrize("field,value", [("input_density", -.01), ("input_density", 1.01),
    ("input_density", float("nan")), ("input_density", float("inf")),
    ("input_weight_boost", -.01), ("input_weight_boost", float("nan")),
    ("input_weight_boost", float("inf"))])
def test_invalid_projection_parameters_fail_before_brain_construction(monkeypatch, field, value):
    monkeypatch.setattr(mn, "Brain", lambda *args, **kwargs: pytest.fail("brain allocated"))
    with pytest.raises(ValueError, match="Input"):
        mn.build_brain(**{field: value})
    with pytest.raises(ValueError, match="Input"):
        replace(LearningConfig(), **{field: value}).validate()


@pytest.mark.parametrize("learning_rule", ["pair", "post_trace"])
def test_dense_sample_boundary_resume_matches_uninterrupted_training(small_builder, tmp_path, learning_rule):
    config = LearningConfig(train_per_class=3, readout_per_class=2, validation_per_class=2,
        epochs=2, train_steps=35, rest_steps=3, checkpoint_every=3, input_density=1.,
        input_weight_boost=.6, stdp_scale=.03, learning_rule=learning_rule)
    raw = np.random.default_rng(42).integers(0, 256, size=(40, 4))
    data = prepare_dataset(raw, np.arange(40) % 2, config, classes=(0, 1), train_boundary=30)
    brain = mn.build_brain(seed=5, input_density=1., input_weight_boost=.6, stdp_scale=.03)
    brain.regions["cortex"].v[0] = 35.
    full, complete = train_condition(copy.deepcopy(brain), data, config, 5, "stdp_normalized",
                                     tmp_path / "full", {})
    train_condition(copy.deepcopy(brain), data, config, 5, "stdp_normalized",
                     tmp_path / "split", {}, stop_after=5)
    checkpoint = tmp_path / "split/sample_00000005"
    resumed, continued = train_condition(copy.deepcopy(brain), data, config, 5, "stdp_normalized",
                                         tmp_path / "split", {}, resume_from=checkpoint)
    assert simulation_digest(resumed) == simulation_digest(full)
    assert continued["weight_deltas"] == complete["weight_deltas"]
    assert continued["norm_target"] == complete["norm_target"]
    for changes in ({"input_density": .15}, {"input_weight_boost": 4.}):
        with pytest.raises(ValueError, match="protocol/data mismatch"):
            train_condition(copy.deepcopy(brain), data, replace(config, **changes), 5, "stdp_normalized",
                             tmp_path / "split", {}, resume_from=checkpoint)


def test_cli_records_and_uses_dense_projection_parameters(small_builder, tmp_path, monkeypatch):
    import examples.mnist_learning_check as study
    raw = np.random.default_rng(1).integers(0, 256, size=(60020, 4))
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: SimpleNamespace(
        data=raw, target=np.arange(60020) % 10))
    output = tmp_path / "dense"
    options = ["--output", str(output), "--seeds", "5", "--train-per-class", "1",
        "--readout-per-class", "1", "--validation-per-class", "1", "--train-steps", "2",
        "--inference-steps", "2", "--rest-steps", "0", "--input-density", "1",
        "--input-weight-boost", ".6", "--stdp-scale", ".03"]
    monkeypatch.setattr("sys.argv", ["learning_check", *options])
    study.main()
    result = json.loads((output / "summary.json").read_text())
    assert result["complete"] and result["source_unchanged"]
    assert result["signature"]["config"]["input_density"] == 1.
    assert result["signature"]["config"]["input_weight_boost"] == .6
    initial = next(row for row in result["results"] if row["condition"] == "initial")
    assert initial["feedforward_weights"]["live_synapses"] == 32
    assert initial["final_sha256"] == simulation_digest(mn.build_brain(
        seed=5, input_density=1., input_weight_boost=.6, stdp_scale=.03))
    monkeypatch.setattr("sys.argv", ["learning_check", *options, "--resume", "--input-density", ".15"])
    with pytest.raises(ValueError, match="Cannot resume with changed code"):
        study.main()
