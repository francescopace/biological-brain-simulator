"""Train-only split, paired controls, cached readouts and checkpoint continuation."""

import copy
from dataclasses import replace
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import (
    LearningConfig, cached_responses, comparison_summary, prepare_dataset,
    probe_decoders, ridge_predictions, train_condition, training_protocol, weight_delta_metrics,
)
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import (
    array_digest, directory_digest, load_training_checkpoint, save_training_checkpoint,
)


@pytest.fixture
def learning_setup(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 8, "N_CORTEX_INH": 8,
                        "INPUT_TO_CORTEX_DENSITY": 0.8, "CLASSES": (0, 1)}.items():
        monkeypatch.setattr(mn, name, value)
    config = LearningConfig(train_per_class=3, readout_per_class=2, validation_per_class=2,
                            epochs=2, train_steps=35, inference_steps=25, rest_steps=3,
                            checkpoint_every=3)
    raw = np.random.default_rng(42).integers(0, 256, size=(40, 4))
    labels = np.arange(40) % 2
    data = prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=30)
    brain = mn.build_brain(seed=5)
    brain.regions["cortex"].v[0] = 35.0  # Ensure this tiny fixture exercises actual STDP.
    return brain, data, config, raw, labels


def test_split_excludes_canonical_test_and_fits_intensity_on_training_only(learning_setup):
    _, data, config, raw, labels = learning_setup
    train, validation = set(data.manifest["train_ids"]), set(data.manifest["validation_ids"])
    assert not train & validation
    assert max(train | validation) < 30
    assert set(data.manifest["readout_ids"]) <= train
    altered = raw.copy()
    altered[list(validation)] = 0
    altered[30:] = 255
    second = prepare_dataset(altered, labels, config, classes=(0, 1), train_boundary=30)
    assert second.manifest["intensity_target_l1"] == data.manifest["intensity_target_l1"]
    np.testing.assert_array_equal(second.train_X, data.train_X)
    assert np.all(second.validation_X == 0)


def test_split_rejects_missing_samples_and_invalid_readout_budget(learning_setup):
    _, _, config, raw, labels = learning_setup
    with pytest.raises(ValueError, match="Readout samples"):
        prepare_dataset(raw, labels, replace(config, readout_per_class=10), classes=(0, 1), train_boundary=30)
    with pytest.raises(ValueError, match="Not enough"):
        prepare_dataset(raw, labels, replace(config, train_per_class=20), classes=(0, 1), train_boundary=30)


def test_fresh_split_excludes_prior_rows_and_is_order_independent(learning_setup):
    _, original, config, raw, labels = learning_setup
    excluded = original.manifest["train_ids"] + original.manifest["validation_ids"]
    fresh = prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=30,
                            excluded_ids=excluded)
    again = prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=30,
                            excluded_ids=excluded[::-1] + excluded[:2])
    assert fresh.manifest == again.manifest
    assert fresh.manifest["excluded_ids"] == sorted(set(excluded))
    train, validation = set(fresh.manifest["train_ids"]), set(fresh.manifest["validation_ids"])
    assert not (train | validation) & set(excluded)
    assert not train & validation
    assert set(fresh.manifest["readout_ids"]) <= train
    np.testing.assert_array_equal(np.bincount(fresh.train_y), [3, 3])
    np.testing.assert_array_equal(np.bincount(fresh.validation_y), [2, 2])
    altered = raw.copy()
    altered[list(validation | set(excluded))] = 0
    changed = prepare_dataset(altered, labels, config, classes=(0, 1), train_boundary=30,
                              excluded_ids=excluded)
    assert changed.manifest["intensity_target_l1"] == fresh.manifest["intensity_target_l1"]
    np.testing.assert_array_equal(changed.train_X, fresh.train_X)


@pytest.mark.parametrize("excluded", [[-1], [30], [1.5], [True], ["1"], {"rows": [1]}])
def test_split_rejects_invalid_exclusions(learning_setup, excluded):
    _, _, config, raw, labels = learning_setup
    with pytest.raises(ValueError, match="Excluded rows"):
        prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=30,
                        excluded_ids=excluded)


def test_exclusions_cannot_silently_reduce_sample_budget(learning_setup):
    _, _, config, raw, labels = learning_setup
    with pytest.raises(ValueError, match="Not enough"):
        prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=30,
                        excluded_ids=list(range(25)))


@pytest.mark.parametrize("condition", ["normalization_only", "stdp_normalized"])
@pytest.mark.parametrize("normalization", ["population_mean", "initial_per_neuron"])
@pytest.mark.parametrize("stdp_scale", [.2, 2.])
@pytest.mark.parametrize("learning_rule", ["pair", "post_trace"])
def test_sample_boundary_resume_matches_uninterrupted_training(learning_setup, tmp_path, condition, normalization, stdp_scale, learning_rule):
    brain, data, config, *_ = learning_setup
    brain = mn.build_brain(seed=5, stdp_scale=stdp_scale)
    brain.regions["cortex"].v[0] = 35.
    config = replace(config, normalization_policy=normalization, stdp_scale=stdp_scale,
                     learning_rule=learning_rule)
    full, complete = train_condition(copy.deepcopy(brain), data, config, 5, condition,
                                     tmp_path / "full", {"test": "version1"})
    _, partial = train_condition(copy.deepcopy(brain), data, config, 5, condition,
                                 tmp_path / "split", {"test": "version1"}, stop_after=5)
    checkpoint = tmp_path / "split/sample_00000005"
    original_files = directory_digest(checkpoint)
    resumed, continued = train_condition(copy.deepcopy(brain), data, config, 5, condition,
                                         tmp_path / "split", {"test": "version1"}, resume_from=checkpoint)
    assert continued["completed_samples"] == 12
    assert partial["completed_samples"] == 5
    assert simulation_digest(full) == simulation_digest(resumed)
    assert complete["weight_deltas"] == continued["weight_deltas"]
    assert complete["norm_target"] == continued["norm_target"]
    assert original_files == directory_digest(checkpoint)
    if normalization == "initial_per_neuron":
        assert continued["norm_target"] == mn.compute_norm_target(brain, per_neuron=True).cpu().tolist()
    if condition == "normalization_only":
        assert all(row["raw_stdp_l1"] == 0 for row in complete["weight_deltas"])
    else:
        assert any(row["raw_stdp_l1"] > 0 for row in complete["weight_deltas"])


def test_standard_condition_matches_existing_training_loop(learning_setup, tmp_path):
    brain, data, config, *_ = learning_setup
    reference = copy.deepcopy(brain)
    with patch.object(mn, "REST_STEPS", config.rest_steps):
        target = mn.train_unsupervised(reference, data.train_X, epochs=config.epochs,
                                       train_present_steps=config.train_steps, seed=5, log_every=0)
    actual, progress = train_condition(brain, data, config, 5, "stdp_normalized", tmp_path, {})
    assert progress["norm_target"] == target
    assert simulation_digest(reference) == simulation_digest(actual)


def test_per_neuron_budget_preserves_initial_weights_before_learning(learning_setup):
    brain, *_ = learning_setup
    projection = brain.get_projection("input", "cortex")
    before = projection.syn_weight.clone()
    target = mn.compute_norm_target(brain, per_neuron=True)
    mn.normalize_feedforward_weights(brain, target)
    assert torch.equal(projection.syn_weight, before)
    projection.syn_weight[:projection.n_synapses] *= 0.5
    mn.normalize_feedforward_weights(brain, target.tolist())
    assert torch.equal(projection.syn_weight, before)


def test_scalar_normalization_preserves_original_arithmetic(learning_setup):
    brain, *_ = learning_setup
    projection = brain.get_projection("input", "cortex")
    ns = projection.n_synapses
    # Python-scalar/Tensor division uses a different floating-point operation
    # sequence from zero-dimensional Tensor/Tensor division in PyTorch.
    target = 3.141592653589793
    expected = projection.syn_weight.clone()
    post = projection.syn_post[:ns].long()
    cortex = brain.regions["cortex"]
    live_exc = ((cortex.neuron_type[post] == mn.NeuronType.EXCITATORY.value)
                & projection.syn_alive[:ns])
    sums = torch.zeros(cortex.n_neurons, device=expected.device)
    sums.index_add_(0, post[live_exc], expected[:ns][live_exc])
    scalable = live_exc & (sums[post] > 1e-9)
    expected[:ns][scalable] *= (target / sums[post[scalable]]).to(torch.float32)
    expected[:ns] = torch.clamp(expected[:ns], projection.syn_min_weight[:ns],
                                projection.syn_max_weight[:ns])
    mn.normalize_feedforward_weights(brain, target)
    assert torch.equal(projection.syn_weight, expected)


@pytest.mark.parametrize("target", [[1.0, 2.0], float("nan"), -1.0, [[1.0]]])
def test_invalid_normalization_targets_do_not_mutate_weights(learning_setup, target):
    brain, *_ = learning_setup
    before = simulation_digest(brain)
    with pytest.raises(ValueError, match="Normalization targets"):
        mn.normalize_feedforward_weights(brain, target)
    assert simulation_digest(brain) == before


def test_unknown_normalization_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalization"):
        LearningConfig(normalization_policy="unknown").validate()


def test_checkpoint_rejects_changed_protocol_data_and_overwrite(learning_setup, tmp_path):
    brain, data, config, *_ = learning_setup
    protocol = training_protocol(data, config, 5, "stdp_normalized", {})
    destination = tmp_path / "checkpoint"
    save_training_checkpoint(brain, destination, {"protocol": protocol, "completed_samples": 0})
    before = directory_digest(destination)
    with pytest.raises(FileExistsError):
        save_training_checkpoint(brain, destination, {"protocol": protocol})
    assert directory_digest(destination) == before
    with pytest.raises(ValueError, match="protocol/data mismatch"):
        load_training_checkpoint(destination, protocol | {"train_sha256": "different"})
    metadata = json.loads((destination / "brain/meta.json").read_text())
    metadata["step_count"] += 1
    (destination / "brain/meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="integrity"):
        load_training_checkpoint(destination, protocol)


def test_checkpoint_save_failure_never_publishes_partial_state(learning_setup, tmp_path, monkeypatch):
    import examples.training_checkpoint as checkpoint
    brain, *_ = learning_setup

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(checkpoint, "save_brain", interrupted)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save_training_checkpoint(brain, tmp_path / "sample", {"protocol": {}})
    assert list(tmp_path.iterdir()) == []


def test_resume_rejects_a_cursor_that_does_not_match_simulated_steps(learning_setup, tmp_path):
    brain, data, config, *_ = learning_setup
    train_condition(copy.deepcopy(brain), data, config, 5, "stdp_normalized", tmp_path, {}, stop_after=5)
    checkpoint = tmp_path / "sample_00000005"
    path = checkpoint / "progress.json"
    metadata = json.loads(path.read_text())
    metadata["progress"]["completed_samples"] = 4
    metadata["progress"]["weight_deltas"] = metadata["progress"]["weight_deltas"][:4]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sample position/state mismatch"):
        train_condition(brain, data, config, 5, "stdp_normalized", tmp_path, {}, resume_from=checkpoint)


def test_cached_response_templates_match_live_inference_and_do_not_change_source(learning_setup, tmp_path, monkeypatch):
    brain, data, config, *_ = learning_setup
    before = simulation_digest(brain)
    path = tmp_path / "responses.npz"
    readout, validation = cached_responses(brain, data, config, path)
    assert simulation_digest(brain) == before
    results = probe_decoders(readout, data.train_y[data.readout_indices], validation,
                             data.validation_y, config, (0, 1))
    with patch.object(mn, "ASSIGN_PRESENT_STEPS", config.inference_steps), \
            patch.object(mn, "TEST_PRESENT_STEPS", config.inference_steps):
        fit = mn.build_readout(brain, data.train_X[data.readout_indices],
                              data.train_y[data.readout_indices], (0, 1))
        accuracy, predictions = mn.evaluate(brain, data.validation_X, data.validation_y, *fit[2:], (0, 1))
    assert results["template"]["accuracy"] == accuracy
    assert results["template"]["predictions"] == predictions.tolist()
    monkeypatch.setattr(mn, "collect_readout_responses", lambda *args: pytest.fail("cache simulated again"))
    again, _ = cached_responses(brain, data, config, path)
    np.testing.assert_array_equal(readout.spikes, again.spikes)
    brain.regions["cortex"].theta[0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        cached_responses(brain, data, config, path)


def test_ridge_scaler_is_fit_on_readout_only():
    X = np.array([[0., 2.], [1., 4.], [2., 6.], [3., 8.]])
    y = np.array([0, 0, 1, 1])
    _, first = ridge_predictions(X, y, np.ones((2, 2)), 1.0)
    _, second = ridge_predictions(X, y, np.ones((2, 2)) * 1e6, 1.0)
    np.testing.assert_array_equal(first[0].mean_, X.mean(axis=0))
    np.testing.assert_array_equal(first[0].mean_, second[0].mean_)
    np.testing.assert_array_equal(first[1].coef_, second[1].coef_)


def test_centered_decoder_is_additive_and_matches_direct_feature_transform(learning_setup, tmp_path):
    brain, data, config, *_ = learning_setup
    readout, validation = cached_responses(brain, data, config, tmp_path / "responses.npz")
    before = copy.deepcopy((readout, validation))
    labels = data.train_y[data.readout_indices]
    results = probe_decoders(readout, labels, validation, data.validation_y, config, (0, 1))
    assert set(results) == {"template", "ridge_spikes", "ridge_spikes_voltage", "ridge_spikes_centered_voltage"}
    for name, centered in (("ridge_spikes_voltage", False), ("ridge_spikes_centered_voltage", True)):
        features = []
        for response in (readout, validation):
            voltage = response.voltages.astype(np.float64)
            if centered:
                voltage -= voltage.mean(axis=1, keepdims=True)
            features.append(np.concatenate((response.spikes, voltage), axis=1))
        predictions, _ = ridge_predictions(features[0], labels, features[1], config.ridge_alpha)
        assert results[name]["predictions"] == predictions.tolist()
        assert results[name]["accuracy"] == float(np.mean(predictions == data.validation_y))
    for old, new in zip(before, (readout, validation)):
        np.testing.assert_array_equal(old.spikes, new.spikes)
        np.testing.assert_array_equal(old.voltages, new.voltages)


def test_delta_metrics_separate_stdp_and_normalization():
    before = torch.ones(3)
    raw = before + torch.tensor([0.25, -0.25, 0.0])
    row = weight_delta_metrics(before, raw, before)
    assert row["raw_stdp_l1"] == row["normalization_l1"] == 0.5
    assert row["raw_normalization_cosine"] == pytest.approx(-1.0)
    assert row["total_to_raw_l1"] == 0.0


def test_summary_pairs_seeds_and_does_not_treat_initial_as_stdp_control():
    rows = [{"seed": seed, "condition": condition,
             "decoders": {d: {"accuracy": accuracy} for d in
                          ("template", "ridge_spikes", "ridge_spikes_voltage")}}
            for seed, condition, accuracy in ((1, "initial", 0.9), (1, "normalization_only", 0.2),
                                              (1, "stdp_normalized", 0.3), (2, "stdp_normalized", 0.8))]
    summary = comparison_summary(rows, [1, 2])
    assert summary["template"]["mean_stdp_gain_pp"] == pytest.approx(10)
    assert len(summary["template"]["paired_seeds"]) == 1
    assert summary["ridge_spikes_centered_voltage"]["mean_stdp_gain_pp"] is None


def test_array_identity_includes_shape_and_dtype():
    a = np.arange(4, dtype=np.int32)
    assert array_digest(a) != array_digest(a.reshape(2, 2))
    assert array_digest(a) != array_digest(a.astype(np.int64))


@pytest.mark.parametrize("coupling", [8., 64.])
@pytest.mark.parametrize("stdp_scale", [.2, 2.])
@pytest.mark.parametrize("learning_rule", ["pair", "post_trace"])
def test_cli_recovers_partial_condition_and_skips_completed_conditions(learning_setup, monkeypatch, tmp_path, coupling, stdp_scale, learning_rule):
    import examples.mnist_learning_check as study
    monkeypatch.setattr(mn, "CLASSES", tuple(range(10)))
    raw = np.random.default_rng(1).integers(0, 256, size=(60020, 4))
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: SimpleNamespace(
        data=raw, target=np.arange(60020) % 10,
    ))
    options = ["--seeds", "5", "--train-per-class", "1", "--readout-per-class", "1",
               "--validation-per-class", "1", "--epochs", "1", "--train-steps", "2",
               "--inference-steps", "2", "--rest-steps", "0", "--checkpoint-every", "5",
               "--exc-to-inh-weight", str(coupling), "--stdp-scale", str(stdp_scale),
               "--learning-rule", learning_rule]
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"

    def run(path, resume=False):
        monkeypatch.setattr("sys.argv", ["learning_check", "--output", str(path), *options,
                                         *(["--resume"] if resume else [])])
        study.main()

    run(full)
    original = study.train_condition

    def interrupt(*args, **kwargs):
        original(*args, **kwargs, stop_after=1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(study, "train_condition", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(interrupted)
    assert (interrupted / "seed_5/normalization_only/checkpoints/sample_00000001/progress.json").exists()
    monkeypatch.setattr(study, "train_condition", original)
    run(interrupted, resume=True)
    reference = json.loads((full / "summary.json").read_text())
    resumed = json.loads((interrupted / "summary.json").read_text())
    assert resumed["complete"]
    assert resumed["source_unchanged"]
    assert "examples/mnist_trace_stdp.py" in resumed["signature"]["source_sha256"]
    assert "examples/mnist_readout_features.py" in resumed["signature"]["source_sha256"]
    for name, digest in resumed["signature"]["source_sha256"].items():
        assert hashlib.sha256((interrupted / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    assert reference["comparison"] == resumed["comparison"]
    for a, b in zip(reference["results"], resumed["results"]):
        assert a["final_sha256"] == b["final_sha256"]
        assert a["decoders"] == b["decoders"]
    monkeypatch.setattr(study, "cached_responses", lambda *args: pytest.fail("completed condition repeated"))
    monkeypatch.setattr(study, "train_condition", lambda *args, **kwargs: pytest.fail("completed training repeated"))
    run(interrupted, resume=True)
    assert "examples/_utils.py" in resumed["signature"]["source_sha256"]
    # Simulate changed helper source without touching workspace files. The
    # shared quiet-step helper changes training and must invalidate a resume.
    original_read = Path.read_bytes
    utilities = Path(study.__file__).resolve().with_name("_utils.py")
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_read(path) + (
        b"\n# simulated helper change\n" if path.resolve() == utilities else b""))
    with pytest.raises(ValueError, match="Cannot resume with changed code"):
        run(interrupted, resume=True)


def test_normalization_control_is_identical_for_both_learning_rules(learning_setup, tmp_path):
    brain, data, config, *_ = learning_setup
    pair, left = train_condition(copy.deepcopy(brain), data, config, 5, "normalization_only",
                                 tmp_path / "pair", {})
    trace, right = train_condition(copy.deepcopy(brain), data, replace(config, learning_rule="post_trace"),
                                   5, "normalization_only", tmp_path / "trace", {})
    assert simulation_digest(pair) == simulation_digest(trace)
    assert left["weight_deltas"] == right["weight_deltas"]


def test_cli_freezes_exclusions_and_rejects_changed_rows_on_resume(learning_setup, monkeypatch, tmp_path):
    import examples.mnist_learning_check as study
    monkeypatch.setattr(mn, "CLASSES", tuple(range(10)))
    raw = np.random.default_rng(1).integers(0, 256, size=(60020, 4))
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: SimpleNamespace(
        data=raw, target=np.arange(60020) % 10))
    exclusion_file, output = tmp_path / "excluded.json", tmp_path / "run"
    exclusion_file.write_text(json.dumps(list(range(100))))
    options = ["learning_check", "--output", str(output), "--exclude-rows", str(exclusion_file),
               "--seeds", "5", "--train-per-class", "1", "--readout-per-class", "1",
               "--validation-per-class", "1", "--train-steps", "2", "--inference-steps", "2",
               "--rest-steps", "0"]
    monkeypatch.setattr("sys.argv", options)
    study.main()
    result = json.loads((output / "summary.json").read_text())
    assert result["complete"]
    assert result["signature"]["excluded_ids"] == list(range(100))
    assert result["dataset"]["excluded_ids"] == list(range(100))
    assert min(result["dataset"]["train_ids"] + result["dataset"]["validation_ids"]) >= 100
    monkeypatch.setattr(study, "cached_responses", lambda *args: pytest.fail("completed condition repeated"))
    monkeypatch.setattr("sys.argv", options + ["--resume"])
    study.main()
    exclusion_file.write_text(json.dumps(list(range(101))))
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: pytest.fail("resumed with changed exclusions"))
    with pytest.raises(ValueError, match="Cannot resume"):
        study.main()


@pytest.mark.parametrize("changes", [{"learning_rule": "pair"}, {"trace_tau": 10.}, {"trace_target": .1}])
def test_trace_checkpoint_rejects_changed_rule_parameters(learning_setup, tmp_path, changes):
    brain, data, config, *_ = learning_setup
    config = replace(config, learning_rule="post_trace")
    train_condition(copy.deepcopy(brain), data, config, 5, "stdp_normalized", tmp_path, {}, stop_after=1)
    with pytest.raises(ValueError, match="protocol/data mismatch"):
        train_condition(copy.deepcopy(brain), data, replace(config, **changes), 5, "stdp_normalized",
                         tmp_path, {}, resume_from=tmp_path / "sample_00000001")


@pytest.mark.parametrize("changed_location", ["workspace", "snapshot"])
def test_cli_does_not_mark_changed_sources_as_complete(learning_setup, tmp_path, monkeypatch, changed_location):
    import examples.mnist_learning_check as study
    monkeypatch.setattr(mn, "CLASSES", tuple(range(10)))
    raw = np.random.default_rng(1).integers(0, 256, size=(60020, 4))
    changed = False

    def fetch(*args, **kwargs):
        nonlocal changed
        changed = True
        return SimpleNamespace(data=raw, target=np.arange(60020) % 10)

    monkeypatch.setattr(mn, "fetch_openml", fetch)
    output = tmp_path / "run"
    helper = (Path(study.__file__).resolve().with_name("mnist_trace_stdp.py")
              if changed_location == "workspace"
              else output / "source_snapshot/examples/mnist_trace_stdp.py")
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_read(path) + (
        b"\n# changed during run\n" if changed and path.resolve() == helper.resolve() else b""))
    monkeypatch.setattr("sys.argv", ["learning_check", "--output", str(output), "--seeds", "5",
        "--train-per-class", "1", "--readout-per-class", "1", "--validation-per-class", "1",
        "--train-steps", "2", "--inference-steps", "2", "--rest-steps", "0",
        "--learning-rule", "post_trace"])
    with pytest.raises(RuntimeError, match="source or snapshot changed"):
        study.main()
    result = json.loads((output / "summary.json").read_text())
    assert not result["complete"]
    assert "source_unchanged" not in result
