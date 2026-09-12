"""Fresh decoder validation must not refit preprocessing or train the SNN."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import examples.mnist_benchmark as mn
import examples.mnist_learning_check as learning
import examples.mnist_readout_check as check
from examples.training_checkpoint import directory_digest


def responses(spikes=None, voltage=None):
    spikes = np.array([[0, 2, 1], [1, 0, 3]], dtype=np.int32) if spikes is None else spikes
    voltage = np.array([[-70., -65., -60.], [-64., -68., -69.]], dtype=np.float32) if voltage is None else voltage
    return mn.ReadoutResponses(np.arange(3), spikes, voltage)


def test_standard_features_match_original_dtype_arithmetic_and_preserve_inputs():
    source = responses()
    before = copy.deepcopy(source)
    actual = check.readout_features(source, "standard")
    expected = np.concatenate((source.spikes, source.voltages), axis=1)
    assert actual.dtype == expected.dtype == np.float64
    assert actual.tobytes() == expected.tobytes()
    actual[:] = 0.
    np.testing.assert_array_equal(source.spikes, before.spikes)
    np.testing.assert_array_equal(source.voltages, before.voltages)


def test_centering_is_per_image_offset_invariant_and_preserves_spikes():
    source = responses()
    centered = check.readout_features(source, "voltage_centered")
    np.testing.assert_array_equal(centered[:, :3], source.spikes)
    np.testing.assert_allclose(centered[:, 3:].mean(axis=1), 0., atol=1e-14)
    offset = source.voltages.astype(float) + np.array([[128.], [-64.]])
    other = responses(source.spikes[::-1], offset[::-1])
    np.testing.assert_array_equal(check.readout_features(other, "voltage_centered"), centered[::-1])
    source.voltages[1] = 10000.
    np.testing.assert_array_equal(check.readout_features(source, "voltage_centered")[0], centered[0])


@pytest.mark.parametrize("spikes,voltage", [(np.zeros((2, 2)), np.zeros((2, 3))),
    (np.full((2, 3), -1.), np.zeros((2, 3))), (np.zeros((2, 3)), np.full((2, 3), np.nan)),
    (np.full((2, 3), np.inf), np.zeros((2, 3)))])
def test_invalid_response_arrays_are_rejected(spikes, voltage):
    with pytest.raises(ValueError):
        check.readout_features(responses(spikes, voltage), "voltage_centered")


def test_unknown_mode_fails():
    with pytest.raises(ValueError, match="Unknown"):
        check.readout_features(responses(), "other")


@pytest.fixture
def split():
    raw = np.random.default_rng(41).integers(0, 256, size=(100, 4))
    labels = np.arange(100) % 2
    config = learning.LearningConfig(train_per_class=3, readout_per_class=2, validation_per_class=2)
    data = learning.prepare_dataset(raw, labels, config, classes=(0, 1), train_boundary=80)
    return raw, labels, data


def test_fresh_split_is_balanced_disjoint_and_keeps_fitted_preprocessing(split, monkeypatch):
    raw, labels, source = split
    monkeypatch.setattr(mn, "l1_equalization_target", lambda *args, **kwargs: pytest.fail("refitted target"))
    fresh = check.additional_validation(raw, labels, source, per_class=5, classes=(0, 1))
    ids = set(fresh.manifest["validation_ids"])
    excluded = set(source.manifest["train_ids"] + source.manifest["validation_ids"])
    assert len(ids) == 10 and not ids & excluded and max(ids) < 80
    assert np.bincount(fresh.validation_y).tolist() == [5, 5]
    assert fresh.manifest["intensity_target_l1"] == source.manifest["intensity_target_l1"]
    assert fresh.train_X is source.train_X and fresh.readout_indices is source.readout_indices
    changed = raw.copy()
    changed[list(excluded)] = 0
    changed[80:] = 255
    again = check.additional_validation(changed, labels, source, per_class=5, classes=(0, 1))
    assert fresh.manifest == again.manifest
    np.testing.assert_array_equal(fresh.validation_X, again.validation_X)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, 100])
def test_invalid_fresh_budget_is_rejected(split, count):
    raw, labels, data = split
    with pytest.raises(ValueError):
        check.additional_validation(raw, labels, data, per_class=count, classes=(0, 1))


def test_invalid_source_partition_is_rejected(split):
    raw, labels, data = split
    data = replace(data, manifest=dict(data.manifest, validation_ids=data.manifest["train_ids"]))
    with pytest.raises(ValueError, match="source study partitions"):
        check.additional_validation(raw, labels, data, classes=(0, 1))


def test_fresh_labels_only_affect_scoring_not_predictions():
    rng = np.random.default_rng(8)
    source = responses(rng.integers(0, 4, size=(10, 3)), rng.normal(size=(10, 3)))
    prior = responses(rng.integers(0, 4, size=(6, 3)), rng.normal(size=(6, 3)))
    fresh = responses(rng.integers(0, 4, size=(8, 3)), rng.normal(size=(8, 3)))
    labels = np.arange(10) % 2
    old_y, fresh_y = np.arange(6) % 2, np.arange(8) % 2
    expected, _ = learning.ridge_predictions(check.readout_features(source, "standard"), labels,
                                             check.readout_features(prior, "standard"), 1.)
    first, paired = check.compare_readouts(source, labels, prior, old_y, fresh, fresh_y, 1., expected.tolist())
    second, _ = check.compare_readouts(source, labels, prior, old_y, fresh, 1 - fresh_y, 1., expected.tolist())
    for mode in check.MODES:
        assert first[mode]["fresh_predictions"] == second[mode]["fresh_predictions"]
    assert sum(value for key, value in paired.items() if key != "gain_pp") == 8
    assert paired["gain_pp"] == pytest.approx(100 * (paired["centered_wins"] - paired["centered_losses"]) / 8)
    with pytest.raises(AssertionError, match="predictions differ"):
        check.compare_readouts(source, labels, prior, old_y, fresh, fresh_y, 1., [-1] * 6)


@pytest.fixture
def source_study(tmp_path, monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 4, "N_CORTEX_INH": 4}.items():
        monkeypatch.setattr(mn, name, value)
    raw = np.random.default_rng(1).integers(0, 256, size=(60020, 4))
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: SimpleNamespace(
        data=raw, target=np.arange(60020) % 10))
    study = tmp_path / "source"
    monkeypatch.setattr("sys.argv", ["learning", "--output", str(study), "--seeds", "5",
        "--train-per-class", "1", "--readout-per-class", "1", "--validation-per-class", "1",
        "--train-steps", "2", "--inference-steps", "2", "--rest-steps", "0", "--input-density", "1"])
    learning.main()
    return study


def run_check(study, output, monkeypatch):
    monkeypatch.setattr("sys.argv", ["readout", "--studies", str(study), "--output", str(output),
                                     "--fresh-per-class", "1"])
    check.main()


def test_cli_preserves_source_and_reproduces_prior_before_scoring_fresh(source_study, tmp_path, monkeypatch):
    before = directory_digest(source_study)
    monkeypatch.setattr(learning, "train_condition", lambda *args, **kwargs: pytest.fail("SNN retraining"))
    output = tmp_path / "check"
    run_check(source_study, output, monkeypatch)
    result = json.loads((output / "summary.json").read_text())
    assert result["complete"] and result["source_unchanged"]
    assert result["results"][0]["prior_responses_bitwise_equal"]
    assert directory_digest(source_study) == before
    fresh = set(result["fresh_dataset"]["validation_ids"])
    assert len(fresh) == 10 and not fresh & set(result["fresh_dataset"]["excluded_ids"])
    for mode in check.MODES:
        assert len(result["results"][0]["scores"][mode]["fresh_predictions"]) == 10


def test_duplicate_or_incomplete_sources_are_rejected(source_study):
    with pytest.raises(ValueError, match="distinct network seeds"):
        check.selected_studies([source_study, source_study])
    path = source_study / "summary.json"
    study = json.loads(path.read_text())
    study["complete"] = False
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="completed learning studies"):
        check.selected_studies([source_study])


def test_changed_source_code_does_not_publish_complete(source_study, tmp_path, monkeypatch):
    output = tmp_path / "check"
    changed = False
    original_compare = check.compare_readouts

    def compare(*args, **kwargs):
        nonlocal changed
        changed = True
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(check, "compare_readouts", compare)
    original_read = Path.read_bytes
    own = Path(check.__file__).resolve()
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_read(path) + (
        b"\n# simulated edit\n" if changed and path.resolve() == own else b""))
    with pytest.raises(RuntimeError, match="source or snapshot changed"):
        run_check(source_study, output, monkeypatch)
    assert not json.loads((output / "summary.json").read_text())["complete"]


def test_cache_mismatch_is_detected(source_study, tmp_path, monkeypatch):
    original = check.verify_response_cache

    def corrupt(path, brain, original_data, config, readout, validation):
        validation.spikes[0, 0] += 1
        return original(path, brain, original_data, config, readout, validation)

    monkeypatch.setattr(check, "verify_response_cache", corrupt)
    with pytest.raises(AssertionError, match="does not reproduce"):
        run_check(source_study, tmp_path / "check", monkeypatch)
