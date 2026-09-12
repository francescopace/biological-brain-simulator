"""Complete sensory coverage, independent inference and recorded Iris selection."""

import copy
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

import examples.iris_benchmark as iris
from examples._utils import inference_brain
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import directory_digest, load_training_checkpoint


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(iris, "N_INPUT", 4)
    monkeypatch.setattr(iris, "N_CORTEX", 4)
    monkeypatch.setattr(iris, "TEST_PRESENT_STEPS", 12)
    monkeypatch.setattr(iris, "REST_STEPS", 2)
    monkeypatch.setattr(iris, "TRAIN_PRESENT_STEPS", 20)
    monkeypatch.setattr(iris, "TEACHER_DELAY", 6)
    return iris.build_brain(seed=7)


@pytest.mark.parametrize("seed", [42, 101, 102])
def test_every_sensory_bin_reaches_each_class_and_all_weights_are_bounded(seed):
    brain = iris.build_brain(seed)
    source = brain.regions["input"]
    assert torch.all(source.neuron_type[:source.n_neurons] == iris.NeuronType.EXCITATORY.value)
    projection = iris._readout_proj(brain)
    n = projection.n_synapses
    pairs = set(zip(projection.syn_pre[:n].tolist(), projection.syn_post[:n].tolist()))
    assert pairs == {(pre, post) for pre in range(80) for post in range(3)}
    for target in [*brain.regions.values(), *brain.projections]:
        weights = target.syn_weight[:target.n_synapses]
        assert torch.all(weights >= target.syn_min_weight[:target.n_synapses])
        assert torch.all(weights <= target.syn_max_weight[:target.n_synapses])


@pytest.mark.parametrize("repeats", [1, 2, 6])
def test_fast_independent_inference_matches_every_repeat_and_preserves_source(tiny, monkeypatch, repeats):
    monkeypatch.setattr(iris, "TEST_REPEATS", repeats)
    projection = iris._readout_proj(tiny)
    projection.syn_weight[:projection.n_synapses] = 3.
    tiny.regions["motor"].theta[:3] = .1
    before = simulation_digest(tiny)
    X = np.array([[0., 1., 0., 1.], [1., 0., 1., 0.], [.4, .8, .3, .9]])
    y = np.array([0, 1, 2])
    calls = []
    original_present = iris.present
    def capture(model, image):
        assert not model.oscillators.enabled and not model.memory.enabled
        assert not model.reward_stdp.enabled and not model.homeostasis.theta_enabled
        assert all(not target.plasticity_enabled for target in [*model.regions.values(), *model.projections])
        assert all(torch.all(r.v[:r.n_neurons] == -65.) for r in model.regions.values())
        response = original_present(model, image)
        calls.append(response)
        return response
    monkeypatch.setattr(iris, "present", capture)
    slow = iris.evaluate(tiny, X, y, fast=False)
    assert len(calls) == len(X) * repeats
    all_repeats = calls.copy()
    calls.clear()
    fast = iris.evaluate(tiny, X, y)
    assert len(calls) == len(X)
    for i, computed_once in enumerate(calls):
        for repeated in all_repeats[i * repeats:(i + 1) * repeats]:
            for left, right in zip(computed_once, repeated):
                assert left.tobytes() == right.tobytes()  # Includes raw mean voltages.
    reverse = iris.evaluate(tiny, X[::-1], y[::-1])
    alone = iris.evaluate(tiny, X[1:2], y[1:2])
    for key in ("spike_predictions", "fallback_predictions", "counts"):
        assert getattr(slow, key).tobytes() == getattr(fast, key).tobytes()
        np.testing.assert_array_equal(getattr(fast, key), getattr(reverse, key)[::-1])
        np.testing.assert_array_equal(getattr(fast, key)[1:2], getattr(alone, key))
    assert simulation_digest(tiny) == before


def test_fast_repeat_request_does_not_reuse_a_still_adapting_network(tiny, monkeypatch):
    tiny.disable_oscillations()
    seen = []
    original = iris.present
    monkeypatch.setattr(iris, "present", lambda *args: (seen.append(1), original(*args))[1])
    iris.test_one_sample(tiny, np.ones(4), 3, independent=True, reuse_identical=True)
    assert len(seen) == 3


def test_runtime_step_and_repeat_overrides_are_resolved_at_call_time(tiny, monkeypatch):
    monkeypatch.setattr(iris, "TEST_PRESENT_STEPS", 2)
    monkeypatch.setattr(iris, "TEST_REPEATS", 3)
    monkeypatch.setattr(iris, "REST_STEPS", 4)
    initial = tiny.step_count
    iris.test_one_sample(tiny, np.ones(4))
    assert tiny.step_count - initial == 3 * (2 + 4)


def test_converted_training_input_keeps_noise_and_simulation_state(tiny):
    original = copy.deepcopy(tiny)
    image = np.array([.25, .123456789, .75, 1.], dtype=np.float64)
    # Emulate the previous encoder entry: original NumPy data every step.
    stimulate = original.stimulate
    with patch.object(original, "stimulate", lambda name, values: stimulate(name, image)):
        left = iris.train_one_sample(original, image, 0, .1, .05)
    right = iris.train_one_sample(tiny, image, 0, .1, .05)
    assert left == right
    assert simulation_digest(original) == simulation_digest(tiny)


@pytest.mark.parametrize("validation_only", [False, True])
def test_main_records_original_split_and_selects_only_on_validation(tiny, monkeypatch, tmp_path, validation_only):
    monkeypatch.setattr(iris, "EPOCHS", 2)
    monkeypatch.setattr(iris, "build_brain", lambda **kwargs: copy.deepcopy(tiny))
    calls = []
    def train(brain, *args, **kwargs):
        brain.regions["motor"].theta[0] += 1.
        return -1
    def evaluate(brain, X, y):
        calls.append((len(y), float(brain.regions["motor"].theta[0])))
        predictions = y.copy() if len(calls) in (2, 4) else np.full(len(y), -1)
        return iris.IrisEvaluation(float(np.mean(predictions == y)), float(np.mean(predictions == y)),
                                    predictions, predictions.copy(), np.zeros((len(y), 3), dtype=int))
    monkeypatch.setattr(iris, "train_one_sample", train)
    monkeypatch.setattr(iris, "evaluate", evaluate)
    output = tmp_path / "recorded"
    result = iris.main(output=output, validation_only=validation_only)
    assert result["complete"] and result["source_unchanged"]
    assert result["best_epoch"] == 1
    assert result["final_partition"] == ("validation" if validation_only else "test")
    assert calls == [(24, 0.), (24, 96.), (24, 192.), (24 if validation_only else 30, 96.)]
    raw = iris.load_iris()
    Xtv, Xt, ytv, yt = iris.train_test_split(raw.data, raw.target, test_size=.2,
                                          random_state=iris.SEED, stratify=raw.target)
    Xr, Xv, yr, yv = iris.train_test_split(Xtv, ytv, test_size=.2, random_state=iris.SEED + 1, stratify=ytv)
    for key, expected in (("train_ids", Xr), ("validation_ids", Xv), ("test_ids", Xt)):
        np.testing.assert_array_equal(raw.data[result["dataset"][key]], expected)
    path = output / "checkpoints/selected"
    meta = json.loads((path / "progress.json").read_text())
    selected, _ = load_training_checkpoint(path, meta["progress"]["protocol"])
    assert float(selected.regions["motor"].theta[0]) == 96.
    assert (output / "checkpoints/epoch_002/progress.json").exists()
    digest = directory_digest(output)
    with pytest.raises(FileExistsError):
        iris.main(output=output)
    assert directory_digest(output) == digest

    from examples import iris_inference_check as check
    X, y = check.encoded_partition(result, "validation")
    assert len(X) == len(y) == 24
    changed = copy.deepcopy(result)
    changed["dataset"]["normalization_lo"][0] -= 1.
    with pytest.raises(ValueError, match="dataset or preprocessing"):
        check.encoded_partition(changed, "validation")
    analysis_output = tmp_path / "inference.json"
    monkeypatch.setattr(sys, "argv", ["iris_inference_check", "--study", str(output),
                                     "--output", str(analysis_output), "--evaluate-test"])
    if not validation_only:
        with pytest.raises(SystemExit):
            check.main()
        assert not analysis_output.exists()
    else:
        evaluations = []
        def selected_evaluation(brain, X, y, **kwargs):
            evaluations.append(len(y))
            return iris.IrisEvaluation(1., 1., y.copy(), y.copy(), np.zeros((len(y), 3), dtype=int))
        monkeypatch.setattr(iris, "evaluate", selected_evaluation)
        check.main()
        assert evaluations == [24, 24, 24, 24, 30]
        saved = json.loads(analysis_output.read_text())
        assert saved["complete"] and saved["source_unchanged"] and saved["counts_and_predictions_equal"]
        assert saved["selected_epoch"] == 1
        assert directory_digest(output) == digest
