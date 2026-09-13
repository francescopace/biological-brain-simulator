import copy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from examples import mnist_paper_version_check as study
from examples.mnist_paper_check import train_pair
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork


def initial(**kwargs):
    c = replace(ReferenceConfig(dynamics_version=1, integration_substeps=1), n_input=8, n_exc=2,
                incoming_sum=4., presentation_ms=40., rest_ms=10., min_spikes=1, input_delay_max=2., **kwargs)
    network = ReferenceNetwork(c, seed=31)
    network.normalize()
    return network


@pytest.mark.parametrize("variant", study.VARIANTS)
def test_version_copy_preserves_all_original_initial_arrays_and_input_clock(variant):
    source = initial()
    before = source.state_digest()
    copied = study.variant_copy(source, variant)
    for name, value in vars(source).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(copied, name), value)
            assert not np.shares_memory(getattr(copied, name), value)
    assert copied.config.dt == source.config.dt
    assert (copied.config.dynamics_version, copied.config.integration_substeps) == study.VARIANTS[variant]
    if variant == "v1":
        assert copied.state_digest() == before
    else:
        assert not copied.release_e.any() and not copied.release_i.any()
    assert source.state_digest() == before


@pytest.mark.parametrize("changed", ["step", "pending", "version", "name"])
def test_invalid_version_conversion_rejected(changed):
    source = initial()
    if changed == "step":
        source.step_count = 1
    elif changed == "pending":
        source.pending_e[0] = True
    elif changed == "version":
        source.config = replace(source.config, dynamics_version=2, integration_substeps=8)
    with pytest.raises(ValueError):
        study.variant_copy(source, "bad" if changed == "name" else "v2")


@pytest.mark.parametrize("zero_learning", [False, True])
def test_coupled_training_replays_v1_and_controls_receive_identical_exposure(zero_learning):
    source = initial(**(dict(eta_pre=0., eta_post=0.) if zero_learning else {}))
    before = source.state_digest()
    trained, control = copy.deepcopy(source), copy.deepcopy(source)
    images = np.array([[255.] * 8, [220.] * 8])
    records = train_pair(trained, control, images, seed=72)
    progress = []
    models, history, responses, costs = study.train_coupled(source, images, records, seed=72,
        callback=lambda n, m, h, c: progress.append(n))
    assert progress == [1, 2] and source.state_digest() == before
    assert models["v1"][study.ARMS[1]].state_digest() == trained.state_digest()
    assert models["v1"][study.ARMS[0]].state_digest() == control.state_digest()
    assert len(history) == 2 and len(responses) == len(costs) == 6
    for variant, arms in models.items():
        for arm, network in arms.items():
            assert network.step_count == trained.step_count
            np.testing.assert_array_equal(network.delays, source.delays)
            assert responses[f"{variant}__{arm}"]["spikes"].shape == (2, 2)
            assert costs[f"{variant}__{arm}"]["cpu_seconds"] > 0
        np.testing.assert_array_equal(arms[study.ARMS[0]].weights, control.weights)
        if zero_learning:
            assert arms[study.ARMS[0]].state_digest() == arms[study.ARMS[1]].state_digest()
    metrics = study.state_metrics(source, models)
    assert metrics["changes"]["v2_to_v2_half"][study.ARMS[0]]["weight_relative_l1"] == 0.
    if zero_learning:
        assert metrics["changes"]["v2_to_v2_half"]["learning_component_relative_l1"] is None


def test_historical_record_mismatch_fails_instead_of_silently_comparing():
    source = initial()
    images = np.full((1, 8), 255.)
    records = train_pair(copy.deepcopy(source), copy.deepcopy(source), images, seed=7)
    records[0]["accepted_exc_spikes"] += 1
    with pytest.raises(ValueError, match="spikes did not replay"):
        study.train_coupled(source, images, records, seed=7)


def test_common_probe_discards_transients_and_training_grid_but_preserves_source():
    source = initial()
    images, ids = np.full((2, 8), 255.), [91, 72]
    responses = []
    for variant in study.VARIANTS:
        model = study.variant_copy(source, variant)
        model.advance(np.ones((2, 8), bool))
        before = model.state_digest()
        responses.append(study.common_probe(model, images, ids, seed=44))
        assert model.state_digest() == before
    for response in responses[1:]:
        for key in response:
            np.testing.assert_array_equal(response[key], responses[0][key])
    reverse = study.common_probe(source, images[::-1], ids[::-1], seed=44)
    for key in reverse:
        np.testing.assert_array_equal(reverse[key][::-1], responses[0][key])


def test_silent_probes_are_retained_and_zero_denominator_is_explicit():
    response = study.common_probe(initial(), np.zeros((1, 8)), [91], seed=44)
    assert response["spikes"].shape == (1, 2) and not response["spikes"].any()
    difference = study.response_difference(response, response)
    assert difference["relative_spike_l1"] is None
    assert difference["mean_absolute_spike_difference"] == difference["centered_voltage_rmse_mv"] == 0.


def test_response_metrics_remove_common_voltage_offset_and_detect_spike_changes():
    a = dict(spikes=np.array([[1, 2]]), voltages=np.array([[-65., -60.]]))
    b = dict(spikes=np.array([[2, 1]]), voltages=a["voltages"] + 10.)
    result = study.response_difference(a, b)
    assert result["relative_spike_l1"] == pytest.approx(2/3)
    assert result["centered_voltage_rmse_mv"] == 0.
    assert result["same_spike_vector_fraction"] == 0.


def test_state_metrics_reject_different_control_normalization():
    source = initial()
    models = {v: {a: study.variant_copy(source, v) for a in study.ARMS} for v in study.VARIANTS}
    models["v2"][study.ARMS[0]].weights[0, 0] += .1
    with pytest.raises(AssertionError):
        study.state_metrics(source, models)


def test_worker_summary_checkpoints_caches_and_no_overwrite(tmp_path, monkeypatch):
    source = initial()
    source.save(tmp_path/"initial", {})
    folder = tmp_path/"run"
    folder.mkdir()
    np.savez_compressed(tmp_path/"images.npz", train_images=np.full((2, 8), 255.), probe_images=np.full((2, 8), 255.))
    row = dict(seed=201, checkpoint=str(tmp_path/"initial"), checkpoint_metadata={}, state_sha256=source.state_digest(),
               legacy_final={a: dict(state_sha256=source.state_digest()) for a in study.ARMS},
               configs={v: asdict(study.variant_copy(source, v).config) for v in study.VARIANTS})
    plan = dict(seeds=[201], rows=[row], source_root=str(tmp_path), records={"201": []}, milestones=[10, 25, 50],
                probe_ids=[71, 72], poisson_seed_offset=100000)
    study.atomic_json(folder/"plan.json", plan)
    monkeypatch.setattr(study, "checked_plan", lambda p: plan)
    def train_fixture(initial, images, records, *, seed, callback):
        models = {v: {a: study.variant_copy(initial, v) for a in study.ARMS} for v in study.VARIANTS}
        history = [dict(position=i, attempts=1) for i in range(50)]
        responses = {f"{v}__{a}": dict(spikes=np.zeros((50, 2), int), voltages=np.zeros((50, 2))) for v in study.VARIANTS for a in study.ARMS}
        costs = {k: dict(cpu_seconds=1., wall_seconds=1.) for k in responses}
        for n in (10, 25, 50):
            callback(n, models, history[:n], costs)
        return models, history, responses, costs
    monkeypatch.setattr(study, "train_coupled", train_fixture)
    study.run_seed(folder, 201)
    study.summarize(folder)
    result = json.loads((folder/"summary.json").read_text())
    assert result["complete"] and result["source_unchanged"]
    assert len(result["results"][0]["checkpoints"]) == 6
    assert result["results"][0]["legacy_full_state_replay"]
    with pytest.raises(FileExistsError):
        study.run_seed(folder, 201)
    with pytest.raises(FileExistsError):
        study.summarize(folder)


def test_checked_plan_rejects_a_changed_source_or_input(tmp_path, monkeypatch):
    repo = tmp_path/"repo"
    repo.mkdir()
    (repo/"model.py").write_text("original")
    snapshot = tmp_path/"source_snapshot"
    snapshot.mkdir()
    (snapshot/"model.py").write_text("original")
    data = tmp_path/"input.npz"
    data.write_bytes(b"original input")
    plan = dict(protocol_version=1, variants={k: list(v) for k, v in study.VARIANTS.items()}, training_images=50,
                milestones=[10, 25, 50], probe_dynamics_version=2, probe_substeps=16, numpy=np.__version__,
                source_sha256={"model.py": study.sha256(repo/"model.py")}, input_files={str(data): study.sha256(data)})
    study.atomic_json(tmp_path/"plan.json", plan)
    monkeypatch.setattr(study, "REPO", repo)
    assert study.checked_plan(tmp_path) == plan
    data.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Input changed"):
        study.checked_plan(tmp_path)
    data.write_bytes(b"original input")
    (repo/"model.py").write_text("changed source")
    with pytest.raises(ValueError, match="Current source changed"):
        study.checked_plan(tmp_path)
