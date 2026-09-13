import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from examples import mnist_paper_training_grid as study
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork


def initial_network(**kwargs):
    c = replace(ReferenceConfig(dynamics_version=1, integration_substeps=1), n_input=8, n_exc=2, incoming_sum=4.,
                presentation_ms=40., rest_ms=10., min_spikes=1, input_delay_max=2., **kwargs)
    n = ReferenceNetwork(c, seed=31)
    n.normalize()
    return n


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_training_refinement_preserves_below_rest_initial_state(factor):
    source = initial_network()
    before = source.state_digest()
    result = study.training_copy(source, factor)
    np.testing.assert_array_equal(result.v_e, np.full(2, source.config.rest_e - 40.))
    np.testing.assert_array_equal(result.v_i, np.full(2, source.config.rest_i - 40.))
    np.testing.assert_array_equal(result.delays * result.config.dt, source.delays * source.config.dt)
    for name in ('weights', 'theta', 'last_pre', 'last_post', 'last_inh'):
        np.testing.assert_array_equal(getattr(result, name), getattr(source, name))
    result.weights[0, 0] = 0.
    assert source.state_digest() == before


def test_training_refinement_rejects_an_evolved_state():
    source = initial_network()
    source.step_count = 1
    with pytest.raises(ValueError, match='untouched'):
        study.training_copy(source, 2)


@pytest.mark.parametrize('factor', [True, 3, 2.])
def test_invalid_refinement_factors_fail(factor):
    with pytest.raises(ValueError, match='factor'):
        study.training_copy(initial_network(), factor)


@pytest.mark.parametrize('zero_rates', [False, True])
def test_coarse_training_replays_entire_original_state_and_controls_share_exposure(zero_rates):
    source = initial_network(**({'eta_pre': 0., 'eta_post': 0.} if zero_rates else {}))
    before = source.state_digest()
    original, control = copy.deepcopy(source), copy.deepcopy(source)
    images = np.array([[255.] * 8, [220.] * 8])
    records = study.pilot.train_pair(original, control, images, seed=72)
    progress = []
    models, history = study.train_coupled(source, images, records, seed=72,
        callback=lambda n, m, h: progress.append((n, len(h))))
    assert models[1]['stdp_normalized'].state_digest() == original.state_digest()
    assert models[1]['normalization_only'].state_digest() == control.state_digest()
    assert progress == [(1, 1), (2, 2)]
    assert source.state_digest() == before
    assert len(history) == len(records)
    for f in study.FACTORS:
        for name, n in models[f].items():
            assert n.step_count * n.config.dt == original.step_count * original.config.dt
        np.testing.assert_array_equal(models[f]['normalization_only'].weights, control.weights)
        if zero_rates:
            assert models[f]['stdp_normalized'].state_digest() == models[f]['normalization_only'].state_digest()
    metrics = study.state_metrics(source, models)
    assert metrics['grid_changes']['1_to_4']['normalization_only']['weight_relative_l1'] == 0.
    if zero_rates:
        assert metrics['grid_changes']['1_to_4']['learning_component_relative_l1'] is None


def test_mismatched_historical_spikes_fail_loudly():
    source = initial_network()
    images = np.full((1, 8), 255.)
    old = study.pilot.train_pair(copy.deepcopy(source), copy.deepcopy(source), images, seed=7)
    old[0]['accepted_exc_spikes'] += 1
    with pytest.raises(ValueError, match='training spikes did not replay'):
        study.train_coupled(source, images, old, seed=7)


def test_common_probe_is_grid_independent_for_identical_learned_parameters_and_frozen():
    source = initial_network()
    images, ids = np.full((2, 8), 255.), [82, 19]
    responses = []
    for factor in study.FACTORS:
        model = study.training_copy(source, factor)
        before = model.state_digest()
        responses.append(study.common_probe(model, images, ids, seed=22, base_config=source.config))
        assert model.state_digest() == before
    for response in responses[1:]:
        for key in response:
            np.testing.assert_array_equal(response[key], responses[0][key])
    reverse = study.common_probe(source, images[::-1], ids[::-1], seed=22, base_config=source.config)
    for key in reverse:
        np.testing.assert_array_equal(reverse[key][::-1], responses[0][key])


def test_silent_common_probe_is_retained_without_retries():
    source = initial_network()
    response = study.common_probe(source, np.zeros((1, 8)), [17], seed=2, base_config=source.config)
    assert response['spikes'].shape == (1, 2)
    assert response['spikes'].sum() == 0
    assert response['attempts'].tolist() == [1]


def test_state_metrics_detect_control_weight_mismatch():
    initial = initial_network()
    models = {f: {a: study.training_copy(initial, f) for a in study.ARMS} for f in study.FACTORS}
    models[2]['normalization_only'].weights[0, 0] += .1
    with pytest.raises(AssertionError):
        study.state_metrics(initial, models)


def test_worker_and_summary_verify_checkpoint_and_probe_artifacts(tmp_path, monkeypatch):
    initial = initial_network()
    folder = tmp_path / 'diagnostic'
    folder.mkdir()
    plan = dict(seeds=[201], initial_rows=[dict(seed=201, checkpoint='original', checkpoint_metadata={},
        state_sha256=initial.state_digest())], base_dt_ms=.5, milestones=[10,25,50], training_images=50,
        records={'201': []}, probe_ids=[71,72], poisson_seed_offset=100000, probe_factor=4)
    study.atomic_json(folder/'plan.json', plan)
    np.savez_compressed(folder/'images.npz', train_images=np.full((2,8),255.), probe_images=np.full((2,8),255.))
    monkeypatch.setattr(study, 'checked_plan', lambda p: plan)
    original_load = study.ReferenceNetwork.load
    monkeypatch.setattr(study.ReferenceNetwork, 'load', lambda path, metadata:
        copy.deepcopy(initial) if path == 'original' else original_load(path, metadata))
    def train_fixture(source, images, records, *, seed, callback):
        models = {f: {a: study.training_copy(source, f) for a in study.ARMS} for f in study.FACTORS}
        history = [{'position': i} for i in range(50)]
        for milestone in plan['milestones']:
            callback(milestone, models, history[:milestone])
        return models, history
    monkeypatch.setattr(study, 'train_coupled', train_fixture)
    study.run_seed(folder, 201)
    study.summarize(folder)
    result = json.loads((folder/'summary.json').read_text())
    assert result['complete'] and result['source_unchanged']
    assert len(result['results'][0]['checkpoints']) == 6
    assert result['results'][0]['coarse_training_exact_replay']
    assert result['results'][0]['probe_metrics']['grid_changes']['1_to_4']['stdp_normalized']['relative_spike_l1'] == 0.
    with pytest.raises(FileExistsError):
        study.run_seed(folder, 201)
    with pytest.raises(FileExistsError):
        study.summarize(folder)
