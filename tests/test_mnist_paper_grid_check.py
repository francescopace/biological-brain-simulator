import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from examples import mnist_paper_grid_check as grid
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork, frozen_responses


def small_network(**kwargs):
    config = replace(ReferenceConfig(dynamics_version=1, integration_substeps=1), n_input=4, n_exc=2, incoming_sum=2,
                     presentation_ms=20., rest_ms=5., input_delay_max=2., **kwargs)
    network = ReferenceNetwork(config, seed=21)
    network.weights[:] = .7
    return network


def test_historical_grid_protocol_rejects_corrected_dynamics_before_collecting_responses():
    source = small_network()
    source.config = replace(source.config, dynamics_version=2, integration_substeps=8)
    with pytest.raises(ValueError, match="requires v1 dynamics"):
        grid.paired_responses(source, np.full((1, 4), 100.), [10], [1], seed=2)


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_lift_preserves_event_counts_and_physical_times(factor):
    tape = np.array([[True, False], [False, True], [True, True]])
    lifted = grid.lift_tape(tape, factor)
    np.testing.assert_array_equal(lifted[::factor], tape)
    np.testing.assert_array_equal(np.nonzero(lifted)[0] * .5 / factor, np.nonzero(tape)[0] * .5)
    np.testing.assert_array_equal(np.nonzero(lifted)[1], np.nonzero(tape)[1])
    assert lifted.sum() == tape.sum()
    assert len(lifted) * .5 / factor == len(tape) * .5


@pytest.mark.parametrize('factor', [0, 3, -1, True, 2.])
def test_unsupported_grid_factors_fail(factor):
    with pytest.raises(ValueError, match='factor'):
        grid.lift_tape(np.ones((2, 4), bool), factor)
    with pytest.raises(ValueError, match='factor'):
        grid.refined_network(small_network(), factor)


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_refinement_changes_only_dt_delay_indices_and_transients(factor):
    source = small_network()
    source.step_count = 100
    before = source.state_digest()
    refined = grid.refined_network(source, factor)
    assert refined.config == replace(source.config, dt=source.config.dt / factor)
    assert refined.step_count == 0
    np.testing.assert_array_equal(refined.delays * refined.config.dt, source.delays * source.config.dt)
    np.testing.assert_array_equal(refined.theta, source.theta)
    np.testing.assert_array_equal(refined.weights, source.weights)
    refined.weights[0, 0] = 0.
    assert source.state_digest() == before


def test_subset_is_balanced_deterministic_unique_and_selected_without_predictions():
    labels = np.tile(np.arange(10), 100)
    indices = grid.balanced_indices(labels)
    assert indices == grid.balanced_indices(labels)
    assert len(indices) == len(set(indices)) == 100
    np.testing.assert_array_equal(np.bincount(labels[indices]), np.full(10, 10))
    assert indices != grid.balanced_indices(labels, seed=20260915)
    with pytest.raises(ValueError, match='Not enough'):
        grid.balanced_indices(labels, per_class=101)


def test_real_coarse_grid_exactly_replays_existing_frozen_runner():
    source = small_network(min_spikes=1, max_attempts=10)
    before = source.state_digest()
    images = np.array([[255., 255., 255., 255.], [200., 100., 250., 150.]])
    ids = np.array([431, 724])
    spikes, volts, attempts = frozen_responses(source, images, ids, seed=202)
    result = grid.paired_responses(source, images, ids, attempts, seed=202)
    baseline = dict(spikes=spikes, voltages=volts, attempts=attempts, failed=np.zeros(2, bool))
    grid.check_replay(result, baseline)
    assert source.state_digest() == before


class ScriptedGrid:
    def __init__(self, config, counts):
        self.config, self.counts = config, counts
        self.weights, self.theta = np.ones((4, 2)), np.ones(2)
        self.delays = np.zeros((4, 2), dtype=int)
        self.reset_transients()

    def reset_transients(self):
        self.attempt = 0

    def advance(self, tape, *, learn, adapt):
        assert learn is False and adapt is False
        self.attempt += 1
        count = self.counts[self.attempt - 1]
        return np.array([count, 0]), np.zeros(2, dtype=int), np.full(2, float(self.attempt))

    def rest(self, *, learn, adapt):
        assert learn is False and adapt is False

    def state_digest(self):
        return 'scripted-frozen'


def test_retry_policies_share_prefix_but_keep_different_accepted_attempts(monkeypatch):
    config = small_network(max_attempts=4).config
    source = ScriptedGrid(config, [0, 5, 20, 30])
    counts = {1: [0, 5, 20, 30], 2: [7, 3, 20, 30], 4: [0, 2, 8, 30]}
    monkeypatch.setattr(grid, 'refined_network', lambda n, f: ScriptedGrid(replace(config, dt=config.dt / f), counts[f]))
    result = grid.paired_responses(source, np.full((1, 4), 100.), [10], [2], seed=2)
    for factor, fixed_count, adaptive_count, accepted in ((1, 5, 5, 2), (2, 3, 7, 1), (4, 2, 8, 3)):
        fixed, adaptive = result['fixed_exposure'][factor], result['adaptive_retry'][factor]
        assert fixed['spikes'][0, 0] == fixed_count
        assert fixed['attempts'][0] == 2
        assert adaptive['spikes'][0, 0] == adaptive_count
        assert adaptive['attempts'][0] == accepted
        assert not adaptive['failed'].any()
        np.testing.assert_array_equal(adaptive['voltages'], [[accepted, accepted]])


def test_exhausted_retry_is_explicit_and_predictions_abstain_without_dropping_rows(monkeypatch):
    config = small_network(max_attempts=3).config
    source = ScriptedGrid(config, [0, 1, 2])
    monkeypatch.setattr(grid, 'refined_network', lambda n, f: ScriptedGrid(replace(config, dt=config.dt / f), [0, 1, 2]))
    result = grid.paired_responses(source, np.full((1, 4), 100.), [10], [1], seed=2)
    for factor in grid.FACTORS:
        fixed, adaptive = result['fixed_exposure'][factor], result['adaptive_retry'][factor]
        assert not fixed['failed'].any()
        assert adaptive['failed'].tolist() == [True]
        assert adaptive['attempts'].tolist() == [3]
        monkeypatch.setattr(grid.fresh, 'predict_all', lambda *args: {'joint_cv': np.array([7])})
        pred = grid.predictions({}, [0, 1], adaptive)['joint_cv']
        assert pred.tolist() == [-1]
        assert (pred == np.array([7])).mean() == 0.


@pytest.mark.parametrize('attempts', [[0], [11], [1.5], [True]])
def test_invalid_original_exposure_fails(attempts):
    with pytest.raises(ValueError, match='retry counts'):
        grid.paired_responses(small_network(), np.full((1, 4), 100.), [10], attempts, seed=2)


def test_response_metrics_are_zero_on_replay_and_voltage_centering_removes_common_offset():
    source = dict(spikes=np.array([[1, 2], [0, 1]]), voltages=np.array([[-60., -55.], [-62., -57.]]),
                  attempts=np.array([1, 2]))
    same = grid.response_difference(source, copy.deepcopy(source))
    assert same == dict(relative_spike_l1=0., same_spike_vector_fraction=1., centered_voltage_rmse_mv=0.,
                        mean_total_spike_change=0., changed_attempt_fraction=0.)
    target = copy.deepcopy(source)
    target['voltages'] += 10.
    assert grid.response_difference(source, target)['centered_voltage_rmse_mv'] == 0.
    target['spikes'][0, 1] += 4
    target['attempts'][0] = 2
    diff = grid.response_difference(source, target)
    assert diff['relative_spike_l1'] == 1.
    assert diff['same_spike_vector_fraction'] == .5
    assert diff['changed_attempt_fraction'] == .5


def test_coarse_replay_rejects_any_changed_response():
    baseline = dict(spikes=np.ones((1, 2), dtype=int), voltages=np.ones((1, 2)),
                    attempts=np.ones(1, dtype=int), failed=np.zeros(1, bool))
    responses = {p: {1: copy.deepcopy(baseline)} for p in grid.POLICIES}
    responses['adaptive_retry'][1]['voltages'][0, 0] += 1e-12
    with pytest.raises(AssertionError):
        grid.check_replay(responses, baseline)


def test_worker_and_summary_replay_frozen_predictions_and_paired_grid_changes(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    output = tmp_path / 'study'
    output.mkdir()
    network = small_network()
    labels, ids = np.arange(10), np.arange(10) + 100
    baseline = dict(spikes=np.tile([5, 0], (10, 1)), voltages=np.zeros((10, 2)),
                    attempts=np.ones(10, dtype=int), failed=np.zeros(10, bool))
    baseline['spikes'][1] = [0, 5]
    np.savez_compressed(source / 'images.npz', images=np.full((10, 4), 255.))
    rows = []
    for condition in grid.fresh.CONDITIONS:
        cache = source / f'{condition}.npz'
        np.savez_compressed(cache, **{k: baseline[k] for k in grid.FIELDS[:-1]}, row_ids=ids, labels=labels,
            identity=json.dumps({'plan_sha256': 'fresh-plan', 'state_sha256': network.state_digest()}))
        rows.append(dict(seed=201, condition=condition, checkpoint='unused', checkpoint_metadata={},
            state_sha256=network.state_digest(), fresh_cache=str(cache), assignments=[0, 1],
            baseline_predictions={m: baseline['spikes'].argmax(axis=1).tolist() for m in grid.fresh.DECODERS}))
    plan = dict(seeds=[201], rows=rows, fresh_root=str(source), fresh_plan_sha256='fresh-plan',
                indices=list(range(10)), row_ids=ids.tolist(), labels=labels.tolist(), base_dt_ms=.5,
                poisson_seed_offset=100000, bootstrap_repeats=100, bootstrap_seed=71)
    grid.atomic_json(output / 'plan.json', plan)
    monkeypatch.setattr(grid, 'checked_plan', lambda root: plan)
    monkeypatch.setattr(grid.ReferenceNetwork, 'load', lambda *args: copy.deepcopy(network))
    monkeypatch.setattr(grid.fresh, 'load_decoders', lambda *args: {})
    monkeypatch.setattr(grid.fresh, 'predict_all', lambda states, assignments, spikes, volts:
                        {m: spikes.argmax(axis=1) for m in grid.fresh.DECODERS})
    def response_fixture(*args, **kwargs):
        response = {p: {f: copy.deepcopy(baseline) for f in grid.FACTORS} for p in grid.POLICIES}
        for p in grid.POLICIES:
            for f in (2, 4):
                response[p][f]['spikes'][0] = [0, 5]
        return response
    monkeypatch.setattr(grid, 'paired_responses', response_fixture)
    grid.run_seed(output, 201)
    grid.summarize(output)
    result = json.loads((output / 'summary.json').read_text())
    assert result['complete'] and result['source_unchanged'] and result['baseline_exact_replay']
    scores = result['metrics']['fixed_exposure']
    assert scores['1']['stdp_normalized']['accuracy']['joint_cv'] == .2
    assert scores['2']['stdp_normalized']['accuracy']['joint_cv'] == .1
    diff = result['grid_changes']['fixed_exposure']['1_to_2']['stdp_normalized']['methods']['joint_cv']
    assert diff['prediction_disagreement_fraction'] == .1
    assert diff['accuracy_delta_pp']['mean_percent_or_pp'] == -10.
    assert result['contrasts']['fixed_exposure']['4']['joint_cv']['initial']['mean_percent_or_pp'] == 0.
    with pytest.raises(FileExistsError):
        grid.run_seed(output, 201)
    with pytest.raises(FileExistsError):
        grid.summarize(output)
