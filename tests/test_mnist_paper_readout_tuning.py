import json

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from examples import mnist_paper_readout_tuning as tuning


def test_candidate_grid_is_fixed_and_contains_baseline_and_endpoints():
    grid = tuning.candidate_grid()
    assert len(grid) == len({tuple(p.values()) for p in grid}) == 49
    assert tuning.BASELINE in grid
    assert sum(p['spike_weight'] == 0 for p in grid) == 7
    assert sum(p['voltage_weight'] == 0 for p in grid) == 7


def test_features_center_voltage_per_image_without_mutation():
    spikes = np.arange(12).reshape(3, 4)
    volts = -60. + spikes * 2
    before = volts.copy()
    features = tuning.feature_matrix(spikes, volts)
    np.testing.assert_array_equal(features[:, :4], spikes)
    np.testing.assert_allclose(features[:, 4:].mean(axis=1), 0)
    np.testing.assert_array_equal(features, tuning.feature_matrix(spikes, volts + np.arange(3)[:, None] * 40))
    np.testing.assert_array_equal(volts, before)


@pytest.mark.parametrize('spikes,volts', [
    (np.ones(4), np.ones(4)), (np.ones((2, 3)), np.ones((2, 4))),
    (np.empty((0, 4)), np.empty((0, 4))),
    (np.full((2, 3), -1), np.ones((2, 3))),
    (np.full((2, 3), np.nan), np.ones((2, 3))),
    (np.ones((2, 3)), np.full((2, 3), np.inf)),
])
def test_bad_feature_inputs_fail(spikes, volts):
    with pytest.raises(ValueError):
        tuning.feature_matrix(spikes, volts)


@pytest.mark.parametrize('parameter', [
    {'spike_weight': 0., 'voltage_weight': 0., 'alpha': 1.},
    {'spike_weight': -1., 'voltage_weight': 1., 'alpha': 1.},
    {'spike_weight': 1., 'voltage_weight': np.nan, 'alpha': 1.},
    {'spike_weight': 1., 'voltage_weight': 1., 'alpha': 0.},
])
def test_invalid_block_weights_fail(parameter):
    with pytest.raises(ValueError):
        tuning.block_weights(8, parameter)


@pytest.mark.parametrize('classes', [2, 10])
def test_baseline_state_matches_existing_sklearn_pipeline(classes):
    rng = np.random.default_rng(4)
    train, valid = rng.normal(size=(100, 16)), rng.normal(size=(25, 16))
    labels = np.arange(100) % classes
    expected = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1., solver='cholesky'))
    expected.fit(train, labels)
    state = tuning.fit_state(train, labels, tuning.BASELINE)
    np.testing.assert_array_equal(tuning.predict_state(state, valid), expected.predict(valid))


def test_weighting_is_after_standardization_and_state_roundtrips(tmp_path):
    rng = np.random.default_rng(14)
    train, valid = rng.normal(size=(30, 12)), rng.normal(size=(11, 12))
    labels = np.arange(30) % 3
    p = {'spike_weight': 1., 'voltage_weight': 4., 'alpha': 100.}
    scaler = StandardScaler().fit(train)
    w = np.r_[np.ones(6), np.full(6, 4.)]
    model = RidgeClassifier(alpha=100., solver='cholesky').fit(scaler.transform(train) * w, labels)
    state = tuning.fit_state(train, labels, p)
    np.testing.assert_array_equal(state['weights'], w)
    np.testing.assert_allclose(state['coef'], model.coef_)
    path = tmp_path / 'model.npz'
    np.savez_compressed(path, **state)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(tuning.predict_state(saved, valid), model.predict(scaler.transform(valid) * w))


def test_fold_plan_is_reproducible_stratified_and_disjoint():
    labels = np.arange(100) % 10
    folds = tuning.fold_plan(labels)
    assert folds == tuning.fold_plan(labels)
    assert len(folds) == 15
    for repeat in range(3):
        heldout = []
        for fold in folds[repeat*5:(repeat+1)*5]:
            assert not set(fold['train']) & set(fold['heldout'])
            np.testing.assert_array_equal(np.bincount(labels[fold['heldout']]), np.full(10, 2))
            heldout.extend(fold['heldout'])
        assert sorted(heldout) == list(range(100))


def test_fold_plan_rejects_insufficient_class_rows():
    with pytest.raises(ValueError):
        tuning.fold_plan(np.arange(20) % 10)


def test_cv_fits_scaler_only_on_fold_training_rows(monkeypatch):
    rng = np.random.default_rng(18)
    features, labels = rng.normal(size=(30, 8)), np.arange(30) % 3
    before = features.copy()
    folds = tuning.fold_plan(labels, splits=3, repeats=2)
    fitted = []
    class SpyScaler(StandardScaler):
        def fit(self, x, y=None, sample_weight=None):
            fitted.append(x.copy())
            return super().fit(x, y, sample_weight=sample_weight)
    monkeypatch.setattr(tuning, 'StandardScaler', SpyScaler)
    counts, predictions = tuning.cross_validate(features, labels, folds, [tuning.BASELINE])
    assert len(fitted) == len(folds)
    for j, fold in enumerate(folds):
        np.testing.assert_array_equal(fitted[j], features[fold['train']])
        np.testing.assert_array_equal(predictions[0, j, fold['train']], -1)
        expected = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1., solver='cholesky'))
        expected.fit(features[fold['train']], labels[fold['train']])
        np.testing.assert_array_equal(predictions[0, j, fold['heldout']], expected.predict(features[fold['heldout']]))
        assert counts[0, j] == np.count_nonzero(predictions[0, j, fold['heldout']] == labels[fold['heldout']])
    np.testing.assert_array_equal(features, before)


def test_cv_rejects_overlapping_fold():
    with pytest.raises(ValueError, match='disjoint'):
        tuning.cross_validate(np.ones((4, 8)), np.arange(4) % 2,
            [{'train': [0, 1, 2], 'heldout': [2, 3]}], [tuning.BASELINE])


def test_exact_ties_prefer_stronger_regularization_then_balance():
    grid = tuning.candidate_grid()
    counts = np.ones((49, 15), dtype=int)
    choice = tuning.select_candidates(counts, grid)
    assert grid[choice['joint_cv']] == {'spike_weight': 1., 'voltage_weight': 1., 'alpha': 10000.}
    counts[0, 0] += 1
    assert tuning.select_candidates(counts, grid)['joint_cv'] == 0
    assert grid[choice['voltage_only_cv']]['spike_weight'] == 0


def test_selection_runs_without_any_validation_arrays(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    cache = tmp_path / 'readout-only.npz'
    labels = np.arange(100) % 10
    np.savez_compressed(cache, readout_y=labels, readout_spikes=rng.poisson(2, (100, 4)),
                        readout_voltages=rng.normal(size=(100, 4)))
    plan = {'rows': [{'seed': 201, 'condition': 'stdp_normalized', 'cache': str(cache)}],
            'readout_y': labels.tolist(), 'candidates': tuning.candidate_grid(),
            'folds': tuning.fold_plan(labels)}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(tuning, 'checked_plan', lambda output: plan)
    before = tuning.sha256(cache)
    tuning.select(tmp_path)
    saved = json.loads((tmp_path / 'selection.json').read_text())
    assert saved['complete']
    assert len(saved['results']) == 1
    assert tuning.sha256(cache) == before
    with pytest.raises(FileExistsError):
        tuning.select(tmp_path)


def test_select_evaluate_pipeline_exports_replayable_decoders(tmp_path, monkeypatch):
    rng = np.random.default_rng(31)
    source = tmp_path / 'source'
    source.mkdir()
    labels, valid_y = np.arange(100) % 10, np.arange(40) % 10
    (source / 'plan.json').write_text(json.dumps({'validation_y': valid_y.tolist()}))
    plan = {'readout_y': labels.tolist(), 'readout_ids': list(range(100)),
            'validation_ids': list(range(100, 140)), 'source_root': str(source),
            'candidates': tuning.candidate_grid(), 'folds': tuning.fold_plan(labels),
            'baseline': tuning.BASELINE, 'rows': []}
    for condition in tuning.CONDITIONS:
        spikes, volts = rng.poisson(2, (100, 4)), rng.normal(size=(100, 4))
        vs, vv = rng.poisson(2, (40, 4)), rng.normal(size=(40, 4))
        cache = source / f'{condition}.npz'
        np.savez_compressed(cache, readout_y=labels, validation_y=valid_y,
            readout_ids=plan['readout_ids'], validation_ids=plan['validation_ids'],
            readout_spikes=spikes, readout_voltages=volts, validation_spikes=vs, validation_voltages=vv)
        baseline = tuning.fit_state(tuning.feature_matrix(spikes, volts), labels, tuning.BASELINE)
        plan['rows'].append({'seed': 201, 'condition': condition, 'cache': str(cache),
            'baseline_predictions': tuning.predict_state(baseline, tuning.feature_matrix(vs, vv)).tolist(),
            'state_sha256': 'synthetic-test-state'})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(tuning, 'checked_plan', lambda output: plan)
    tuning.select(tmp_path)
    frozen_selection = (tmp_path / 'selection.json').read_bytes()
    tuning.evaluate(tmp_path)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['complete'] and result['source_unchanged']
    assert len(result['results']) == 3
    assert (tmp_path / 'selection.json').read_bytes() == frozen_selection
    for row in result['results']:
        assert set(row['methods']) == {'fixed_baseline', *tuning.METHODS}
        assert tuning.sha256(row['decoder_path']) == row['decoder_sha256']
    with pytest.raises(FileExistsError):
        tuning.evaluate(tmp_path)
