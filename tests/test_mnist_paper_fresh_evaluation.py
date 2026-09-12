import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples import mnist_paper_fresh_evaluation as fresh
from examples import mnist_paper_readout_tuning as tuning


def test_historic_ids_match_actual_loader_including_test_rng(monkeypatch):
    from examples import mnist_benchmark as mn
    labels = np.arange(60100) % 10
    pixels = np.arange(60100, dtype=float)[:, None]
    monkeypatch.setattr(mn, 'fetch_openml', lambda *a, **k: SimpleNamespace(data=pixels, target=labels))
    monkeypatch.setattr(mn, 'downsample_images', lambda x, **kwargs: x)
    train, _, _, _ = mn.load_reduced_mnist(train_per_class=10, test_per_class=5, seed=73)
    expected = sorted(np.rint(train[:, 0] * 255).astype(int).tolist())
    assert fresh.legacy_training_ids(labels, seed=73, per_class=10) == expected


def test_fresh_ids_are_balanced_disjoint_reproducible_and_not_test_rows():
    images = np.arange(220, dtype=float)[:, None]
    labels = np.arange(220) % 10
    excluded = list(range(20))
    ids, counts = fresh.fresh_ids(images, labels, excluded, per_class=3, boundary=200)
    assert (ids, counts) == fresh.fresh_ids(images, labels, excluded[::-1], per_class=3, boundary=200)
    assert len(ids) == len(set(ids)) == 30
    assert not set(ids) & set(excluded)
    assert max(ids) < 200
    np.testing.assert_array_equal(np.bincount(labels[ids]), np.full(10, 3))


def test_duplicate_images_are_removed_across_exclusions_and_selected_rows():
    images = np.arange(200, dtype=float)[:, None]
    labels = np.arange(200) % 10
    images[20] = images[0]
    images[40] = images[30]
    ids, _ = fresh.fresh_ids(images, labels, [0], per_class=10, boundary=200)
    assert 20 not in ids
    assert len(fresh.pixel_hashes(images[ids])) == len(set(fresh.pixel_hashes(images[ids])))
    assert fresh.pixel_hashes(images[[0]])[0] not in fresh.pixel_hashes(images[ids])


@pytest.mark.parametrize('excluded', [[-1], [200], [True], [1.5]])
def test_invalid_exclusions_fail(excluded):
    with pytest.raises(ValueError):
        fresh.fresh_ids(np.arange(200)[:, None], np.arange(200) % 10, excluded, boundary=200)


def test_fresh_ids_never_silently_shrink_a_class():
    with pytest.raises(ValueError, match='Not enough fresh'):
        fresh.fresh_ids(np.arange(200)[:, None], np.arange(200) % 10, [], per_class=21, boundary=200)


@pytest.mark.parametrize('pixels', [np.array([[np.nan]]), np.array([[-1]]), np.array([[256]]), np.array([[.5]])])
def test_pixel_hashing_rejects_lossy_conversion(pixels):
    with pytest.raises(ValueError):
        fresh.pixel_hashes(pixels)


def test_bootstrap_keeps_seeds_paired_instead_of_inflating_sample_size():
    labels = np.arange(100) % 10
    row = np.arange(100) % 3 == 0
    one = fresh.paired_interval(row[None, :], labels, repeats=100)
    repeated = fresh.paired_interval(np.tile(row, (3, 1)), labels, repeats=100)
    assert one['mean_percent_or_pp'] == repeated['mean_percent_or_pp']
    assert one['conditional_image_bootstrap_95_interval'] == repeated['conditional_image_bootstrap_95_interval']
    opposite = fresh.paired_interval(np.stack((row.astype(float), -row.astype(float))), labels, repeats=100)
    assert opposite['mean_percent_or_pp'] == 0
    assert opposite['conditional_image_bootstrap_95_interval'] == [0., 0.]


def test_bootstrap_preserves_class_balance():
    labels = np.arange(100) % 10
    result = fresh.paired_interval((labels < 3)[None, :], labels, repeats=100)
    assert result['mean_percent_or_pp'] == 30
    np.testing.assert_allclose(result['conditional_image_bootstrap_95_interval'], [30., 30.])


def test_frozen_class_assignments_keep_silent_neurons_unassigned():
    spikes = np.array([[0, 0, 50], [3, 1, 50], [1, 5, 50]])
    np.testing.assert_array_equal(fresh.class_predictions(spikes, [0, 1, -1]), [-1, 0, 1])


def decoder_fixture(tmp_path):
    path = tmp_path / 'decoders.npz'
    state = {'mean': np.zeros(4), 'scale': np.ones(4), 'weights': np.ones(4),
             'coef': np.array([[1., -1., 0., 0.]]), 'intercept': np.zeros(1), 'classes': np.array([0, 1])}
    identity = {'plan_sha256': 'tuning-plan', 'selection_sha256': 'selection', 'network_state_sha256': 'frozen'}
    arrays = {f'{m}__{k}': v for m in fresh.DECODERS[:-1] for k, v in state.items()}
    np.savez_compressed(path, **arrays, identity=json.dumps(identity))
    row = {'decoder_path': str(path), 'decoder_sha256': fresh.sha256(path), 'state_sha256': 'frozen'}
    plan = {'tuning_plan_sha256': 'tuning-plan', 'selection_sha256': 'selection'}
    return row, plan


def test_decoder_identity_and_immutability(tmp_path):
    row, plan = decoder_fixture(tmp_path)
    states = fresh.load_decoders(row, plan)
    with pytest.raises(ValueError):
        states['joint_cv']['coef'][0, 0] = 7
    with pytest.raises(ValueError, match='identity'):
        fresh.load_decoders(dict(row, state_sha256='wrong-network'), plan)
    with pytest.raises(ValueError, match='changed'):
        fresh.load_decoders(dict(row, decoder_sha256='wrong-hash'), plan)


def test_run_seed_replays_old_rows_and_uses_frozen_decoders_without_fit(tmp_path, monkeypatch):
    row, plan = decoder_fixture(tmp_path)
    ids, old_ids, replay_ids = np.arange(6), np.arange(200, 204), np.arange(100, 105)
    def responses(row_ids):
        spikes = np.c_[row_ids % 3 + 1, row_ids % 2 + 1]
        return spikes, spikes.astype(float) - 60, np.ones(len(row_ids), dtype=int)
    old, replay = responses(old_ids), responses(replay_ids)
    source_cache = tmp_path / 'old-responses.npz'
    np.savez_compressed(source_cache, validation_spikes=old[0], validation_voltages=old[1],
        readout_spikes=replay[0], readout_voltages=replay[1], readout_attempts=replay[2])
    old_pred = fresh.predict_all(fresh.load_decoders(row, plan), [0, 1], old[0], old[1])
    plan.update(seeds=[201], rows=[dict(row, seed=201, condition=c, checkpoint='unused', checkpoint_metadata={},
        source_cache=str(source_cache), assignments=[0, 1], old_predictions={k:v.tolist() for k,v in old_pred.items()})
        for c in fresh.CONDITIONS], chunk_size=2, poisson_seed_offset=100000,
        row_ids=ids.tolist(), labels=(ids % 2).tolist(), bootstrap_seed=17, bootstrap_repeats=100)
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    np.savez_compressed(tmp_path / 'images.npz', images=ids[:, None], labels=ids % 2, row_ids=ids,
                        replay_images=replay_ids[:, None], replay_ids=replay_ids)
    monkeypatch.setattr(fresh, 'checked_plan', lambda output: plan)
    monkeypatch.setattr(fresh.ReferenceNetwork, 'load', lambda *a: SimpleNamespace(state_digest=lambda: 'frozen'))
    calls = []
    def frozen(network, images, row_ids, *, seed):
        np.testing.assert_array_equal(images[:, 0], row_ids)
        calls.append(list(row_ids))
        assert seed == 100201
        return responses(row_ids)
    monkeypatch.setattr(fresh, 'frozen_responses', frozen)
    monkeypatch.setattr(tuning, 'fit_state', lambda *a, **k: pytest.fail('Decoder was refitted'))
    fresh.run_seed(tmp_path, 201)
    saved = json.loads((tmp_path / 'seed_201/summary.json').read_text())
    assert saved['complete'] and saved['source_unchanged']
    assert len(calls) == 12
    for result in saved['results'].values():
        assert result['old_response_replay_rows'] == 5
        assert result['old_predictions_reproduced']
    fresh.summarize(tmp_path)
    aggregate = json.loads((tmp_path / 'summary.json').read_text())
    assert aggregate['complete'] and aggregate['source_unchanged']
    assert aggregate['planned_estimates']['stdp_minus_initial']['mean_percent_or_pp'] == 0
    with pytest.raises(FileExistsError):
        fresh.summarize(tmp_path)
    with pytest.raises(FileExistsError):
        fresh.run_seed(tmp_path, 201)
