import copy
from dataclasses import replace

import numpy as np
import pytest

from examples.mnist_paper_check import decoders, train_pair, weight_diagnostics
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork


def test_zero_learning_rate_paired_training_replays_exactly():
    config = replace(ReferenceConfig(), n_input=8, n_exc=2, incoming_sum=4.,
                     min_spikes=1, eta_pre=0., eta_post=0.)
    left = ReferenceNetwork(config)
    right = copy.deepcopy(left)
    images = np.array([[255.] * 8, [230.] * 8])
    checkpoints = []
    records = train_pair(left, right, images, seed=42,
                        callback=lambda n, rows: checkpoints.append((n, len(rows))))
    assert left.state_digest() == right.state_digest()
    assert checkpoints == [(1, 1), (2, 2)]
    assert sorted(r['training_index'] for r in records) == [0, 1]
    attempts = sum(r['attempts'] for r in records)
    assert left.step_count == attempts * round((config.presentation_ms + config.rest_ms) / config.dt)


def test_paired_training_rejects_different_initial_states():
    a, b = ReferenceNetwork(), ReferenceNetwork(seed=42)
    with pytest.raises(ValueError, match='identical'):
        train_pair(a, b, np.ones((1, 784)), seed=42)


def test_common_decoders_do_not_fit_validation_labels():
    rng = np.random.default_rng(3)
    train, valid = rng.poisson(2, (100, 12)), rng.poisson(2, (20, 12))
    tv, vv = rng.normal(size=(100, 12)), rng.normal(size=(20, 12))
    first = decoders(train, tv, np.arange(100) % 10, valid, vv, np.zeros(20, dtype=int))
    second = decoders(train, tv, np.arange(100) % 10, valid, vv, np.ones(20, dtype=int))
    for decoder in first:
        assert first[decoder]['predictions'] == second[decoder]['predictions']
    assert first['class_average']['assignments'] == second['class_average']['assignments']


def test_weight_diagnostics_are_read_only_and_report_identity():
    model = ReferenceNetwork()
    before = model.state_digest()
    diagnostics = weight_diagnostics(model, model)
    assert diagnostics['relative_l1_from_initial'] == 0
    assert 0 < diagnostics['mean_off_diagonal_weight_cosine'] < 1
    assert 0 < diagnostics['mean_normalized_weight_entropy'] < 1
    assert model.state_digest() == before
