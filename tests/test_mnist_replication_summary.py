"""Paired uncertainty uses shared images and retains per-seed effects."""

import numpy as np
import pytest

from examples.mnist_replication_summary import paired_effect


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_constant_difference_has_exact_interval(value):
    result = paired_effect(np.full((3, 20), value), np.arange(20) % 10, resamples=100)
    assert result["mean_gain_pp"] == value * 100
    np.testing.assert_allclose(result["conditional_image_bootstrap_95_pp"], [value * 100] * 2)
    assert result["registered_positive_criterion_met"] == (value > 0)


def test_seeds_are_averaged_within_shared_images():
    differences = np.array([[1] * 20, [0] * 20, [-1] * 20])
    result = paired_effect(differences, np.arange(20) % 10, resamples=100)
    assert result["seed_gains_pp"] == [100, 0, -100]
    assert result["conditional_image_bootstrap_95_pp"] == [0, 0]
    assert not result["registered_positive_criterion_met"]


def test_positive_mean_does_not_hide_negative_seed():
    result = paired_effect([[1] * 20, [1] * 20, [-1] * 20], np.arange(20) % 10, resamples=100)
    assert result["mean_gain_pp"] > 0
    assert result["conditional_image_bootstrap_95_pp"][0] > 0
    assert not result["registered_positive_criterion_met"]


def test_bootstrap_is_reproducible():
    differences = np.random.default_rng(1).integers(-1, 2, (3, 40))
    labels = np.arange(40) % 10
    assert paired_effect(differences, labels, resamples=100) == paired_effect(differences, labels, resamples=100)
