import numpy as np
import pytest

from examples.mnist_cached_diagnosis import balanced_deletions, summarize_cohort


def test_balanced_deletions_are_reproducible_unique_and_training_only():
    labels = np.arange(100) % 10
    first, second = balanced_deletions(labels), balanced_deletions(labels)
    assert len(first) == 100
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert len(set(a)) == 90 and a.min() >= 0 and a.max() < 100
        np.testing.assert_array_equal(np.bincount(labels[a]), np.full(10, 9))


def test_invalid_readout_budget_is_rejected():
    with pytest.raises(ValueError, match="ten labelled"):
        balanced_deletions(np.arange(90) % 10)


def test_fixed_decoder_and_refit_components_sum_to_diagonal_difference():
    rows = [{"cross_readouts": {d: {"accuracy_matrix": [[.3, .31], [.35, .4]]} for d in
        ("ridge_spikes", "ridge_spikes_centered_voltage")},
        "label_deletion": {d: {"gain_pp": [-1., 2.]} for d in
        ("ridge_spikes", "ridge_spikes_centered_voltage")}}]
    for result in summarize_cohort(rows).values():
        assert result["fixed_control_decoder_gain_pp"] == pytest.approx(1.)
        assert result["subsequent_refit_component_pp"] == pytest.approx(9.)
        assert result["diagonal_gain_pp"] == pytest.approx(10.)
        assert result["deletion_fraction_positive"] == .5
