import numpy as np
import pytest

from examples.mnist_benchmark import ReadoutResponses
from examples.mnist_readout_subset_check import (
    balanced_subsets, feature_matrix, summarize_counts, verify_original_readout,
)


def test_balanced_subsets_keep_100_unique_labels_and_reproduce():
    labels = np.arange(1000) % 10
    before = labels.copy()
    subsets = balanced_subsets(labels)
    np.testing.assert_array_equal(subsets, balanced_subsets(labels))
    assert subsets.shape == (100, 100)
    for subset in subsets:
        assert len(set(subset)) == 100
        assert subset.min() >= 0 and subset.max() < len(labels)
        np.testing.assert_array_equal(np.bincount(labels[subset]), np.full(10, 10))
    assert not np.array_equal(subsets, balanced_subsets(labels, seed=42))
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize("labels,kwargs", [
    ([], {}), ([[0, 1]], {}), ([0.] * 100, {}), (np.arange(90) % 10, {}),
    (np.arange(100) % 10, {"repeats": 0}), (np.arange(100) % 10, {"per_class": True}),
])
def test_invalid_subset_input_rejected(labels, kwargs):
    with pytest.raises(ValueError):
        balanced_subsets(labels, **kwargs)


def test_response_verification_preserves_order_and_requires_exact_bytes():
    pool = ReadoutResponses(np.arange(4), np.arange(20, dtype=np.int32).reshape(5, 4),
                            np.arange(20, dtype=np.float32).reshape(5, 4))
    ids = np.array([3, 0, 2])
    original = ReadoutResponses(pool.exc_indices.copy(), pool.spikes[ids], pool.voltages[ids])
    verify_original_readout(pool, original, ids)
    with pytest.raises(ValueError, match="bytes"):
        verify_original_readout(pool, original, ids[::-1])
    original.voltages[0, 0] += .01
    with pytest.raises(ValueError, match="voltages"):
        verify_original_readout(pool, original, ids)


def test_summary_pairs_seed_counts_before_counting_positive_subsets():
    counts = {"initial": [[1, 2, 3], [1, 2, 3]],
              "normalization_only": [[4, 4, 4], [4, 4, 4]],
              "stdp_normalized": [[6, 3, 5], [3, 4, 3]]}
    result = summarize_counts(counts, 10)
    paired = result["contrasts"]["normalization_only"]
    assert paired["subset_mean_gains_pp"] == [5., -5., 0.]
    assert paired["mean_gain_pp"] == 0.
    assert paired["positive_subsets"] == paired["zero_subsets"] == paired["negative_subsets"] == 1
    assert paired["all_seeds_positive_subsets"] == 0
    assert paired["per_seed_positive_subsets"] == [2, 0]
    assert result["mean_accuracy"]["normalization_only"] == .4
    np.testing.assert_allclose(result["contrasts"]["initial"]["mean_gain_pp"], 20.)


@pytest.mark.parametrize("counts", [
    {"initial": [[1]]},
    {"initial": [[1]], "normalization_only": [[1]], "stdp_normalized": [[11]]},
    {"initial": [[1]], "normalization_only": [[1]], "stdp_normalized": [[1, 2]]},
    {"initial": [[1]], "normalization_only": [[1]], "stdp_normalized": [[1.5]]},
])
def test_summary_rejects_unpaired_or_invalid_counts(counts):
    with pytest.raises(ValueError):
        summarize_counts(counts, 10)


def test_feature_modes_are_read_only_and_explicit():
    response = ReadoutResponses(np.arange(2), np.array([[1, 2]]), np.array([[3., 5.]]))
    assert feature_matrix(response, "ridge_spikes") is response.spikes
    np.testing.assert_array_equal(feature_matrix(response, "ridge_spikes_centered_voltage"), [[1, 2, -1, 1]])
    np.testing.assert_array_equal(response.voltages, [[3., 5.]])
    with pytest.raises(ValueError, match="Unknown decoder"):
        feature_matrix(response, "typo")
