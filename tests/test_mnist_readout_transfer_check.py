"""Transferring validation rows must not replace fitted preprocessing."""

import copy
import numpy as np
import pytest

from examples.mnist_learning_check import Dataset
from examples.mnist_readout_transfer_check import transferred_dataset


@pytest.fixture
def inputs():
    raw = np.full((30, 4), 255)
    labels = np.arange(30) % 10
    original = Dataset(np.ones((2, 4)), labels[:2], np.array([0, 1]), np.ones((2, 4)), labels[2:4],
        {"canonical_train_boundary": 20, "train_ids": [0, 1], "validation_ids": [2, 3],
         "readout_ids": [0, 1], "intensity_target_l1": 2., "validation_sha256": "old"})
    return raw, labels, original


def test_transfer_keeps_source_preprocessing_readout_and_input_objects(inputs):
    raw, labels, original = inputs
    before = copy.deepcopy(original.manifest)
    target = {"validation_ids": [4, 5], "intensity_target_l1": 999.}
    data = transferred_dataset(raw, labels, original, target)
    assert data.train_X is original.train_X and data.readout_indices is original.readout_indices
    np.testing.assert_array_equal(data.validation_X, np.full((2, 4), .5))
    np.testing.assert_array_equal(data.validation_y, [4, 5])
    assert data.manifest["intensity_target_l1"] == 2.
    assert original.manifest == before


@pytest.mark.parametrize("ids", [[0, 4], [3, 4], [4, 4], [20], [-1], [1.5], [True], []])
def test_transfer_rejects_reused_invalid_or_test_rows(inputs, ids):
    with pytest.raises(ValueError, match="validation rows"):
        transferred_dataset(*inputs, {"validation_ids": ids})
