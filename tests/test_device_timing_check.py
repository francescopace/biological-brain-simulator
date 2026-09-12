"""Checks for the fixed-model CPU/MPS timing harness."""

import copy

import pytest
import torch

from examples import mnist_benchmark as mn
from examples.device_timing_check import place_fixture, targets, tensor_digest
from examples.mnist_optimization_check import simulation_digest
from src.device import DEVICE


@pytest.fixture
def tiny_brain(monkeypatch):
    for key, value in (("N_INPUT", 8), ("N_CORTEX_EXC", 4), ("N_CORTEX_INH", 4)):
        monkeypatch.setattr(mn, key, value)
    return mn.build_brain(seed=101, exc_to_inh_weight=64.0)


def test_placement_preserves_every_initial_tensor_and_source(tiny_brain):
    original = simulation_digest(tiny_brain)
    expected = tensor_digest(tiny_brain)
    model = place_fixture(copy.deepcopy(tiny_brain), torch.device("cpu"))
    assert tensor_digest(model) == expected
    place_fixture(model, DEVICE)
    assert tensor_digest(model) == expected
    for target in targets(model):
        for value in vars(target).values():
            if isinstance(value, (torch.Tensor, torch.Generator)):
                assert value.device.type == DEVICE.type
    assert simulation_digest(tiny_brain) == original


def test_placement_rejects_nonfixture_state(tiny_brain):
    tiny_brain.oscillators.enabled = True
    with pytest.raises(ValueError, match="fresh MNIST"):
        place_fixture(tiny_brain, DEVICE)


def test_tensor_digest_detects_topology_and_weight_changes(tiny_brain):
    expected = tensor_digest(tiny_brain)
    model = copy.deepcopy(tiny_brain)
    model.regions["cortex"].syn_pre[0] += 1
    assert tensor_digest(model) != expected
    model = copy.deepcopy(tiny_brain)
    model.regions["cortex"].syn_weight[0] += 1
    assert tensor_digest(model) != expected
