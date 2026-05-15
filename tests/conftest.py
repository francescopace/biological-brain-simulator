"""Shared fixtures for the test suite."""

import pytest
import torch

from src.brain import Brain
from src.region import Region, RegionType


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def region(seed):
    """A small SENSORY region with 20 neurons and sparse internal wiring."""
    r = Region("test", RegionType.SENSORY, max_neurons=100)
    r._rng.manual_seed(seed)
    r.populate(20, connectivity=0.1)
    return r


@pytest.fixture
def brain(seed):
    """A minimal two-region brain with a projection."""
    b = Brain(seed=seed)
    b.add_region("input", RegionType.SENSORY, n_neurons=10, connectivity=0.0, max_neurons=50)
    b.add_region("output", RegionType.MOTOR, n_neurons=10, connectivity=0.1, max_neurons=50)
    b.connect_regions("input", "output", density=0.3)
    return b
