"""Tests for stimulus encoding strategies."""

import torch
import pytest

from src.stimulus import StimulusEncoder, EncodingStrategy, encode_text, encode_number
from src.region import Region, RegionType


@pytest.fixture
def sensory_region():
    r = Region("input", RegionType.SENSORY, max_neurons=100)
    r.populate(50, connectivity=0.0)
    return r


class TestRateEncoding:
    def test_injects_current(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        count = enc.encode([1.0, 0.5, 0.0], sensory_region)
        assert count > 0
        assert sensory_region.current[0].item() > 0

    def test_zero_input_zero_current(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        enc.encode([0.0] * 50, sensory_region)
        assert torch.all(sensory_region.current[:50] == 0.0)

    def test_1_to_1_fast_path(self):
        """When n_inputs <= n_neurons and neurons_per_input == 1, fast path is used."""
        r = Region("input", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        count = enc.encode([0.8, 0.4, 0.2, 0.1, 0.0], r)
        assert count == 5
        # Higher input → higher current
        assert r.current[0].item() > r.current[3].item()

    def test_handles_tensor_input(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        vals = torch.tensor([0.5, 0.3])
        count = enc.encode(vals, sensory_region)
        assert count > 0

    def test_empty_region(self):
        r = Region("empty", RegionType.SENSORY, max_neurons=10)
        enc = StimulusEncoder()
        count = enc.encode([0.5], r)
        assert count == 0


class TestTemporalEncoding:
    def test_injects_current(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.TEMPORAL, noise_level=0.0)
        count = enc.encode([0.8, 0.5], sensory_region)
        assert count > 0
        assert sensory_region.current[0].item() > 0

    def test_clamps_to_0_1(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.TEMPORAL, noise_level=0.0, max_current=10.0)
        enc.encode([2.0], sensory_region)
        # Value clamped to 1.0 → current should be max_current
        assert sensory_region.current[0].item() == pytest.approx(10.0)


class TestPopulationEncoding:
    def test_injects_gaussian_pattern(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.POPULATION, noise_level=0.0)
        count = enc.encode([0.5], sensory_region)
        assert count > 0
        # Center neurons should have more current than edges
        center = sensory_region.n_neurons // 2
        assert sensory_region.current[center].item() > sensory_region.current[0].item()


class TestHelpers:
    def test_encode_text(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        count = encode_text("AB", sensory_region, enc)
        assert count > 0

    def test_encode_number(self, sensory_region):
        enc = StimulusEncoder(strategy=EncodingStrategy.RATE, noise_level=0.0)
        count = encode_number(0.5, sensory_region, enc)
        assert count > 0
