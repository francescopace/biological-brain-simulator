"""Tests for memory system: trace capture, consolidation, pattern completion."""

import torch
import pytest

from src.memory import MemorySystem, MemoryTrace
from src.region import Region, RegionType


@pytest.fixture
def active_region():
    """A region with some neurons showing activity."""
    r = Region("mem", RegionType.MEMORY, max_neurons=20)
    r.populate(10, connectivity=0.0)
    r.activity[:5] = 0.3  # 5 active neurons above threshold
    return r


class TestTraceCapture:
    def test_capture_creates_trace(self, active_region):
        ms = MemorySystem(trace_threshold=0.15)
        trace = ms.capture_trace(active_region, current_time=100.0)
        assert trace is not None
        assert len(trace.neuron_indices) == 5
        assert trace.region_name == "mem"
        assert trace.creation_time == 100.0

    def test_capture_below_threshold(self):
        r = Region("test", RegionType.MEMORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.activity[:5] = 0.01  # below threshold
        ms = MemorySystem(trace_threshold=0.15)
        trace = ms.capture_trace(r, 1.0)
        assert trace is None

    def test_capacity_limit(self, active_region):
        ms = MemorySystem(trace_capacity=3, trace_threshold=0.15)
        for i in range(5):
            ms.capture_trace(active_region, float(i))
        assert len(ms.traces) <= 3

    def test_empty_region_no_trace(self):
        r = Region("empty", RegionType.MEMORY, max_neurons=10)
        ms = MemorySystem()
        assert ms.capture_trace(r, 1.0) is None


class TestConsolidation:
    def test_consolidation_injects_current(self, active_region):
        ms = MemorySystem(consolidation_interval=1, replay_strength=2.0, trace_threshold=0.15)
        ms.capture_trace(active_region, 1.0)

        active_region.current[:] = 0.0
        replayed = ms.consolidate({"mem": active_region}, 2.0)
        assert replayed > 0
        # At least some neurons should have received replay current
        assert torch.any(active_region.current[:10] > 0)

    def test_no_consolidation_without_traces(self, active_region):
        ms = MemorySystem(consolidation_interval=1)
        replayed = ms.consolidate({"mem": active_region}, 1.0)
        assert replayed == 0

    def test_consolidation_only_at_interval(self, active_region):
        ms = MemorySystem(consolidation_interval=100, trace_threshold=0.15)
        ms.capture_trace(active_region, 1.0)
        replayed = ms.consolidate({"mem": active_region}, 2.0)
        assert replayed == 0  # step_counter = 1, not at interval

    def test_replay_strengthens_trace(self, active_region):
        ms = MemorySystem(consolidation_interval=1, trace_threshold=0.15)
        ms.capture_trace(active_region, 1.0)
        initial_strength = ms.traces[0].strength
        ms.consolidate({"mem": active_region}, 2.0)
        assert ms.traces[0].strength > initial_strength


class TestPatternCompletion:
    def test_completes_from_partial(self, active_region):
        ms = MemorySystem(trace_threshold=0.15)
        ms.capture_trace(active_region, 1.0)

        # Provide partial input (first 2 of 5 active neurons)
        partial = torch.tensor([0, 1], dtype=torch.int64)
        active_region.current[:] = 0.0
        activated = ms.pattern_completion(active_region, partial, boost_current=3.0)
        # Should activate the missing neurons from the trace
        assert activated > 0

    def test_no_match_returns_zero(self, active_region):
        ms = MemorySystem(trace_threshold=0.15)
        ms.capture_trace(active_region, 1.0)

        # Unrelated indices
        partial = torch.tensor([8, 9], dtype=torch.int64)
        activated = ms.pattern_completion(active_region, partial)
        assert activated == 0
