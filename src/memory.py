"""
Memory system operating on vectorized Region arrays.

Memory traces store neuron indices (not objects). Consolidation
injects current directly into the region's current array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .region import Region


@dataclass
class MemoryTrace:
    """Activity snapshot stored as neuron indices + activity levels."""
    neuron_indices: np.ndarray
    activity_snapshot: np.ndarray
    region_name: str
    strength: float = 1.0
    replay_count: int = 0
    creation_time: float = 0.0


class MemorySystem:
    def __init__(
        self,
        trace_capacity: int = 100,
        consolidation_interval: int = 5000,
        replay_strength: float = 0.7,
        trace_threshold: float = 0.15,
    ):
        self.trace_capacity = trace_capacity
        self.consolidation_interval = consolidation_interval
        self.replay_strength = replay_strength
        self.trace_threshold = trace_threshold

        self.traces: list[MemoryTrace] = []
        self._step_counter = 0
        self._rng = np.random.default_rng()

    def capture_trace(
        self,
        region: Region,
        current_time: float,
    ) -> MemoryTrace | None:
        n = region.n_neurons
        if n == 0:
            return None

        active = (
            (region.activity[:n] > self.trace_threshold)
            & region.neuron_alive[:n]
        )
        indices = np.where(active)[0]
        if len(indices) < 3:
            return None

        trace = MemoryTrace(
            neuron_indices=indices.copy(),
            activity_snapshot=region.activity[indices].copy(),
            region_name=region.name,
            creation_time=current_time,
        )

        self.traces.append(trace)
        if len(self.traces) > self.trace_capacity:
            self.traces.sort(key=lambda t: t.strength, reverse=True)
            self.traces = self.traces[:self.trace_capacity]

        return trace

    def consolidate(self, regions: dict[str, Region], current_time: float) -> int:
        self._step_counter += 1
        if self._step_counter % self.consolidation_interval != 0:
            return 0
        if not self.traces:
            return 0

        n_replay = min(len(self.traces), 5)
        weights = np.array([t.strength for t in self.traces])
        weights /= weights.sum()
        indices = self._rng.choice(
            len(self.traces), size=n_replay, replace=False, p=weights
        )

        replayed = 0
        for idx in indices:
            trace = self.traces[idx]
            region = regions.get(trace.region_name)
            if region is None:
                continue

            # Inject current into remembered neurons (if still alive)
            valid = (
                (trace.neuron_indices < region.n_neurons)
                & region.neuron_alive[trace.neuron_indices]
            )
            live_idx = trace.neuron_indices[valid]
            if len(live_idx) > 0:
                region.current[live_idx] += self.replay_strength

            trace.replay_count += 1
            trace.strength = min(trace.strength + 0.1, 5.0)
            replayed += 1

        for trace in self.traces:
            if trace.replay_count == 0:
                age = current_time - trace.creation_time
                trace.strength *= max(0.0, 1.0 - age / 100000.0)

        self.traces = [t for t in self.traces if t.strength > 0.05]
        return replayed

    def pattern_completion(
        self,
        region: Region,
        partial_indices: np.ndarray,
        boost_current: float = 2.0,
    ) -> int:
        partial_set = set(partial_indices)
        best_trace = None
        best_overlap = 0

        for trace in self.traces:
            if trace.region_name != region.name:
                continue
            overlap = len(partial_set & set(trace.neuron_indices))
            if overlap > best_overlap:
                best_overlap = overlap
                best_trace = trace

        if best_trace is None or best_overlap < 2:
            return 0

        activated = 0
        for idx in best_trace.neuron_indices:
            if idx not in partial_set and idx < region.n_neurons and region.neuron_alive[idx]:
                region.current[idx] += boost_current
                activated += 1
        return activated
