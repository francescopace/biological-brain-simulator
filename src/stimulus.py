"""
Stimulus encoder: converts external inputs into currents injected
directly into a Region's current array (vectorized).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .region import Region


class EncodingStrategy(enum.Enum):
    RATE = "rate"
    TEMPORAL = "temporal"
    POPULATION = "population"


@dataclass
class StimulusEncoder:
    strategy: EncodingStrategy = EncodingStrategy.RATE
    max_current: float = 15.0
    noise_level: float = 0.5
    _rng: np.random.Generator = None

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng()

    def encode(
        self,
        values: np.ndarray | list[float],
        region: Region,
        current_time: float = 0.0,
    ) -> int:
        values = np.asarray(values, dtype=np.float64)
        n = region.n_neurons
        if n == 0:
            return 0

        if self.strategy == EncodingStrategy.RATE:
            return self._rate_encode(values, region)
        elif self.strategy == EncodingStrategy.TEMPORAL:
            return self._temporal_encode(values, region)
        elif self.strategy == EncodingStrategy.POPULATION:
            return self._population_encode(values, region)
        return 0

    def _rate_encode(self, values: np.ndarray, region: Region) -> int:
        n = region.n_neurons
        n_inputs = len(values)
        neurons_per_input = max(1, n // max(n_inputs, 1))
        stimulated = 0

        for i, val in enumerate(values):
            s = i * neurons_per_input
            e = min(s + neurons_per_input, n)
            count = e - s
            if count <= 0:
                continue
            current = val * self.max_current
            noise = self._rng.normal(0, self.noise_level, size=count)
            region.current[s:e] += np.maximum(0.0, current + noise)
            stimulated += count

        return stimulated

    def _temporal_encode(self, values: np.ndarray, region: Region) -> int:
        n = region.n_neurons
        n_inputs = len(values)
        neurons_per_input = max(1, n // max(n_inputs, 1))
        stimulated = 0

        for i, val in enumerate(values):
            s = i * neurons_per_input
            e = min(s + neurons_per_input, n)
            count = e - s
            if count <= 0:
                continue
            current = np.clip(val, 0, 1) * self.max_current
            noise = self._rng.normal(0, self.noise_level * 0.5, size=count)
            region.current[s:e] += np.maximum(0.0, current + noise)
            stimulated += count

        return stimulated

    def _population_encode(self, values: np.ndarray, region: Region) -> int:
        n = region.n_neurons
        stimulated = 0
        sigma = max(1, n // 10)

        indices = np.arange(n)
        for val in values:
            center = int(np.clip(val, 0, 1) * (n - 1))
            activation = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
            current = activation * self.max_current
            mask = current > 0.5
            noise = self._rng.normal(0, self.noise_level, size=int(np.sum(mask)))
            region.current[:n][mask] += np.maximum(0.0, current[mask] + noise)
            stimulated += int(np.sum(mask))

        return stimulated


def encode_text(text: str, region: Region, encoder: StimulusEncoder) -> int:
    values = [ord(c) / 127.0 for c in text]
    return encoder.encode(values, region)


def encode_number(value: float, region: Region, encoder: StimulusEncoder) -> int:
    return encoder.encode([np.clip(value, 0, 1)], region)
