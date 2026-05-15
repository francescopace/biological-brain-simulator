"""
Stimulus encoder: converts external inputs into currents injected
directly into a Region's current array (vectorized).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .device import DEVICE

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
    _rng: torch.Generator = None

    def __post_init__(self):
        if self._rng is None:
            self._rng = torch.Generator(device=DEVICE)

    def encode(
        self,
        values: torch.Tensor | list | object,
        region: Region,
        current_time: float = 0.0,
    ) -> int:
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
        elif values.device != DEVICE or values.dtype != torch.float32:
            values = values.to(dtype=torch.float32, device=DEVICE)

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

    def _rate_encode(self, values: torch.Tensor, region: Region) -> int:
        n = region.n_neurons
        n_inputs = len(values)
        neurons_per_input = max(1, n // max(n_inputs, 1))

        if neurons_per_input == 1 and n_inputs <= n:
            # Fast path: 1:1 mapping (common case, e.g. MNIST 784 pixels -> 784 neurons)
            k = min(n_inputs, n)
            currents = values[:k] * self.max_current
            noise = torch.randn(k, device=DEVICE, generator=self._rng) * self.noise_level
            region.current[:k] += torch.clamp(currents + noise, min=0.0)
            return k

        # General path: each input maps to neurons_per_input neurons
        total = min(n_inputs * neurons_per_input, n)
        expanded = values[:n_inputs].repeat_interleave(neurons_per_input)[:total] * self.max_current
        noise = torch.randn(total, device=DEVICE, generator=self._rng) * self.noise_level
        region.current[:total] += torch.clamp(expanded + noise, min=0.0)
        return total

    def _temporal_encode(self, values: torch.Tensor, region: Region) -> int:
        n = region.n_neurons
        n_inputs = len(values)
        neurons_per_input = max(1, n // max(n_inputs, 1))

        if neurons_per_input == 1 and n_inputs <= n:
            k = min(n_inputs, n)
            currents = torch.clamp(values[:k], 0.0, 1.0) * self.max_current
            noise = torch.randn(k, device=DEVICE, generator=self._rng) * (self.noise_level * 0.5)
            region.current[:k] += torch.clamp(currents + noise, min=0.0)
            return k

        total = min(n_inputs * neurons_per_input, n)
        clamped = torch.clamp(values[:n_inputs], 0.0, 1.0)
        expanded = clamped.repeat_interleave(neurons_per_input)[:total] * self.max_current
        noise = torch.randn(total, device=DEVICE, generator=self._rng) * (self.noise_level * 0.5)
        region.current[:total] += torch.clamp(expanded + noise, min=0.0)
        return total

    def _population_encode(self, values: torch.Tensor, region: Region) -> int:
        n = region.n_neurons
        stimulated = 0
        sigma = max(1, n // 10)

        indices = torch.arange(n, dtype=torch.float32, device=DEVICE)
        for val in values:
            center = torch.clamp(val, 0.0, 1.0) * (n - 1)
            activation = torch.exp(-0.5 * ((indices - center) / sigma) ** 2)
            current = activation * self.max_current
            mask = current > 0.5
            count = int(torch.sum(mask).item())
            noise = torch.randn(count, device=DEVICE, generator=self._rng) * self.noise_level
            region.current[:n][mask] += torch.clamp(current[mask] + noise, min=0.0)
            stimulated += count

        return stimulated


def encode_text(text: str, region: Region, encoder: StimulusEncoder) -> int:
    values = [ord(c) / 127.0 for c in text]
    return encoder.encode(values, region)


def encode_number(value: float, region: Region, encoder: StimulusEncoder) -> int:
    return encoder.encode([max(0.0, min(value, 1.0))], region)
