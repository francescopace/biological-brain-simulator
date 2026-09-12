"""CPU event lookup with a checked adjacency index and a PyTorch fallback.

Endpoint tensors are public and mutable. Checking their actual contents on each
lookup also detects edits made through NumPy views or Tensor.data, which bypass
PyTorch's version counters. This check still scans endpoints, but avoids the
larger gather/mask/nonzero sequence when activity is sparse.
"""

from __future__ import annotations

import numpy as np
import torch


class SynapseEventIndex:
    """Select live synapses in their original storage order, without updating them."""

    enabled = True
    min_synapses = 2048

    def __init__(self):
        self._endpoints = None
        self._neuron_count = None
        self._max_endpoint = -1
        self._order = None
        self._pointers = None
        self._grouped = False

    def __deepcopy__(self, memo):
        # Derived arrays are rebuilt lazily; copying a model need not copy them.
        result = type(self)()
        if "enabled" in self.__dict__:
            result.enabled = self.enabled
        memo[id(self)] = result
        return result

    @staticmethod
    def _scan(fired, endpoints, alive):
        return torch.where(fired[endpoints] & alive)[0]

    def select(self, fired: torch.Tensor, endpoints: torch.Tensor,
               alive: torch.Tensor) -> torch.Tensor:
        if (not self.enabled or endpoints.numel() < self.min_synapses
                or fired.device.type != "cpu" or endpoints.device.type != "cpu"
                or alive.device.type != "cpu"
                or fired.dtype != torch.bool or alive.dtype != torch.bool
                or endpoints.dtype not in (torch.int32, torch.int64)
                or fired.ndim != 1 or endpoints.ndim != 1 or alive.ndim != 1
                or endpoints.shape != alive.shape
                or endpoints.is_neg()):
            return self._scan(fired, endpoints, alive)

        fired_array = fired.numpy()
        neurons = np.flatnonzero(fired_array)
        # Dense activity favors the vectorized scan and does not need a cache.
        if neurons.size * 4 >= fired_array.size:
            return self._scan(fired, endpoints, alive)

        values = endpoints.numpy()
        if (self._neuron_count is None or self._neuron_count < fired_array.size
                or self._endpoints is None
                or not np.array_equal(values, self._endpoints)):
            # Preserve PyTorch's existing indexing/error behavior, including
            # valid negative indices and invalid indices on dead connections.
            if np.any(values < 0) or np.any(values >= fired_array.size):
                return self._scan(fired, endpoints, alive)
            self._endpoints = values.copy()
            self._neuron_count = fired_array.size
            self._max_endpoint = int(values.max()) if values.size else -1
            self._order = np.argsort(values, kind="stable")
            self._pointers = np.concatenate((
                [0], np.bincount(values, minlength=fired_array.size).cumsum(),
            ))
            self._grouped = bool(np.all(values[1:] >= values[:-1]))
        elif self._max_endpoint >= fired_array.size:
            return self._scan(fired, endpoints, alive)

        if not neurons.size:
            return torch.empty(0, dtype=torch.int64, device=endpoints.device)
        active = np.concatenate([
            self._order[self._pointers[n]:self._pointers[n + 1]] for n in neurons
        ])
        # Read liveness each time: pruning/reward restrictions need no rebuild.
        active = active[alive.numpy()[active]]
        if not self._grouped:
            active.sort()
        return torch.from_numpy(active)
