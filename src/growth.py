"""
Growth and pruning controller operating on vectorized Region arrays.

- Neurogenesis: new neurons in active regions
- Synaptogenesis: new connections between co-active neurons
- Synaptic pruning: removal of weak/unused connections
- Apoptosis: death of chronically inactive neurons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .neuron import FiringPattern, NeuronType, PATTERN_PARAMS
from .synapse import NeurotransmitterType

if TYPE_CHECKING:
    from .region import Region


@dataclass
class GrowthStats:
    neurons_born: int = 0
    synapses_created: int = 0
    synapses_pruned: int = 0
    neurons_died: int = 0


class GrowthController:
    def __init__(
        self,
        growth_interval: int = 500,
        synapse_prune_threshold: float = 0.02,
        synapse_age_threshold: int = 3000,
        neurogenesis_threshold: float = 0.3,
        apoptosis_threshold: float = 0.001,
        apoptosis_age: int = 10000,
        max_new_synapses_per_cycle: int = 10,
        max_new_neurons_per_cycle: int = 3,
    ):
        self.growth_interval = growth_interval
        self.synapse_prune_threshold = synapse_prune_threshold
        self.synapse_age_threshold = synapse_age_threshold
        self.neurogenesis_threshold = neurogenesis_threshold
        self.apoptosis_threshold = apoptosis_threshold
        self.apoptosis_age = apoptosis_age
        self.max_new_synapses_per_cycle = max_new_synapses_per_cycle
        self.max_new_neurons_per_cycle = max_new_neurons_per_cycle

        self._step_counter = 0
        self._rng = np.random.default_rng()
        self.history: list[GrowthStats] = []

    def step(self, regions: list[Region]) -> GrowthStats | None:
        self._step_counter += 1
        if self._step_counter % self.growth_interval != 0:
            return None

        stats = GrowthStats()
        for region in regions:
            stats.synapses_pruned += self._prune_synapses(region)
            stats.neurons_died += self._apoptosis(region)
            stats.synapses_created += self._synaptogenesis(region)
            stats.neurons_born += self._neurogenesis(region)

        self.history.append(stats)
        return stats

    def _prune_synapses(self, region: Region) -> int:
        ns = region.n_synapses
        if ns == 0:
            return 0

        s = slice(0, ns)
        mask = (
            region.syn_alive[s]
            & (region.syn_age[s] > self.synapse_age_threshold)
            & (np.abs(region.syn_weight[s]) < self.synapse_prune_threshold)
            & (region.syn_recent[s] < 0.01)
        )
        count = int(np.sum(mask))
        if count > 0:
            region.syn_alive[:ns][mask] = False
        return count

    def _apoptosis(self, region: Region) -> int:
        n = region.n_neurons
        ns = region.n_synapses
        if n == 0:
            return 0

        s = slice(0, n)
        mask = (
            region.neuron_alive[s]
            & (region.neuron_age[s] > self.apoptosis_age)
            & (region.activity[s] < self.apoptosis_threshold)
            & (region.total_spikes[s] < 10)
        )
        dead = np.where(mask)[0]
        if len(dead) == 0:
            return 0

        region.neuron_alive[dead] = False

        # Kill all synapses connected to dead neurons
        if ns > 0:
            pre_dead = np.isin(region.syn_pre[:ns], dead)
            post_dead = np.isin(region.syn_post[:ns], dead)
            region.syn_alive[:ns][pre_dead | post_dead] = False

        return len(dead)

    def _synaptogenesis(self, region: Region) -> int:
        n = region.n_neurons
        ns = region.n_synapses
        if n < 2:
            return 0

        active_mask = (region.activity[:n] > 0.05) & region.neuron_alive[:n]
        active = np.where(active_mask)[0]
        if len(active) < 2:
            return 0

        # Build set of existing connections for fast lookup
        alive_syn = region.syn_alive[:ns]
        existing = set()
        if ns > 0:
            pre_alive = region.syn_pre[:ns][alive_syn]
            post_alive = region.syn_post[:ns][alive_syn]
            for p, q in zip(pre_alive, post_alive):
                existing.add((p, q))
                existing.add((q, p))

        # Find unconnected co-active pairs, scored by co-activity
        pairs = []
        act = region.activity[:n]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a_i, a_j = active[i], active[j]
                if (a_i, a_j) not in existing:
                    score = act[a_i] * act[a_j]
                    pairs.append((a_i, a_j, score))

        if not pairs:
            return 0

        pairs.sort(key=lambda x: x[2], reverse=True)
        created = 0

        for a_i, a_j, _ in pairs[:self.max_new_synapses_per_cycle]:
            pre, post = (a_i, a_j) if self._rng.random() < 0.5 else (a_j, a_i)
            region.add_one_synapse(pre, post)
            existing.add((pre, post))
            created += 1

        return created

    def _neurogenesis(self, region: Region) -> int:
        if region.mean_activity < self.neurogenesis_threshold:
            return 0
        if region.n_alive_neurons >= region.max_neurons:
            return 0

        n_new = min(
            self.max_new_neurons_per_cycle,
            region.max_neurons - region.n_alive_neurons,
        )
        born = 0

        for _ in range(n_new):
            is_exc = self._rng.random() < 0.8
            ntype = NeuronType.EXCITATORY if is_exc else NeuronType.INHIBITORY
            pattern = FiringPattern.REGULAR_SPIKING if is_exc else FiringPattern.FAST_SPIKING
            idx = region.add_neuron(ntype, pattern)
            if idx < 0:
                break

            # Connect to active neighbors
            active_mask = (
                (region.activity[:region.n_neurons] > 0.02)
                & region.neuron_alive[:region.n_neurons]
            )
            active_mask[idx] = False
            active = np.where(active_mask)[0]

            if len(active) > 0:
                n_connect = min(len(active), self._rng.integers(2, 6))
                targets = self._rng.choice(active, size=n_connect, replace=False)
                for t in targets:
                    region.add_one_synapse(t, idx)
                    if self._rng.random() < 0.3:
                        region.add_one_synapse(idx, t)

            born += 1

        return born
