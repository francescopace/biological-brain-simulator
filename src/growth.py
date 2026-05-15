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

import torch

from .device import DEVICE
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
        self._rng = torch.Generator(device=DEVICE)
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
            & (torch.abs(region.syn_weight[s]) < self.synapse_prune_threshold)
            & (region.syn_recent[s] < 0.01)
        )
        count = int(torch.sum(mask).item())
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
        dead = torch.where(mask)[0]
        if len(dead) == 0:
            return 0

        region.neuron_alive[dead] = False

        # Kill all synapses connected to dead neurons
        if ns > 0:
            pre_dead = torch.isin(region.syn_pre[:ns], dead)
            post_dead = torch.isin(region.syn_post[:ns], dead)
            mask = pre_dead | post_dead
            if torch.any(mask):
                region.syn_alive[:ns][mask] = False

        return len(dead)

    def _synaptogenesis(self, region: Region) -> int:
        n = region.n_neurons
        ns = region.n_synapses
        if n < 2:
            return 0

        active_mask = (region.activity[:n] > 0.05) & region.neuron_alive[:n]
        active = torch.where(active_mask)[0]
        if len(active) < 2:
            return 0

        # Build dense adjacency matrix for existing connections
        existing = torch.zeros((n, n), dtype=torch.bool, device=DEVICE)
        if ns > 0:
            alive_syn = region.syn_alive[:ns]
            pre_alive = region.syn_pre[:ns][alive_syn].to(torch.int64)
            post_alive = region.syn_post[:ns][alive_syn].to(torch.int64)
            existing[pre_alive, post_alive] = True
            existing[post_alive, pre_alive] = True

        # Vectorized outer product of activity
        act = region.activity[:n]
        scores = torch.outer(act, act)
        
        # Mask out non-active, existing, and self connections
        scores[~active_mask, :] = 0.0
        scores[:, ~active_mask] = 0.0
        scores[existing] = 0.0
        scores.fill_diagonal_(0.0)

        # We only need upper triangle to avoid duplicate pairs
        scores = torch.triu(scores)

        # Find top pairs
        flat_scores = scores.view(-1)
        k = min(self.max_new_synapses_per_cycle, int((flat_scores > 0).sum().item()))
        if k == 0:
            return 0

        top_scores, top_idx = torch.topk(flat_scores, k)
        
        created = 0
        for idx in top_idx:
            if flat_scores[idx] <= 0:
                break
            a_i = (idx // n).item()
            a_j = (idx % n).item()
            
            pre, post = (a_i, a_j) if torch.rand(1, device=DEVICE, generator=self._rng).item() < 0.5 else (a_j, a_i)
            region.add_one_synapse(pre, post)
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
            is_exc = torch.rand(1, device=DEVICE, generator=self._rng).item() < 0.8
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
            active = torch.where(active_mask)[0]

            if len(active) > 0:
                n_connect = min(len(active), torch.randint(2, 6, (1,), device=DEVICE, generator=self._rng).item())
                perm = torch.randperm(len(active), device=DEVICE, generator=self._rng)
                targets = active[perm[:n_connect]]
                for t in targets:
                    region.add_one_synapse(t.item(), idx)
                    if torch.rand(1, device=DEVICE, generator=self._rng).item() < 0.3:
                        region.add_one_synapse(idx, t.item())

            born += 1

        return born
