"""
Brain region: a population of neurons with internal connectivity.

All neuron and synapse state is stored in dense PyTorch tensors for
vectorized computation. A single Region.step() call advances all
neurons and synapses in parallel.

Spike delays are handled via a ring buffer: when a presynaptic neuron
fires, the postsynaptic current is deposited into a future slot of
the buffer. Each timestep reads and clears the current slot.
"""

from __future__ import annotations

import enum

import torch

from .device import DEVICE
from .neuron import (
    EXCITATORY_RATIO,
    PATTERN_PARAMS,
    FiringPattern,
    NeuronType,
)
from .synapse import NT_PROPERTIES, NeurotransmitterType

MAX_DELAY_STEPS = 20

# Lazy import to avoid circular dependency
_morphology_module = None
def _get_morphology_module():
    global _morphology_module
    if _morphology_module is None:
        from . import morphology as _m
        _morphology_module = _m
    return _morphology_module
_ACTIVITY_DECAY = 0.995
_RECOVERY_RATE = 0.005
_DEPLETION_RATE = 0.1
_TRANSMISSION_DECAY = 0.999

# Attribute name, dtype, default value — used for dynamic resizing
_SYN_SPECS: list[tuple[str, torch.dtype, float | int | bool]] = [
    ("syn_pre",         torch.int32,   0),
    ("syn_post",        torch.int32,   0),
    ("syn_weight",      torch.float32, 0.0),
    ("syn_delay",       torch.int32,   1),
    ("syn_modulation",  torch.float32, 1.0),
    ("syn_resource",    torch.float32, 1.0),
    ("syn_facilitation",torch.float32, 0.0),
    ("syn_age",         torch.int32,   0),
    ("syn_recent",      torch.float32, 0.0),
    ("syn_total_tx",    torch.int32,   0),
    ("syn_alive",       torch.bool,    False),
    ("syn_max_weight",  torch.float32, 10.0),
    ("syn_min_weight",  torch.float32, 0.0),
    ("syn_A_plus",      torch.float32, 0.01),
    ("syn_A_minus",     torch.float32, 0.012),
    ("syn_eligibility", torch.float32, 0.0),
    ("syn_attenuation", torch.float32, 1.0),
]


class RegionType(enum.Enum):
    SENSORY = "sensory"
    ASSOCIATION = "association"
    MEMORY = "memory"
    MOTOR = "motor"
    REWARD = "reward"


# Default firing patterns per region type: (excitatory, inhibitory)
REGION_PATTERNS: dict[RegionType, tuple[FiringPattern, FiringPattern]] = {
    RegionType.SENSORY:      (FiringPattern.REGULAR_SPIKING,        FiringPattern.FAST_SPIKING),
    RegionType.ASSOCIATION:  (FiringPattern.REGULAR_SPIKING,        FiringPattern.FAST_SPIKING),
    RegionType.MEMORY:       (FiringPattern.INTRINSICALLY_BURSTING, FiringPattern.LOW_THRESHOLD_SPIKING),
    RegionType.MOTOR:        (FiringPattern.CHATTERING,             FiringPattern.FAST_SPIKING),
    RegionType.REWARD:       (FiringPattern.REGULAR_SPIKING,        FiringPattern.FAST_SPIKING),
}


class Region:
    """A brain region with vectorized neuron and synapse arrays."""

    def __init__(
        self,
        name: str,
        region_type: RegionType,
        max_neurons: int = 1000,
        dt: float = 1.0,
    ):
        self.name = name
        self.region_type = region_type
        self.max_neurons = max_neurons
        self.dt = dt
        self._rng = torch.Generator(device=DEVICE)

        N = max_neurons
        self.n_neurons: int = 0

        # ── Neuron state arrays (pre-allocated to max_neurons) ───────
        self.v = torch.full((N,), -65.0, dtype=torch.float32, device=DEVICE)
        self.u = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.a = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.b = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.c = torch.full((N,), -65.0, dtype=torch.float32, device=DEVICE)
        self.d = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.neuron_type = torch.zeros(N, dtype=torch.int8, device=DEVICE)
        self.neuron_alive = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        self.activity = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.neuron_age = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        self.total_spikes = torch.zeros(N, dtype=torch.int32, device=DEVICE)
        self.last_spike_time = torch.full((N,), -float('inf'), dtype=torch.float32, device=DEVICE)
        self.current = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        self.fired = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        self.theta = torch.zeros(N, dtype=torch.float32, device=DEVICE)

        # ── Synapse arrays (dynamically sized) ───────────────────────
        self.n_synapses: int = 0
        self._syn_capacity: int = 0
        self._alloc_synapse_arrays(1024)

        # ── Ring buffer for delayed spike delivery ───────────────────
        self.spike_buffer = torch.zeros((MAX_DELAY_STEPS, N), dtype=torch.float32, device=DEVICE)

        # ── Morphology (optional, assigned after population) ─────────
        self.morphology_manager = None

    # ── Array management ─────────────────────────────────────────────

    def _alloc_synapse_arrays(self, capacity: int) -> None:
        """Allocate fresh synapse arrays with given capacity."""
        self._syn_capacity = capacity
        for attr, dtype, default in _SYN_SPECS:
            setattr(self, attr, torch.full((capacity,), default, dtype=dtype, device=DEVICE))

    def _ensure_synapse_capacity(self, additional: int) -> None:
        """Grow synapse arrays if needed (doubling strategy)."""
        needed = self.n_synapses + additional
        if needed <= self._syn_capacity:
            return
        new_cap = max(self._syn_capacity * 2, needed)
        for attr, dtype, default in _SYN_SPECS:
            old = getattr(self, attr)
            new = torch.full((new_cap,), default, dtype=dtype, device=DEVICE)
            new[:self.n_synapses] = old[:self.n_synapses]
            setattr(self, attr, new)
        self._syn_capacity = new_cap

    # ── Morphology ───────────────────────────────────────────────────

    def enable_morphology(self) -> None:
        """
        Assign morphologies to all neurons and compute synapse attenuation.
        Excitatory neurons get pyramidal morphology, inhibitory get interneuron.
        """
        morph_mod = _get_morphology_module()
        self.morphology_manager = morph_mod.MorphologyManager()
        self.morphology_manager.assign_default_morphologies(
            self.n_neurons, self.neuron_type,
        )
        if self.n_synapses > 0:
            atten = self.morphology_manager.compute_synapse_attenuation(
                self.n_synapses,
                self.syn_post[:self.n_synapses],
                self._rng,
            )
            self.syn_attenuation[:self.n_synapses] = atten

    # ── Population ───────────────────────────────────────────────────

    def populate(self, n_neurons: int, connectivity: float = 0.1) -> None:
        """Create initial neuron population with random internal wiring."""
        exc_pat, inh_pat = REGION_PATTERNS[self.region_type]
        ea, eb, ec, ed = PATTERN_PARAMS[exc_pat]
        ia, ib, ic, id_ = PATTERN_PARAMS[inh_pat]

        n = min(n_neurons, self.max_neurons - self.n_neurons)
        if n <= 0:
            return

        is_exc = torch.rand(n, device=DEVICE, generator=self._rng) < EXCITATORY_RATIO
        a_vals = torch.where(is_exc, ea, ia) * (1.0 + torch.randn(n, device=DEVICE, generator=self._rng) * 0.05)
        b_vals = torch.where(is_exc, eb, ib) * (1.0 + torch.randn(n, device=DEVICE, generator=self._rng) * 0.05)
        c_vals = torch.where(is_exc, ec, ic) + torch.randn(n, device=DEVICE, generator=self._rng) * 2.0
        d_vals = torch.where(is_exc, ed, id_) * (1.0 + torch.randn(n, device=DEVICE, generator=self._rng) * 0.1)
        types = torch.where(is_exc, NeuronType.EXCITATORY.value, NeuronType.INHIBITORY.value).to(torch.int8)

        self._add_neurons_bulk(types, a_vals, b_vals, c_vals, d_vals)
        self._create_random_connections(connectivity)

    def _add_neurons_bulk(
        self,
        types: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> int:
        """Add neurons from parameter arrays. Returns count actually added."""
        count = len(types)
        space = self.max_neurons - self.n_neurons
        count = min(count, space)
        if count <= 0:
            return 0

        s = self.n_neurons
        e = s + count
        self.v[s:e] = -65.0
        self.u[s:e] = b[:count] * (-65.0)
        self.a[s:e] = a[:count]
        self.b[s:e] = b[:count]
        self.c[s:e] = c[:count]
        self.d[s:e] = d[:count]
        self.neuron_type[s:e] = types[:count]
        self.neuron_alive[s:e] = True
        self.activity[s:e] = 0.0
        self.neuron_age[s:e] = 0
        self.total_spikes[s:e] = 0
        self.last_spike_time[s:e] = -float('inf')
        self.current[s:e] = 0.0
        self.fired[s:e] = False
        self.theta[s:e] = 0.0
        self.n_neurons = e
        return count

    def add_neuron(
        self,
        ntype: NeuronType = NeuronType.EXCITATORY,
        pattern: FiringPattern = FiringPattern.REGULAR_SPIKING,
    ) -> int:
        """Add a single neuron. Returns its index, or -1 if at capacity."""
        if self.n_neurons >= self.max_neurons:
            return -1
        a, b, c, d = PATTERN_PARAMS[pattern]
        self._add_neurons_bulk(
            torch.tensor([int(ntype.value)], dtype=torch.int8, device=DEVICE),
            torch.tensor([a], dtype=torch.float32, device=DEVICE),
            torch.tensor([b], dtype=torch.float32, device=DEVICE),
            torch.tensor([c], dtype=torch.float32, device=DEVICE),
            torch.tensor([d], dtype=torch.float32, device=DEVICE),
        )
        return self.n_neurons - 1

    def _create_random_connections(self, connectivity: float) -> None:
        """Create random synaptic connections within the region."""
        n = self.n_neurons
        if n < 2 or connectivity <= 0:
            return

        conn = torch.rand((n, n), device=DEVICE, generator=self._rng) < connectivity
        conn.fill_diagonal_(False)
        pre_idx, post_idx = torch.where(conn)
        if len(pre_idx) == 0:
            return

        pre_types = self.neuron_type[pre_idx]
        nt_types = torch.where(
            pre_types == NeuronType.EXCITATORY.value,
            NeurotransmitterType.GLUTAMATE.value,
            NeurotransmitterType.GABA.value,
        )
        weights = -torch.log(torch.rand(len(pre_idx), device=DEVICE, generator=self._rng)) * 0.5 # exponential
        delays_ms = torch.rand(len(pre_idx), device=DEVICE, generator=self._rng) * 4.0 + 1.0 # uniform 1-5

        self.add_synapses(pre_idx, post_idx, weights, delays_ms, nt_types)

    # ── Synapse management ───────────────────────────────────────────

    def add_synapses(
        self,
        pre_idx: torch.Tensor,
        post_idx: torch.Tensor,
        weights: torch.Tensor,
        delays_ms: torch.Tensor,
        nt_values: torch.Tensor,
    ) -> None:
        """
        Add multiple synapses at once.

        Args:
            pre_idx, post_idx: neuron indices (int tensors)
            weights: initial weights (positive; sign set from NT type)
            delays_ms: axonal delay in ms (converted to timesteps)
            nt_values: neurotransmitter string values (e.g. "glutamate")
        """
        count = len(pre_idx)
        self._ensure_synapse_capacity(count)

        s = self.n_synapses
        e = s + count

        delay_steps = torch.clamp(
            torch.round(delays_ms.to(torch.float32) / self.dt).to(torch.int32),
            1, MAX_DELAY_STEPS - 1,
        )

        signs = torch.ones(count, dtype=torch.float32, device=DEVICE)
        mods = torch.ones(count, dtype=torch.float32, device=DEVICE)
        for nt_enum, (sign_val, mod_val) in NT_PROPERTIES.items():
            nt_mask = (nt_values == nt_enum.value)
            signs[nt_mask] = sign_val
            mods[nt_mask] = mod_val

        w = weights.to(torch.float32).clone()
        inh = signs < 0
        w[inh] = -torch.abs(w[inh])

        self.syn_pre[s:e] = pre_idx.to(torch.int32)
        self.syn_post[s:e] = post_idx.to(torch.int32)
        self.syn_weight[s:e] = w
        self.syn_delay[s:e] = delay_steps
        self.syn_modulation[s:e] = mods
        self.syn_resource[s:e] = 1.0
        self.syn_facilitation[s:e] = 0.0
        self.syn_age[s:e] = 0
        self.syn_recent[s:e] = 0.0
        self.syn_total_tx[s:e] = 0
        self.syn_alive[s:e] = True
        self.syn_max_weight[s:e] = torch.where(inh, 0.0, 10.0)
        self.syn_min_weight[s:e] = torch.where(inh, -10.0, 0.0)
        self.syn_A_plus[s:e] = 0.01
        self.syn_A_minus[s:e] = 0.012

        # Compute morphological attenuation for new synapses
        if self.morphology_manager is not None:
            # morphology_manager needs to be updated to support torch
            pass

        self.n_synapses = e

    def add_one_synapse(
        self,
        pre: int,
        post: int,
        weight: float | None = None,
        delay_ms: float | None = None,
        nt: NeurotransmitterType | None = None,
    ) -> None:
        """Convenience: add a single synapse."""
        if nt is None:
            nt = (NeurotransmitterType.GLUTAMATE
                  if self.neuron_type[pre] == NeuronType.EXCITATORY.value
                  else NeurotransmitterType.GABA)
        if weight is None:
            weight = -torch.log(torch.rand(1, device=DEVICE, generator=self._rng)).item() * 0.3
        if delay_ms is None:
            delay_ms = (torch.rand(1, device=DEVICE, generator=self._rng).item() * 4.0) + 1.0

        self.add_synapses(
            torch.tensor([pre], dtype=torch.int32, device=DEVICE),
            torch.tensor([post], dtype=torch.int32, device=DEVICE),
            torch.tensor([weight], dtype=torch.float32, device=DEVICE),
            torch.tensor([delay_ms], dtype=torch.float32, device=DEVICE),
            torch.tensor([nt.value], dtype=torch.int32, device=DEVICE),
        )

    def add_lateral_inhibition(
        self,
        weight: float,
        delay_ms: float = 1.0,
        nt: NeurotransmitterType = NeurotransmitterType.GABA,
        neuron_indices: torch.Tensor | list[int] | None = None,
    ) -> int:
        """Add all-to-all inhibitory wiring across a neuron subset."""
        if neuron_indices is None:
            indices = torch.arange(self.n_neurons, dtype=torch.int64, device=DEVICE)
        elif isinstance(neuron_indices, torch.Tensor):
            indices = neuron_indices.to(dtype=torch.int64, device=DEVICE)
        else:
            indices = torch.as_tensor(neuron_indices, dtype=torch.int64, device=DEVICE)

        if len(indices) < 2:
            return 0

        pre_grid, post_grid = torch.meshgrid(indices, indices, indexing="ij")
        mask = pre_grid != post_grid
        count = int(torch.sum(mask).item())
        if count == 0:
            return 0

        self.add_synapses(
            pre_grid[mask].to(torch.int32),
            post_grid[mask].to(torch.int32),
            torch.full((count,), weight, dtype=torch.float32, device=DEVICE),
            torch.full((count,), delay_ms, dtype=torch.float32, device=DEVICE),
            torch.full((count,), nt.value, dtype=torch.int32, device=DEVICE),
        )
        return count

    # ── Simulation step ──────────────────────────────────────────────

    def step(self, time: float, step_count: int) -> torch.Tensor:
        """
        Advance all neurons by one timestep (vectorized).
        Returns array of indices of neurons that fired.
        """
        n = self.n_neurons
        ns = self.n_synapses
        dt = self.dt

        if n == 0:
            return torch.tensor([], dtype=torch.int32, device=DEVICE)

        # 1. Synapse housekeeping (recovery, facilitation decay)
        if ns > 0:
            s = slice(0, ns)
            self.syn_resource[s] = torch.clamp(self.syn_resource[s] + _RECOVERY_RATE, max=1.0)
            self.syn_facilitation[s] *= 0.98
            self.syn_age[s] += 1
            self.syn_recent[s] *= _TRANSMISSION_DECAY

        # 2. Read delayed spikes from ring buffer
        slot = step_count % MAX_DELAY_STEPS
        self.current[:n] += self.spike_buffer[slot, :n]
        self.spike_buffer[slot, :n] = 0.0

        # 3. Izhikevich dynamics (vectorized Euler with substeps)
        v = self.v[:n].clone()
        u = self.u[:n].clone()
        I = self.current[:n] - self.theta[:n]
        a = self.a[:n]
        b = self.b[:n]

        substeps = max(1, int(dt / 0.5))
        sub_dt = dt / substeps
        for _ in range(substeps):
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * sub_dt
            du = a * (b * v - u) * sub_dt
            v += dv
            u += du

        # 4. Spike detection
        fired = (v >= 30.0) & self.neuron_alive[:n]

        # 5. Reset fired neurons
        c = self.c[:n]
        d = self.d[:n]
        v = torch.where(fired, c, v)
        u = torch.where(fired, u + d, u)

        self.v[:n] = v
        self.u[:n] = u

        # 6. Update spike tracking
        fired_idx = torch.where(fired)[0]
        if len(fired_idx) > 0:
            self.last_spike_time[fired_idx] = time
            self.total_spikes[fired_idx] += 1
        self.fired[:n] = fired

        # 7. Activity trace (EMA)
        self.activity[:n] = self.activity[:n] * _ACTIVITY_DECAY + fired.to(torch.float32)

        # 8. Age neurons
        self.neuron_age[:n] += 1

        # 9. Clear input current
        self.current[:n] = 0.0

        # 10. Propagate spikes through internal synapses
        if len(fired_idx) > 0 and ns > 0:
            pre_fired = fired[self.syn_pre[:ns]] & self.syn_alive[:ns]
            active_syns = torch.where(pre_fired)[0]

            if len(active_syns) > 0:
                res = self.syn_resource[active_syns]
                fac = self.syn_facilitation[active_syns]
                mod = self.syn_modulation[active_syns]
                atten = self.syn_attenuation[active_syns]
                effective = self.syn_weight[active_syns] * res * (1.0 + fac) * mod * atten

                self.syn_resource[active_syns] = torch.clamp(res - _DEPLETION_RATE, min=0.0)
                self.syn_facilitation[active_syns] += 0.05
                self.syn_total_tx[active_syns] += 1
                self.syn_recent[active_syns] += 1.0

                target_slots = (step_count + self.syn_delay[active_syns]) % MAX_DELAY_STEPS
                target_neurons = self.syn_post[active_syns].to(torch.int64)
                flat_indices = target_slots.to(torch.int64) * n + target_neurons
                self.spike_buffer.view(-1).index_add_(0, flat_indices, effective)

        return fired_idx

    # ── Properties ───────────────────────────────────────────────────

    @property
    def mean_activity(self) -> float:
        if self.n_neurons == 0:
            return 0.0
        alive = self.neuron_alive[:self.n_neurons]
        if not torch.any(alive):
            return 0.0
        return float(torch.mean(self.activity[:self.n_neurons][alive]).item())

    @property
    def n_alive_neurons(self) -> int:
        return int(torch.sum(self.neuron_alive[:self.n_neurons]).item())

    @property
    def n_alive_synapses(self) -> int:
        return int(torch.sum(self.syn_alive[:self.n_synapses]).item())

    def __repr__(self) -> str:
        return (
            f"Region({self.name}, {self.region_type.value}, "
            f"neurons={self.n_alive_neurons}, "
            f"synapses={self.n_alive_synapses})"
        )
