"""
Brain region: a population of neurons with internal connectivity.

All neuron and synapse state is stored in dense NumPy arrays for
vectorized computation. A single Region.step() call advances all
neurons and synapses in parallel.

Spike delays are handled via a ring buffer: when a presynaptic neuron
fires, the postsynaptic current is deposited into a future slot of
the buffer. Each timestep reads and clears the current slot.
"""

from __future__ import annotations

import enum

import numpy as np

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
_SYN_SPECS: list[tuple[str, type, float | int | bool]] = [
    ("syn_pre",         np.int32,   0),
    ("syn_post",        np.int32,   0),
    ("syn_weight",      np.float64, 0.0),
    ("syn_delay",       np.int32,   1),
    ("syn_modulation",  np.float64, 1.0),
    ("syn_resource",    np.float64, 1.0),
    ("syn_facilitation",np.float64, 0.0),
    ("syn_age",         np.int32,   0),
    ("syn_recent",      np.float64, 0.0),
    ("syn_total_tx",    np.int32,   0),
    ("syn_alive",       np.bool_,   False),
    ("syn_max_weight",  np.float64, 10.0),
    ("syn_min_weight",  np.float64, 0.0),
    ("syn_A_plus",      np.float64, 0.01),
    ("syn_A_minus",     np.float64, 0.012),
    ("syn_eligibility", np.float64, 0.0),
    ("syn_attenuation", np.float64, 1.0),
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
        self._rng = np.random.default_rng()

        N = max_neurons
        self.n_neurons: int = 0

        # ── Neuron state arrays (pre-allocated to max_neurons) ───────
        self.v = np.full(N, -65.0)
        self.u = np.zeros(N)
        self.a = np.zeros(N)
        self.b = np.zeros(N)
        self.c = np.full(N, -65.0)
        self.d = np.zeros(N)
        self.neuron_type = np.zeros(N, dtype=np.int8)
        self.neuron_alive = np.zeros(N, dtype=bool)
        self.activity = np.zeros(N)
        self.neuron_age = np.zeros(N, dtype=np.int32)
        self.total_spikes = np.zeros(N, dtype=np.int32)
        self.last_spike_time = np.full(N, -np.inf)
        self.current = np.zeros(N)
        self.fired = np.zeros(N, dtype=bool)

        # ── Synapse arrays (dynamically sized) ───────────────────────
        self.n_synapses: int = 0
        self._syn_capacity: int = 0
        self._alloc_synapse_arrays(1024)

        # ── Ring buffer for delayed spike delivery ───────────────────
        self.spike_buffer = np.zeros((MAX_DELAY_STEPS, N))

        # ── Morphology (optional, assigned after population) ─────────
        self.morphology_manager = None

    # ── Array management ─────────────────────────────────────────────

    def _alloc_synapse_arrays(self, capacity: int) -> None:
        """Allocate fresh synapse arrays with given capacity."""
        self._syn_capacity = capacity
        for attr, dtype, default in _SYN_SPECS:
            setattr(self, attr, np.full(capacity, default, dtype=dtype))

    def _ensure_synapse_capacity(self, additional: int) -> None:
        """Grow synapse arrays if needed (doubling strategy)."""
        needed = self.n_synapses + additional
        if needed <= self._syn_capacity:
            return
        new_cap = max(self._syn_capacity * 2, needed)
        for attr, dtype, default in _SYN_SPECS:
            old = getattr(self, attr)
            new = np.full(new_cap, default, dtype=dtype)
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

        is_exc = self._rng.random(n) < EXCITATORY_RATIO
        a_vals = np.where(is_exc, ea, ia) * (1.0 + self._rng.normal(0, 0.05, n))
        b_vals = np.where(is_exc, eb, ib) * (1.0 + self._rng.normal(0, 0.05, n))
        c_vals = np.where(is_exc, ec, ic) + self._rng.normal(0, 2.0, n)
        d_vals = np.where(is_exc, ed, id_) * (1.0 + self._rng.normal(0, 0.1, n))
        types = np.where(is_exc, NeuronType.EXCITATORY, NeuronType.INHIBITORY).astype(np.int8)

        self._add_neurons_bulk(types, a_vals, b_vals, c_vals, d_vals)
        self._create_random_connections(connectivity)

    def _add_neurons_bulk(
        self,
        types: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
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
        self.last_spike_time[s:e] = -np.inf
        self.current[s:e] = 0.0
        self.fired[s:e] = False
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
            np.array([int(ntype)], dtype=np.int8),
            np.array([a]), np.array([b]), np.array([c]), np.array([d]),
        )
        return self.n_neurons - 1

    def _create_random_connections(self, connectivity: float) -> None:
        """Create random synaptic connections within the region."""
        n = self.n_neurons
        if n < 2 or connectivity <= 0:
            return

        conn = self._rng.random((n, n)) < connectivity
        np.fill_diagonal(conn, False)
        pre_idx, post_idx = np.where(conn)
        if len(pre_idx) == 0:
            return

        pre_types = self.neuron_type[pre_idx]
        nt_types = np.where(
            pre_types == NeuronType.EXCITATORY,
            NeurotransmitterType.GLUTAMATE.value,
            NeurotransmitterType.GABA.value,
        )
        weights = self._rng.exponential(0.5, size=len(pre_idx))
        delays_ms = self._rng.uniform(1.0, 5.0, size=len(pre_idx))

        self.add_synapses(pre_idx, post_idx, weights, delays_ms, nt_types)

    # ── Synapse management ───────────────────────────────────────────

    def add_synapses(
        self,
        pre_idx: np.ndarray,
        post_idx: np.ndarray,
        weights: np.ndarray,
        delays_ms: np.ndarray,
        nt_values: np.ndarray,
    ) -> None:
        """
        Add multiple synapses at once.

        Args:
            pre_idx, post_idx: neuron indices (int arrays)
            weights: initial weights (positive; sign set from NT type)
            delays_ms: axonal delay in ms (converted to timesteps)
            nt_values: neurotransmitter string values (e.g. "glutamate")
        """
        count = len(pre_idx)
        self._ensure_synapse_capacity(count)

        s = self.n_synapses
        e = s + count

        delay_steps = np.clip(
            np.round(np.asarray(delays_ms, dtype=np.float64) / self.dt).astype(np.int32),
            1, MAX_DELAY_STEPS - 1,
        )

        signs = np.ones(count)
        mods = np.ones(count)
        for i in range(count):
            nt = NeurotransmitterType(nt_values[i]) if isinstance(nt_values[i], str) else nt_values[i]
            sign, mod = NT_PROPERTIES[nt]
            signs[i] = sign
            mods[i] = mod

        w = np.array(weights, dtype=np.float64)
        inh = signs < 0
        w[inh] = -np.abs(w[inh])

        self.syn_pre[s:e] = pre_idx
        self.syn_post[s:e] = post_idx
        self.syn_weight[s:e] = w
        self.syn_delay[s:e] = delay_steps
        self.syn_modulation[s:e] = mods
        self.syn_resource[s:e] = 1.0
        self.syn_facilitation[s:e] = 0.0
        self.syn_age[s:e] = 0
        self.syn_recent[s:e] = 0.0
        self.syn_total_tx[s:e] = 0
        self.syn_alive[s:e] = True
        self.syn_max_weight[s:e] = np.where(inh, 0.0, 10.0)
        self.syn_min_weight[s:e] = np.where(inh, -10.0, 0.0)
        self.syn_A_plus[s:e] = 0.01
        self.syn_A_minus[s:e] = 0.012

        # Compute morphological attenuation for new synapses
        if self.morphology_manager is not None:
            atten = self.morphology_manager.compute_synapse_attenuation(
                count, post_idx.astype(np.int32), self._rng,
            )
            self.syn_attenuation[s:e] = atten

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
                  if self.neuron_type[pre] == NeuronType.EXCITATORY
                  else NeurotransmitterType.GABA)
        if weight is None:
            weight = self._rng.exponential(0.3)
        if delay_ms is None:
            delay_ms = self._rng.uniform(1.0, 5.0)

        self.add_synapses(
            np.array([pre], dtype=np.int32),
            np.array([post], dtype=np.int32),
            np.array([weight]),
            np.array([delay_ms]),
            np.array([nt.value]),
        )

    # ── Simulation step ──────────────────────────────────────────────

    def step(self, time: float, step_count: int) -> np.ndarray:
        """
        Advance all neurons by one timestep (vectorized).
        Returns array of indices of neurons that fired.
        """
        n = self.n_neurons
        ns = self.n_synapses
        dt = self.dt

        if n == 0:
            return np.array([], dtype=np.int32)

        # 1. Synapse housekeeping (recovery, facilitation decay)
        if ns > 0:
            s = slice(0, ns)
            self.syn_resource[s] = np.minimum(1.0, self.syn_resource[s] + _RECOVERY_RATE)
            self.syn_facilitation[s] *= 0.98
            self.syn_age[s] += 1
            self.syn_recent[s] *= _TRANSMISSION_DECAY

        # 2. Read delayed spikes from ring buffer
        slot = step_count % MAX_DELAY_STEPS
        nslice = slice(0, n)
        self.current[nslice] += self.spike_buffer[slot, :n]
        self.spike_buffer[slot, :n] = 0.0

        # 3. Izhikevich dynamics (vectorized Euler with substeps)
        v = self.v[:n].copy()
        u = self.u[:n].copy()
        I = self.current[:n]
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
        v[fired] = c[fired]
        u[fired] += d[fired]

        self.v[:n] = v
        self.u[:n] = u

        # 6. Update spike tracking
        fired_idx = np.where(fired)[0]
        if len(fired_idx) > 0:
            self.last_spike_time[fired_idx] = time
            self.total_spikes[fired_idx] += 1
        self.fired[:n] = fired

        # 7. Activity trace (EMA)
        self.activity[:n] = self.activity[:n] * _ACTIVITY_DECAY + fired.astype(np.float64)

        # 8. Age neurons
        self.neuron_age[:n] += 1

        # 9. Clear input current
        self.current[:n] = 0.0

        # 10. Propagate spikes through internal synapses
        if len(fired_idx) > 0 and ns > 0:
            pre_fired = fired[self.syn_pre[:ns]] & self.syn_alive[:ns]
            active_syns = np.where(pre_fired)[0]

            if len(active_syns) > 0:
                res = self.syn_resource[active_syns]
                fac = self.syn_facilitation[active_syns]
                mod = self.syn_modulation[active_syns]
                atten = self.syn_attenuation[active_syns]
                effective = self.syn_weight[active_syns] * res * (1.0 + fac) * mod * atten

                self.syn_resource[active_syns] = np.maximum(0.0, res - _DEPLETION_RATE)
                self.syn_facilitation[active_syns] += 0.05
                self.syn_total_tx[active_syns] += 1
                self.syn_recent[active_syns] += 1.0

                target_slots = (step_count + self.syn_delay[active_syns]) % MAX_DELAY_STEPS
                target_neurons = self.syn_post[active_syns]
                np.add.at(self.spike_buffer, (target_slots, target_neurons), effective)

        return fired_idx

    # ── Properties ───────────────────────────────────────────────────

    @property
    def mean_activity(self) -> float:
        if self.n_neurons == 0:
            return 0.0
        alive = self.neuron_alive[:self.n_neurons]
        if not np.any(alive):
            return 0.0
        return float(np.mean(self.activity[:self.n_neurons][alive]))

    @property
    def n_alive_neurons(self) -> int:
        return int(np.sum(self.neuron_alive[:self.n_neurons]))

    @property
    def n_alive_synapses(self) -> int:
        return int(np.sum(self.syn_alive[:self.n_synapses]))

    def __repr__(self) -> str:
        return (
            f"Region({self.name}, {self.region_type.value}, "
            f"neurons={self.n_alive_neurons}, "
            f"synapses={self.n_alive_synapses})"
        )
