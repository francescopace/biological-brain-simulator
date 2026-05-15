"""
The Brain: top-level orchestrator with vectorized simulation.

Inter-region connections are stored as Projection objects with the
same array layout as Region internal synapses. Spike delivery
uses the target region's ring buffer for delay handling.
"""

from __future__ import annotations

import json
import time as wall_time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .device import DEVICE
from .growth import GrowthController, GrowthStats
from .memory import MemorySystem
from .neuron import NeuronType
from .oscillator import OscillatorBank
from .plasticity import STDP, HomeostaticPlasticity, Metaplasticity, RewardModulatedSTDP
from .region import (
    MAX_DELAY_STEPS,
    Region,
    RegionType,
    _DEPLETION_RATE,
    _RECOVERY_RATE,
    _SYN_SPECS,
    _TRANSMISSION_DECAY,
)
from .stimulus import EncodingStrategy, StimulusEncoder
from .synapse import NT_PROPERTIES, NeurotransmitterType


# ── Inter-region projection ─────────────────────────────────────────

class Projection:
    """Synapse arrays connecting a source region to a target region."""

    def __init__(self, source_name: str, target_name: str):
        self.source_name = source_name
        self.target_name = target_name
        self.n_synapses: int = 0
        self._syn_capacity: int = 0
        self._alloc(256)

    def _alloc(self, capacity: int) -> None:
        self._syn_capacity = capacity
        for attr, dtype, default in _SYN_SPECS:
            setattr(self, attr, torch.full((capacity,), default, dtype=dtype, device=DEVICE))

    def _ensure_capacity(self, additional: int) -> None:
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


# ── Brain stats ──────────────────────────────────────────────────────

@dataclass
class BrainStats:
    time: float
    total_neurons: int
    total_synapses: int
    total_spikes: int
    regions: dict[str, dict]
    growth: GrowthStats | None = None


# ── Brain ────────────────────────────────────────────────────────────

class Brain:
    """
    Vectorized biological brain simulator.

    Usage:
        brain = Brain()
        brain.add_region("visual", RegionType.SENSORY, n_neurons=50)
        brain.add_region("cortex", RegionType.ASSOCIATION, n_neurons=100)
        brain.connect_regions("visual", "cortex", density=0.05)

        for step in range(10000):
            brain.stimulate("visual", [0.8, 0.2, 0.5])
            brain.step()
    """

    def __init__(self, dt: float = 1.0, seed: int | None = None):
        self.dt = dt
        self.time: float = 0.0
        self.step_count: int = 0

        self.regions: dict[str, Region] = {}
        self.projections: list[Projection] = []

        self.stdp = STDP()
        self.reward_stdp = RewardModulatedSTDP()
        self.homeostasis = HomeostaticPlasticity()
        self.metaplasticity = Metaplasticity()
        self.growth = GrowthController()
        self.memory = MemorySystem()
        self.encoder = StimulusEncoder(strategy=EncodingStrategy.RATE)
        self.oscillators = OscillatorBank()

        self.stats_history: list[BrainStats] = []
        self._stats_interval = 500

        self._rng = torch.Generator(device=DEVICE)
        if seed is not None:
            self._rng.manual_seed(seed)

    # ── Region management ────────────────────────────────────────────

    def add_region(
        self,
        name: str,
        region_type: RegionType,
        n_neurons: int = 50,
        connectivity: float = 0.1,
        max_neurons: int = 1000,
    ) -> Region:
        region = Region(name, region_type, max_neurons, dt=self.dt)
        region.populate(n_neurons, connectivity)
        self.regions[name] = region
        self.oscillators.add_region(name, region_type)
        return region

    def connect_regions(
        self,
        source: str,
        target: str,
        density: float = 0.05,
        bidirectional: bool = False,
        nt: NeurotransmitterType = NeurotransmitterType.GLUTAMATE,
    ) -> int:
        src = self.regions[source]
        tgt = self.regions[target]
        proj = Projection(source, target)

        src_n = src.n_neurons
        tgt_n = tgt.n_neurons

        # Only excitatory neurons project long-range
        exc_mask = src.neuron_type[:src_n] == NeuronType.EXCITATORY.value
        exc_idx = torch.where(exc_mask)[0]

        if len(exc_idx) == 0 or tgt_n == 0:
            self.projections.append(proj)
            if bidirectional:
                return self.connect_regions(target, source, density, False, nt)
            return 0

        # Vectorized connection generation
        conn = torch.rand((len(exc_idx), tgt_n), device=DEVICE, generator=self._rng) < density
        pre_local, post_idx = torch.where(conn)
        pre_idx = exc_idx[pre_local]

        if len(pre_idx) == 0:
            self.projections.append(proj)
            if bidirectional:
                return self.connect_regions(target, source, density, False, nt)
            return 0

        count = len(pre_idx)
        proj._ensure_capacity(count)
        s = proj.n_synapses
        e = s + count

        sign, mod = NT_PROPERTIES[nt]
        weights = -torch.log(torch.rand(count, device=DEVICE, generator=self._rng)) * 0.3
        if sign < 0:
            weights = -weights
        delays_ms = torch.rand(count, device=DEVICE, generator=self._rng) * 12.0 + 3.0
        delay_steps = torch.clamp(
            torch.round(delays_ms / self.dt).to(torch.int32),
            1, MAX_DELAY_STEPS - 1,
        )

        proj.syn_pre[s:e] = pre_idx.to(torch.int32)
        proj.syn_post[s:e] = post_idx.to(torch.int32)
        proj.syn_weight[s:e] = weights.to(torch.float32)
        proj.syn_delay[s:e] = delay_steps
        proj.syn_modulation[s:e] = mod
        proj.syn_resource[s:e] = 1.0
        proj.syn_facilitation[s:e] = 0.0
        proj.syn_alive[s:e] = True
        proj.syn_max_weight[s:e] = 0.0 if sign < 0 else 10.0
        proj.syn_min_weight[s:e] = -10.0 if sign < 0 else 0.0
        proj.syn_A_plus[s:e] = 0.01
        proj.syn_A_minus[s:e] = 0.012
        proj.n_synapses = e

        self.projections.append(proj)
        created = count

        if bidirectional:
            created += self.connect_regions(target, source, density, False, nt)

        return created

    # ── Stimulation ──────────────────────────────────────────────────

    def stimulate(
        self,
        region_name: str,
        values: list[float] | torch.Tensor | object,
    ) -> int:
        region = self.regions[region_name]
        return self.encoder.encode(values, region, self.time)

    # ── Reward / Punishment (per-target dopamine) ──────────────────

    def reward(self, amount: float, target: str) -> None:
        """Deliver a dopamine pulse to a specific target.

        Target syntax:
          - "region:<name>"        for a region's internal synapses
          - "proj:<src>-><dst>"    for an inter-region projection

        Use Brain.region_target(name) and Brain.projection_target(src, dst)
        to build target strings without typos.
        """
        self.reward_stdp.reward(amount, target)

    def punish(self, amount: float, target: str) -> None:
        """Deliver a dopamine dip to a specific target. See `reward()` for syntax."""
        self.reward_stdp.punish(amount, target)

    def dopamine(self, target: str) -> float:
        """Current dopamine level at `target` (baseline if unseen)."""
        return self.reward_stdp.get(target)

    @staticmethod
    def region_target(name: str) -> str:
        return f"region:{name}"

    @staticmethod
    def projection_target(source: str, target: str) -> str:
        return f"proj:{source}->{target}"

    def inject_current(
        self,
        region_name: str,
        neuron_indices: list[int] | torch.Tensor,
        current: float,
    ) -> None:
        region = self.regions[region_name]
        if not isinstance(neuron_indices, torch.Tensor):
            idx = torch.tensor(neuron_indices, dtype=torch.int64, device=DEVICE)
        else:
            idx = neuron_indices.to(torch.int64)
        valid = idx < region.n_neurons
        region.current[idx[valid]] += current

    # ── Simulation step ──────────────────────────────────────────────

    def step(self) -> dict[str, torch.Tensor]:
        self.time += self.dt
        self.step_count += 1

        # 1. Projection housekeeping (STP recovery for inter-region synapses)
        for proj in self.projections:
            ns = proj.n_synapses
            if ns == 0:
                continue
            s = slice(0, ns)
            proj.syn_resource[s] = torch.clamp(proj.syn_resource[s] + _RECOVERY_RATE, max=1.0)
            proj.syn_facilitation[s] *= 0.98
            proj.syn_age[s] += 1
            proj.syn_recent[s] *= _TRANSMISSION_DECAY

        # 2. Inject oscillatory background currents
        osc_currents = self.oscillators.step(self.dt)
        for name, osc_current in osc_currents.items():
            region = self.regions.get(name)
            if region is not None and region.n_neurons > 0:
                region.current[:region.n_neurons] += osc_current

        # 3. Step each region (reads buffer, integrates, propagates internal)
        all_fired: dict[str, torch.Tensor] = {}
        for name, region in self.regions.items():
            fired = region.step(self.time, self.step_count)
            all_fired[name] = fired

        # 4. Propagate spikes through inter-region projections
        for proj in self.projections:
            ns = proj.n_synapses
            if ns == 0:
                continue

            source = self.regions[proj.source_name]
            target = self.regions[proj.target_name]

            pre_fired = source.fired[proj.syn_pre[:ns]] & proj.syn_alive[:ns]
            active = torch.where(pre_fired)[0]

            if len(active) > 0:
                res = proj.syn_resource[active]
                fac = proj.syn_facilitation[active]
                mod = proj.syn_modulation[active]

                proj.syn_resource[active] = torch.clamp(res - _DEPLETION_RATE, min=0.0)
                proj.syn_facilitation[active] += 0.05
                proj.syn_total_tx[active] += 1
                proj.syn_recent[active] += 1.0

                # Build temporary CSR for projection (since projections don't change topology often, we could cache this too)
                # But for now, we can just use index_add_ or build a quick COO tensor
                # Since the plan specifically asked for sparse tensors:
                effective = proj.syn_weight[active] * res * (1.0 + fac) * mod
                slots = (self.step_count + proj.syn_delay[active]) % MAX_DELAY_STEPS
                posts = proj.syn_post[active]
                
                # We can use index_add_ here because building CSR for active only is not SpMV, it's just scatter.
                # To do SpMV, we need the full W matrix. Since projections don't have a built-in CSR cache yet,
                # let's just use index_add_ for projections, as the plan mainly targets Region's internal dense connections.
                # Actually, let's use index_add_ for projections to keep it simple.
                flat_indices = slots * target.spike_buffer.size(1) + posts
                target.spike_buffer.view(-1).index_add_(0, flat_indices.to(torch.int64), effective)

        # 5. Event-driven R-STDP on region-internal synapses.
        #    Each region's own `fired` array drives STDP computations; the
        #    eligibility trace then waits for dopamine targeted at this region.
        for name, region in self.regions.items():
            ns = region.n_synapses
            if ns == 0:
                continue
            n = region.n_neurons
            self.reward_stdp.apply_target(
                target=self.region_target(name),
                fired_pre=region.fired[:n],
                fired_post=region.fired[:n],
                syn_pre=region.syn_pre[:ns],
                syn_post=region.syn_post[:ns],
                pre_last_spike_arr=region.last_spike_time[:n],
                post_last_spike_arr=region.last_spike_time[:n],
                weights=region.syn_weight[:ns],
                A_plus=region.syn_A_plus[:ns],
                A_minus=region.syn_A_minus[:ns],
                alive=region.syn_alive[:ns],
                min_weight=region.syn_min_weight[:ns],
                max_weight=region.syn_max_weight[:ns],
                eligibility=region.syn_eligibility[:ns],
                current_time=self.time,
            )

        # 6. Event-driven R-STDP on inter-region projections.
        for proj in self.projections:
            ns = proj.n_synapses
            if ns == 0:
                continue
            source = self.regions[proj.source_name]
            target_region = self.regions[proj.target_name]
            self.reward_stdp.apply_target(
                target=self.projection_target(proj.source_name, proj.target_name),
                fired_pre=source.fired[:source.n_neurons],
                fired_post=target_region.fired[:target_region.n_neurons],
                syn_pre=proj.syn_pre[:ns],
                syn_post=proj.syn_post[:ns],
                pre_last_spike_arr=source.last_spike_time[:source.n_neurons],
                post_last_spike_arr=target_region.last_spike_time[:target_region.n_neurons],
                weights=proj.syn_weight[:ns],
                A_plus=proj.syn_A_plus[:ns],
                A_minus=proj.syn_A_minus[:ns],
                alive=proj.syn_alive[:ns],
                min_weight=proj.syn_min_weight[:ns],
                max_weight=proj.syn_max_weight[:ns],
                eligibility=proj.syn_eligibility[:ns],
                current_time=self.time,
            )

        # 7. Homeostatic plasticity
        self.homeostasis.apply(list(self.regions.values()), self.time)

        # 8. Metaplasticity
        self.metaplasticity.update_thresholds(list(self.regions.values()), self.time)

        # 9. Growth / pruning
        self.growth.step(list(self.regions.values()))

        # 10. Memory consolidation
        self.memory.consolidate(self.regions, self.time)

        # 11. Auto-capture memory traces
        for region in self.regions.values():
            if region.mean_activity > 0.1:
                self.memory.capture_trace(region, self.time)

        # 12. Stats
        if self.step_count % self._stats_interval == 0:
            self.stats_history.append(self._snapshot())

        return all_fired

    def run(
        self,
        steps: int,
        stimulus_fn=None,
        progress_interval: int = 1000,
        verbose: bool = True,
    ) -> None:
        t0 = wall_time.time()
        for i in range(steps):
            if stimulus_fn is not None:
                stimulus_fn(self, i)
            self.step()
            if verbose and (i + 1) % progress_interval == 0:
                elapsed = wall_time.time() - t0
                snap = self._snapshot()
                print(
                    f"  Step {self.step_count:>8d} | "
                    f"t={self.time:.0f}ms | "
                    f"neurons={snap.total_neurons} | "
                    f"synapses={snap.total_synapses} | "
                    f"spikes={snap.total_spikes} | "
                    f"memories={len(self.memory.traces)} | "
                    f"{elapsed:.1f}s"
                )
        if verbose:
            elapsed = wall_time.time() - t0
            print(f"\n  Simulation complete: {steps} steps in {elapsed:.1f}s")

    # ── Inspection ───────────────────────────────────────────────────

    def _snapshot(self) -> BrainStats:
        total_neurons = sum(r.n_alive_neurons for r in self.regions.values())
        total_synapses = (
            sum(r.n_alive_synapses for r in self.regions.values())
            + sum(int(torch.sum(p.syn_alive[:p.n_synapses]).item()) for p in self.projections)
        )
        total_spikes = sum(
            int(torch.sum(r.total_spikes[:r.n_neurons]).item())
            for r in self.regions.values()
        )
        regions_info = {}
        for name, region in self.regions.items():
            regions_info[name] = {
                "neurons": region.n_alive_neurons,
                "synapses": region.n_alive_synapses,
                "activity": float(region.mean_activity),
            }
        growth_stats = self.growth.history[-1] if self.growth.history else None
        return BrainStats(
            time=self.time,
            total_neurons=total_neurons,
            total_synapses=total_synapses,
            total_spikes=total_spikes,
            regions=regions_info,
            growth=growth_stats,
        )

    def summary(self) -> str:
        snap = self._snapshot()
        lines = [
            f"=== Brain Summary (t={self.time:.0f}ms, step={self.step_count}) ===",
            f"  Total neurons:  {snap.total_neurons}",
            f"  Total synapses: {snap.total_synapses}",
            f"  Total spikes:   {snap.total_spikes}",
            f"  Memory traces:  {len(self.memory.traces)}",
            f"  Growth cycles:  {len(self.growth.history)}",
            "",
        ]
        for name, info in snap.regions.items():
            lines.append(
                f"  [{name}] neurons={info['neurons']}, "
                f"synapses={info['synapses']}, "
                f"activity={info['activity']:.4f}"
            )
        if self.growth.history:
            recent = self.growth.history[-1]
            lines.extend([
                "",
                "  Last growth cycle:",
                f"    Neurons born:     {recent.neurons_born}",
                f"    Synapses created: {recent.synapses_created}",
                f"    Synapses pruned:  {recent.synapses_pruned}",
                f"    Neurons died:     {recent.neurons_died}",
            ])
        return "\n".join(lines)

    def save_state(self, path: str | Path) -> None:
        snap = self._snapshot()
        data = {
            "time": snap.time,
            "step_count": self.step_count,
            "total_neurons": snap.total_neurons,
            "total_synapses": snap.total_synapses,
            "total_spikes": snap.total_spikes,
            "memory_traces": len(self.memory.traces),
            "regions": snap.regions,
            "growth_history": [
                {
                    "neurons_born": g.neurons_born,
                    "synapses_created": g.synapses_created,
                    "synapses_pruned": g.synapses_pruned,
                    "neurons_died": g.neurons_died,
                }
                for g in self.growth.history
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def __repr__(self) -> str:
        snap = self._snapshot()
        return (
            f"Brain(t={self.time:.0f}ms, "
            f"regions={len(self.regions)}, "
            f"neurons={snap.total_neurons}, "
            f"synapses={snap.total_synapses})"
        )
