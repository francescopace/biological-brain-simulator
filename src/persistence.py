"""
Full brain serialization and deserialization.

Saves the complete brain state (neuron arrays, synapse arrays, projections,
subsystem counters, oscillator phases, memory traces, RNG state) to disk
so the brain can be stopped and restarted without losing any state.

Format:
    brain_save/
      meta.json               # version, time, step_count, config
      regions/
        <name>.pt             # all neuron + synapse arrays (torch)
      projections/
        <src>_to_<tgt>.pt    # inter-region synapse arrays (torch)
      memory.json             # memory traces
      subsystems.json         # plasticity, growth, oscillator state
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .brain import Brain, Projection
from .device import DEVICE
from .growth import GrowthStats
from .memory import MemoryTrace
from .oscillator import FrequencyBand
from .region import MAX_DELAY_STEPS, Region, RegionType, _SYN_SPECS
from .stimulus import EncodingStrategy

_VERSION = 4


def save_brain(brain: Brain, path: str | Path) -> None:
    """Save the complete brain state to a directory."""
    root = Path(path)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "regions").mkdir()
    (root / "projections").mkdir()

    # ── Meta ─────────────────────────────────────────────────────
    meta = {
        "version": _VERSION,
        "time": brain.time,
        "step_count": brain.step_count,
        "dt": brain.dt,
        "stats_interval": brain._stats_interval,
        "region_names": list(brain.regions.keys()),
    }
    _write_json(root / "meta.json", meta)

    # Save brain RNG
    torch.save(brain._rng.get_state(), root / "brain_rng.pt")

    # ── Regions ──────────────────────────────────────────────────
    for name, region in brain.regions.items():
        _save_region(root / "regions" / f"{name}.pt", region)

    # ── Projections ──────────────────────────────────────────────
    proj_info = []
    for i, proj in enumerate(brain.projections):
        fname = f"{proj.source_name}_to_{proj.target_name}_{i}.pt"
        _save_projection(root / "projections" / fname, proj)
        proj_info.append({
            "source": proj.source_name,
            "target": proj.target_name,
            "file": fname,
            "n_synapses": proj.n_synapses,
        })
    _write_json(root / "projections.json", proj_info)

    # ── Memory ───────────────────────────────────────────────────
    memory_data = {
        "trace_capacity": brain.memory.trace_capacity,
        "consolidation_interval": brain.memory.consolidation_interval,
        "replay_strength": brain.memory.replay_strength,
        "trace_threshold": brain.memory.trace_threshold,
        "step_counter": brain.memory._step_counter,
        "traces": [
            {
                "neuron_indices": t.neuron_indices.tolist(),
                "activity_snapshot": t.activity_snapshot.tolist(),
                "region_name": t.region_name,
                "strength": t.strength,
                "replay_count": t.replay_count,
                "creation_time": t.creation_time,
            }
            for t in brain.memory.traces
        ],
    }
    _write_json(root / "memory.json", memory_data)
    torch.save(brain.memory._rng.get_state(), root / "memory_rng.pt")

    # ── Subsystems ───────────────────────────────────────────────
    subsystems = {
        "stdp": {
            "learning_rate": brain.stdp.learning_rate,
            "stdp_window": brain.stdp.stdp_window,
            "tau_plus": brain.stdp.tau_plus,
            "tau_minus": brain.stdp.tau_minus,
        },
        "reward_stdp": {
            "tau_eligibility": brain.reward_stdp.tau_eligibility,
            "dopamine_decay": brain.reward_stdp.dopamine_decay,
            "baseline_dopamine": brain.reward_stdp.baseline_dopamine,
            "dopamine": dict(brain.reward_stdp.dopamine),
        },
        "homeostasis": {
            "target_rate": brain.homeostasis.target_rate,
            "scaling_rate": brain.homeostasis.scaling_rate,
            "check_interval": brain.homeostasis.check_interval,
            "step_counter": brain.homeostasis._step_counter,
            "theta_plus": brain.homeostasis.theta_plus,
            "theta_leak": brain.homeostasis.theta_leak,
        },
        "metaplasticity": {
            "adaptation_rate": brain.metaplasticity.adaptation_rate,
        },
        "growth": {
            "growth_interval": brain.growth.growth_interval,
            "synapse_prune_threshold": brain.growth.synapse_prune_threshold,
            "synapse_age_threshold": brain.growth.synapse_age_threshold,
            "neurogenesis_threshold": brain.growth.neurogenesis_threshold,
            "apoptosis_threshold": brain.growth.apoptosis_threshold,
            "apoptosis_age": brain.growth.apoptosis_age,
            "max_new_synapses_per_cycle": brain.growth.max_new_synapses_per_cycle,
            "max_new_neurons_per_cycle": brain.growth.max_new_neurons_per_cycle,
            "step_counter": brain.growth._step_counter,
            "history": [
                {
                    "neurons_born": g.neurons_born,
                    "synapses_created": g.synapses_created,
                    "synapses_pruned": g.synapses_pruned,
                    "neurons_died": g.neurons_died,
                }
                for g in brain.growth.history
            ],
        },
        "oscillators": _serialize_oscillators(brain.oscillators),
        "encoder": {
            "strategy": brain.encoder.strategy.value,
            "max_current": brain.encoder.max_current,
            "noise_level": brain.encoder.noise_level,
        },
    }
    _write_json(root / "subsystems.json", subsystems)
    torch.save(brain.growth._rng.get_state(), root / "growth_rng.pt")
    torch.save(brain.encoder._rng.get_state(), root / "encoder_rng.pt")


def load_brain(path: str | Path) -> Brain:
    """Restore a brain from a saved directory."""
    root = Path(path)

    meta = _read_json(root / "meta.json")
    assert meta["version"] == _VERSION, f"Unsupported save version: {meta['version']}"

    brain = Brain(dt=meta["dt"])
    brain.time = meta["time"]
    brain.step_count = meta["step_count"]
    brain._stats_interval = meta["stats_interval"]

    rng_state_path = root / "brain_rng.pt"
    if rng_state_path.exists():
        brain._rng.set_state(torch.load(rng_state_path, weights_only=True))

    # ── Regions ──────────────────────────────────────────────────
    for name in meta["region_names"]:
        region = _load_region(root / "regions" / f"{name}.pt", brain.dt)
        brain.regions[name] = region
        brain.oscillators.add_region(name, region.region_type)

    # ── Projections ──────────────────────────────────────────────
    proj_info = _read_json(root / "projections.json")
    brain.projections = []
    for pi in proj_info:
        proj = _load_projection(
            root / "projections" / pi["file"],
            pi["source"], pi["target"],
        )
        brain.projections.append(proj)

    # ── Memory ───────────────────────────────────────────────────
    mem_data = _read_json(root / "memory.json")
    brain.memory.trace_capacity = mem_data["trace_capacity"]
    brain.memory.consolidation_interval = mem_data["consolidation_interval"]
    brain.memory.replay_strength = mem_data["replay_strength"]
    brain.memory.trace_threshold = mem_data["trace_threshold"]
    brain.memory._step_counter = mem_data["step_counter"]
    mem_rng_path = root / "memory_rng.pt"
    if mem_rng_path.exists():
        brain.memory._rng.set_state(torch.load(mem_rng_path, weights_only=True))
    brain.memory.traces = [
        MemoryTrace(
            neuron_indices=torch.tensor(t["neuron_indices"], dtype=torch.int64, device=DEVICE),
            activity_snapshot=torch.tensor(t["activity_snapshot"], dtype=torch.float32, device=DEVICE),
            region_name=t["region_name"],
            strength=t["strength"],
            replay_count=t["replay_count"],
            creation_time=t["creation_time"],
        )
        for t in mem_data["traces"]
    ]

    # ── Subsystems ───────────────────────────────────────────────
    sub = _read_json(root / "subsystems.json")

    brain.stdp.learning_rate = sub["stdp"]["learning_rate"]
    brain.stdp.stdp_window = sub["stdp"]["stdp_window"]
    brain.stdp.tau_plus = sub["stdp"]["tau_plus"]
    brain.stdp.tau_minus = sub["stdp"]["tau_minus"]

    brain.reward_stdp.tau_eligibility = sub["reward_stdp"]["tau_eligibility"]
    brain.reward_stdp.dopamine_decay = sub["reward_stdp"]["dopamine_decay"]
    brain.reward_stdp.baseline_dopamine = sub["reward_stdp"]["baseline_dopamine"]
    brain.reward_stdp.dopamine = dict(sub["reward_stdp"]["dopamine"])

    brain.homeostasis.target_rate = sub["homeostasis"]["target_rate"]
    brain.homeostasis.scaling_rate = sub["homeostasis"]["scaling_rate"]
    brain.homeostasis.check_interval = sub["homeostasis"]["check_interval"]
    brain.homeostasis._step_counter = sub["homeostasis"]["step_counter"]
    brain.homeostasis.theta_plus = sub["homeostasis"].get("theta_plus", 0.10)
    brain.homeostasis.theta_leak = sub["homeostasis"].get("theta_leak", 0.005)

    brain.metaplasticity.adaptation_rate = sub["metaplasticity"]["adaptation_rate"]

    g = sub["growth"]
    brain.growth.growth_interval = g["growth_interval"]
    brain.growth.synapse_prune_threshold = g["synapse_prune_threshold"]
    brain.growth.synapse_age_threshold = g["synapse_age_threshold"]
    brain.growth.neurogenesis_threshold = g["neurogenesis_threshold"]
    brain.growth.apoptosis_threshold = g["apoptosis_threshold"]
    brain.growth.apoptosis_age = g["apoptosis_age"]
    brain.growth.max_new_synapses_per_cycle = g["max_new_synapses_per_cycle"]
    brain.growth.max_new_neurons_per_cycle = g["max_new_neurons_per_cycle"]
    brain.growth._step_counter = g["step_counter"]
    growth_rng_path = root / "growth_rng.pt"
    if growth_rng_path.exists():
        brain.growth._rng.set_state(torch.load(growth_rng_path, weights_only=True))
    brain.growth.history = [
        GrowthStats(**h) for h in g["history"]
    ]

    _deserialize_oscillators(brain.oscillators, sub["oscillators"])

    brain.encoder.strategy = EncodingStrategy(sub["encoder"]["strategy"])
    brain.encoder.max_current = sub["encoder"]["max_current"]
    brain.encoder.noise_level = sub["encoder"]["noise_level"]
    encoder_rng_path = root / "encoder_rng.pt"
    if encoder_rng_path.exists():
        brain.encoder._rng.set_state(torch.load(encoder_rng_path, weights_only=True))

    return brain


# ── Internal helpers ─────────────────────────────────────────────────

def _save_region(path: Path, region: Region) -> None:
    """Save region neuron + synapse arrays."""
    n = region.n_neurons
    ns = region.n_synapses

    state = {
        "region_type": region.region_type.value,
        "max_neurons": region.max_neurons,
        "n_neurons": n,
        "n_synapses": ns,
        # Neuron arrays
        "v": region.v[:n],
        "u": region.u[:n],
        "a": region.a[:n],
        "b": region.b[:n],
        "c": region.c[:n],
        "d": region.d[:n],
        "neuron_type": region.neuron_type[:n],
        "neuron_alive": region.neuron_alive[:n],
        "activity": region.activity[:n],
        "neuron_age": region.neuron_age[:n],
        "total_spikes": region.total_spikes[:n],
        "last_spike_time": region.last_spike_time[:n],
        "current": region.current[:n],
        "fired": region.fired[:n],
        "theta": region.theta[:n],
        # Spike buffer
        "spike_buffer": region.spike_buffer[:, :n],
        # RNG
        "rng_state": region._rng.get_state(),
    }

    # Synapse arrays
    for attr, dtype, default in _SYN_SPECS:
        state[attr] = getattr(region, attr)[:ns]

    torch.save(state, path)


def _load_region(path: Path, dt: float) -> Region:
    """Load region from torch checkpoint."""
    data = torch.load(path, map_location=DEVICE, weights_only=True)

    region_type = RegionType(data["region_type"])
    max_neurons = data["max_neurons"]
    n = data["n_neurons"]
    ns = data["n_synapses"]

    region = Region(
        name=path.stem,
        region_type=region_type,
        max_neurons=max_neurons,
        dt=dt,
    )

    region.n_neurons = n
    region.v[:n] = data["v"]
    region.u[:n] = data["u"]
    region.a[:n] = data["a"]
    region.b[:n] = data["b"]
    region.c[:n] = data["c"]
    region.d[:n] = data["d"]
    region.neuron_type[:n] = data["neuron_type"]
    region.neuron_alive[:n] = data["neuron_alive"]
    region.activity[:n] = data["activity"]
    region.neuron_age[:n] = data["neuron_age"]
    region.total_spikes[:n] = data["total_spikes"]
    region.last_spike_time[:n] = data["last_spike_time"]
    region.current[:n] = data["current"]
    region.fired[:n] = data["fired"]
    if "theta" in data:
        region.theta[:n] = data["theta"]
    region.spike_buffer[:, :n] = data["spike_buffer"]

    # Synapse arrays
    region._ensure_synapse_capacity(ns)
    region.n_synapses = ns
    for attr, dtype, default in _SYN_SPECS:
        if attr in data:
            getattr(region, attr)[:ns] = data[attr]

    # RNG state
    if "rng_state" in data:
        region._rng.set_state(data["rng_state"])

    return region


def _save_projection(path: Path, proj: Projection) -> None:
    ns = proj.n_synapses
    state = {"n_synapses": ns}
    for attr, dtype, default in _SYN_SPECS:
        state[attr] = getattr(proj, attr)[:ns]
    torch.save(state, path)


def _load_projection(path: Path, source: str, target: str) -> Projection:
    data = torch.load(path, map_location=DEVICE, weights_only=True)
    ns = data["n_synapses"]

    proj = Projection(source, target)
    proj._ensure_capacity(ns)
    proj.n_synapses = ns
    for attr, dtype, default in _SYN_SPECS:
        if attr in data:
            getattr(proj, attr)[:ns] = data[attr]
    return proj


def _serialize_oscillators(bank) -> dict:
    result = {
        "base_amplitude": bank.base_amplitude,
        "gamma_theta_coupling": bank.gamma_theta_coupling,
        "regions": {},
    }
    for name, oscs in bank.oscillators.items():
        result["regions"][name] = [
            {
                "frequency": o.frequency,
                "amplitude": o.amplitude,
                "phase": o.phase,
                "band": o.band.value,
                "coupling_strength": o.coupling_strength,
            }
            for o in oscs
        ]
    return result


def _deserialize_oscillators(bank, data: dict) -> None:
    bank.base_amplitude = data["base_amplitude"]
    bank.gamma_theta_coupling = data["gamma_theta_coupling"]

    for name, osc_list in data["regions"].items():
        if name not in bank.oscillators:
            continue
        existing = bank.oscillators[name]
        for i, osc_data in enumerate(osc_list):
            if i < len(existing):
                existing[i].frequency = osc_data["frequency"]
                existing[i].amplitude = osc_data["amplitude"]
                existing[i].phase = osc_data["phase"]
                existing[i].coupling_strength = osc_data["coupling_strength"]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _json_default(obj):
    """Handle torch types in JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
