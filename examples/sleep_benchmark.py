"""
Sleep / replay benefit benchmark.

Tests whether offline consolidation (replay of memory traces during
a silent phase) improves retention compared to no-replay.

Protocol:
  1. Train SNN on MNIST (300/class) for 1 epoch
  2. Evaluate → accuracy_before_sleep
  3. From identical copies, run all four replay on/off × STDP on/off arms
  4. Keep scaling, adaptive theta and topology frozen during every sleep arm
  5. Re-evaluate with the unchanged pre-sleep decoder and report weight changes

STDP is manual, feedforward-only and explicit in each arm. Replay uses the
stored trace bank without capturing new traces during sleep. Single-seed
accuracy differences are descriptive; this experiment is not a significance test.
"""

from __future__ import annotations

import sys
import time
import copy
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.mnist_benchmark import (
    CLASSES,
    SEED,
    THETA_LEAK,
    THETA_PLUS,
    build_readout,
    build_readout_subset,
    compute_norm_target,
    load_reduced_mnist,
    present_sample,
    reset_brain_state,
    normalize_feedforward_weights,
    evaluate,
    apply_feedforward_stdp,
)
from src.brain import Brain
from src.device import DEVICE
from src.neuron import NeuronType, FiringPattern
from src.region import RegionType
from src.persistence import save_brain

TRAIN_PER_CLASS = 300
TEST_PER_CLASS = 50
READOUT_PER_CLASS = 30
PRESENT_STEPS = 200
REST_STEPS = 50
SLEEP_STEPS = 5000
CONSOLIDATION_INTERVAL = 100


def build_brain_with_memory(seed: int = SEED) -> Brain:
    """Build brain like mnist_benchmark but with memory system enabled."""
    from examples.mnist_benchmark import (
        N_INPUT, N_CORTEX_EXC, N_CORTEX_INH,
        INPUT_TO_CORTEX_DENSITY, INPUT_WEIGHT_BOOST,
        STDP_A_PLUS, STDP_A_MINUS, STDP_SCALE,
        ENCODER_MAX_CURRENT, ENCODER_NOISE,
        wire_cortex_microcircuit,
    )

    brain = Brain(dt=1.0, seed=seed)

    input_region = brain.add_region(
        "input", RegionType.SENSORY, n_neurons=0,
        connectivity=0.0, max_neurons=N_INPUT,
    )
    for _ in range(N_INPUT):
        input_region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)

    cortex = brain.add_region(
        "cortex", RegionType.ASSOCIATION, n_neurons=0,
        connectivity=0.0, max_neurons=N_CORTEX_EXC + N_CORTEX_INH,
    )
    for _ in range(N_CORTEX_EXC):
        cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    for _ in range(N_CORTEX_INH):
        cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
    wire_cortex_microcircuit(cortex)

    brain.connect_regions("input", "cortex", density=INPUT_TO_CORTEX_DENSITY)
    brain.freeze_plasticity()
    proj = brain.enable_projection_plasticity(
        "input", "cortex",
        A_plus=STDP_A_PLUS * STDP_SCALE,
        A_minus=STDP_A_MINUS * STDP_SCALE,
    )
    ns = proj.n_synapses
    proj.syn_weight[:ns] *= INPUT_WEIGHT_BOOST
    proj.syn_weight[:ns] = torch.clamp(
        proj.syn_weight[:ns], proj.syn_min_weight[:ns], proj.syn_max_weight[:ns],
    )

    cortex_types = cortex.neuron_type[:cortex.n_neurons]
    exc_post = cortex_types[proj.syn_post[:ns].to(torch.int64)] == NeuronType.EXCITATORY.value
    proj.syn_A_plus[:ns][~exc_post] = 0.0
    proj.syn_A_minus[:ns][~exc_post] = 0.0

    # Keep R-STDP disabled but enable memory system
    brain.disable_reward_modulated_plasticity()
    brain.reset_traces()
    brain.freeze_structural_plasticity()
    # Memory system stays active (not monkey-patched)
    brain.memory.consolidation_interval = CONSOLIDATION_INTERVAL
    brain.memory.replay_strength = 1.0
    brain.memory.trace_threshold = 0.10

    brain.homeostasis.theta_plus = THETA_PLUS
    brain.homeostasis.theta_leak = THETA_LEAK
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE

    return brain


def train_snn(brain, X_train, y_train):
    norm_target = compute_norm_target(brain)

    order = np.random.default_rng(SEED).permutation(len(X_train))
    for idx in order:
        present_sample(brain, X_train[idx], PRESENT_STEPS, learn=True)
        normalize_feedforward_weights(brain, norm_target)
        reset_brain_state(brain, REST_STEPS)
    return norm_target


def eval_snn(brain, X_readout, y_readout, X_test, y_test):
    _, _, spike_t, voltage_t = build_readout(brain, X_readout, y_readout, CLASSES)
    acc, _ = evaluate(brain, X_test, y_test, spike_t, voltage_t, CLASSES)
    return acc


def _weight_targets(brain):
    return {
        **{f"region:{name}": region for name, region in brain.regions.items()},
        **{f"proj:{p.source_name}->{p.target_name}": p for p in brain.projections},
    }


def sleep_phase(brain, n_steps, enable_replay=True, *, enable_stdp=True):
    """Controlled sleep; return actual weight changes, not inferred learning.

    Both plastic and frozen arms disable R-STDP, scaling, theta adaptation and
    growth. Only the plastic arms call feedforward STDP. Automatic capture is
    off; explicit replay draws from pre-sleep traces at the configured cadence.
    Pending eligibility/dopamine are discarded at entry. Caller control flags
    are restored even if a step fails.
    """
    if n_steps < 0:
        raise ValueError("Sleep duration must be nonnegative")
    projection = brain.get_projection("input", "cortex")
    if enable_stdp and not projection.plasticity_enabled:
        raise ValueError("Sleep STDP requires an enabled feedforward pathway")
    targets = _weight_targets(brain)
    before = {key: t.syn_weight[:t.n_synapses].clone() for key, t in targets.items()}
    flags = (
        brain.homeostasis.scaling_enabled, brain.homeostasis.theta_enabled,
        brain.metaplasticity_enabled, brain.growth.growth_interval,
        brain.memory.enabled, brain.reward_stdp.enabled,
    )
    replayed = 0
    stdp_steps = 0
    brain.freeze_homeostatic_scaling()
    brain.freeze_adaptive_thresholds()
    brain.freeze_structural_plasticity()
    brain.disable_reward_modulated_plasticity()
    brain.disable_memory()
    try:
        for _ in range(n_steps):
            brain.step()
            if enable_stdp:
                apply_feedforward_stdp(brain)
                stdp_steps += 1
            if enable_replay:
                brain.memory.enabled = True
                try:
                    replayed += brain.memory.consolidate(brain.regions, brain.time)
                finally:
                    brain.memory.enabled = False
    finally:
        (
            brain.homeostasis.scaling_enabled, brain.homeostasis.theta_enabled,
            brain.metaplasticity_enabled, brain.growth.growth_interval,
            brain.memory.enabled, brain.reward_stdp.enabled,
        ) = flags

    changes = {}
    for key, target in targets.items():
        weights = target.syn_weight[:target.n_synapses]
        delta = weights - before[key]
        changes[key] = {
            "synapses": target.n_synapses,
            "changed": int(torch.count_nonzero(delta)),
            "mean_abs_delta": float(delta.abs().mean()) if delta.numel() else 0.0,
            "max_abs_delta": float(delta.abs().max()) if delta.numel() else 0.0,
            "net_delta": float(delta.sum()),
        }
        allowed = enable_stdp and key == "proj:input->cortex"
        if not allowed and changes[key]["changed"]:
            raise AssertionError(f"Unexpected weight update in {key}")
    return {
        "replay": enable_replay, "stdp": enable_stdp,
        "scaling": False, "adaptive_theta": False,
        "steps": n_steps, "stdp_steps": stdp_steps,
        "replayed_traces": replayed, "weight_changes": changes,
    }


def compare_sleep_conditions(brain, X_readout, y_readout, X_test, y_test, n_steps):
    """Use one decoder and identical pre-sleep copies for the four arms."""
    _, _, spike_t, voltage_t = build_readout(brain, X_readout, y_readout, CLASSES)
    before, _ = evaluate(brain, X_test, y_test, spike_t, voltage_t, CLASSES)
    results = {"accuracy_before": before, "decoder": "fixed_pre_sleep", "conditions": []}
    print(f"    Accuracy before sleep: {before:.1%}")
    for replay, stdp in ((True, True), (False, True), (True, False), (False, False)):
        print(f"  Sleep: replay={replay} STDP={stdp} ({n_steps} steps)...", flush=True)
        candidate = copy.deepcopy(brain)
        start = time.time()
        stats = sleep_phase(candidate, n_steps, replay, enable_stdp=stdp)
        accuracy, _ = evaluate(candidate, X_test, y_test, spike_t, voltage_t, CLASSES)
        stats.update(accuracy=accuracy, elapsed_s=time.time() - start)
        results["conditions"].append(stats)
        changed = sum(item["changed"] for item in stats["weight_changes"].values())
        print(f"    Accuracy={accuracy:.1%}, changed weights={changed}, replayed traces={stats['replayed_traces']}")
    return results


# --- Main -------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  SLEEP / REPLAY BENEFIT BENCHMARK")
    print(f"  (Train, sleep {SLEEP_STEPS} steps ± replay, re-evaluate)")
    print("=" * 68)

    X_train, y_train, X_test, y_test = load_reduced_mnist(
        classes=CLASSES, train_per_class=TRAIN_PER_CLASS, test_per_class=TEST_PER_CLASS,
    )
    X_readout, y_readout = build_readout_subset(
        X_train,
        y_train,
        readout_per_class=READOUT_PER_CLASS,
        classes=CLASSES,
        seed=SEED,
    )

    # --- Train ---
    print("\n  Training SNN...")
    t0 = time.time()
    brain = build_brain_with_memory(seed=SEED)
    train_snn(brain, X_train, y_train)
    print(f"    done in {time.time() - t0:.0f}s")
    print(f"    Memory traces captured: {len(brain.memory.traces)}")

    # Preserve this protocol's checkpoint separately from historical runs.
    run_dir = Path("results") / f"sleep_{time.time_ns()}"
    save_brain(brain, run_dir / "checkpoint")
    results = compare_sleep_conditions(
        brain, X_readout, y_readout, X_test, y_test, SLEEP_STEPS,
    )
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    # --- Summary ---
    print("\n" + "=" * 68)
    print("  Summary")
    print("=" * 68)
    print(f"  Before sleep: {results['accuracy_before']:.1%}")
    for item in results["conditions"]:
        print(f"  Replay={item['replay']} STDP={item['stdp']}: {item['accuracy']:.1%}")
    active, control, replay_frozen, silent_frozen = results["conditions"]
    print(f"  Replay difference with STDP: {active['accuracy'] - control['accuracy']:+.1%}")
    print(f"  Replay difference with frozen weights: {replay_frozen['accuracy'] - silent_frozen['accuracy']:+.1%}")
    print("  Single seed; these differences do not establish statistical significance.")
    print(f"  Saved checkpoint and summary: {run_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
