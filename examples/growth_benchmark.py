"""
Growth vs fixed-size benchmark.

Tests whether neurogenesis and synaptogenesis improve learning
compared to a fixed-topology network.

Protocol:
  1. Train SNN on MNIST with growth DISABLED (fixed topology) → accuracy_fixed
  2. Train SNN on MNIST with growth ENABLED → accuracy_growth
  3. Compare: does structural plasticity help?
  4. Report neuron/synapse counts before and after training

A positive result: growth-enabled network reaches higher accuracy
or comparable accuracy with fewer initial neurons.
"""

from __future__ import annotations

import sys
import time
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
)
from src.brain import Brain
from src.device import DEVICE
from src.neuron import NeuronType, FiringPattern
from src.region import RegionType

TRAIN_PER_CLASS = 300
TEST_PER_CLASS = 50
READOUT_PER_CLASS = 30
PRESENT_STEPS = 200
REST_STEPS = 50

# Start with fewer neurons so growth has room to help
N_CORTEX_EXC_START = 200
N_CORTEX_INH_START = 200
MAX_CORTEX = 800  # growth can add up to this


def build_brain_growth(seed: int, enable_growth: bool) -> Brain:
    """Build brain with or without structural plasticity."""
    from examples.mnist_benchmark import (
        N_INPUT, INPUT_TO_CORTEX_DENSITY, INPUT_WEIGHT_BOOST,
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
        connectivity=0.0, max_neurons=MAX_CORTEX,
    )
    for _ in range(N_CORTEX_EXC_START):
        cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    for _ in range(N_CORTEX_INH_START):
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

    brain.reward_stdp.apply_target = lambda *args, **kwargs: 0
    brain.reset_traces()
    brain.memory.capture_trace = lambda *args, **kwargs: None
    brain.memory.consolidate = lambda *args, **kwargs: 0

    if enable_growth:
        brain.growth.growth_interval = 200
        brain.growth.max_new_neurons_per_cycle = 2
        brain.growth.max_new_synapses_per_cycle = 10
        brain.metaplasticity_enabled = True
    else:
        brain.freeze_structural_plasticity()

    brain.homeostasis.theta_plus = THETA_PLUS
    brain.homeostasis.theta_leak = THETA_LEAK
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE

    return brain


def run_experiment(enable_growth: bool, X_train, y_train, X_readout, y_readout, X_test, y_test):
    label = "GROWTH" if enable_growth else "FIXED"
    print(f"\n  --- {label} ---")

    brain = build_brain_growth(seed=SEED, enable_growth=enable_growth)
    snap_before = brain._snapshot()
    print(f"    Before: {snap_before.total_neurons} neurons, {snap_before.total_synapses} synapses")

    norm_target = compute_norm_target(brain)

    t0 = time.time()
    order = np.random.default_rng(SEED).permutation(len(X_train))
    for i, idx in enumerate(order, 1):
        present_sample(brain, X_train[idx], PRESENT_STEPS, learn=True)
        normalize_feedforward_weights(brain, norm_target)
        reset_brain_state(brain, REST_STEPS)
        if i % 500 == 0:
            print(f"    sample {i}/{len(order)}  elapsed={time.time() - t0:.0f}s")
    print(f"    Training: {time.time() - t0:.0f}s")

    snap_after = brain._snapshot()
    print(f"    After:  {snap_after.total_neurons} neurons, {snap_after.total_synapses} synapses")
    if enable_growth and brain.growth.history:
        total_born = sum(g.neurons_born for g in brain.growth.history)
        total_created = sum(g.synapses_created for g in brain.growth.history)
        total_pruned = sum(g.synapses_pruned for g in brain.growth.history)
        total_died = sum(g.neurons_died for g in brain.growth.history)
        print(f"    Growth: +{total_born} neurons born, +{total_created} synapses, "
              f"-{total_pruned} pruned, -{total_died} died")

    _, _, spike_t, voltage_t = build_readout(brain, X_readout, y_readout, CLASSES)
    acc, _ = evaluate(brain, X_test, y_test, spike_t, voltage_t, CLASSES)
    print(f"    Accuracy: {acc:.1%}")
    return acc, snap_before, snap_after


# --- Main -------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  GROWTH vs FIXED-SIZE BENCHMARK")
    print(f"  (Start: {N_CORTEX_EXC_START} exc, max: {MAX_CORTEX})")
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

    acc_fixed, _, _ = run_experiment(False, X_train, y_train, X_readout, y_readout, X_test, y_test)
    acc_growth, _, _ = run_experiment(True, X_train, y_train, X_readout, y_readout, X_test, y_test)

    # --- Summary ---
    print("\n" + "=" * 68)
    print("  Summary")
    print("=" * 68)
    print(f"  Fixed topology:  {acc_fixed:.1%}")
    print(f"  With growth:     {acc_growth:.1%}")
    diff = acc_growth - acc_fixed
    print(f"  Difference:      {diff:+.1%}")
    if diff > 0.01:
        print("  -> Growth improves learning.")
    elif diff > -0.01:
        print("  -> No significant difference.")
    else:
        print("  -> Growth hurts (possible destabilization).")
    print("=" * 68)


if __name__ == "__main__":
    main()
