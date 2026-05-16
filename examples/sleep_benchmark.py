"""
Sleep / replay benefit benchmark.

Tests whether offline consolidation (replay of memory traces during
a silent phase) improves retention compared to no-replay.

Protocol:
  1. Train SNN on MNIST (300/class) for 1 epoch
  2. Evaluate → accuracy_before_sleep
  3. Run a silent "sleep" phase: no input, but memory consolidation
     replays stored traces and STDP is active
  4. Evaluate → accuracy_after_sleep
  5. Compare: does replay improve or maintain accuracy?
  6. Control: run same duration of silence WITHOUT replay → accuracy_no_replay

A positive result: post-replay accuracy >= pre-sleep accuracy, and
significantly better than the no-replay control.
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
from src.persistence import save_brain, load_brain

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
    brain.reward_stdp.apply_target = lambda *args, **kwargs: 0
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


def sleep_phase(brain, n_steps, enable_replay=True):
    """Run silent steps. If enable_replay, memory consolidation is active."""
    if not enable_replay:
        original_consolidate = brain.memory.consolidate
        brain.memory.consolidate = lambda *args, **kwargs: 0

    for _ in range(n_steps):
        brain.step()

    if not enable_replay:
        brain.memory.consolidate = original_consolidate


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
    norm_target = train_snn(brain, X_train, y_train)
    print(f"    done in {time.time() - t0:.0f}s")
    print(f"    Memory traces captured: {len(brain.memory.traces)}")

    # --- Evaluate before sleep ---
    print("  Evaluating before sleep...")
    acc_before = eval_snn(brain, X_readout, y_readout, X_test, y_test)
    print(f"    Accuracy before sleep: {acc_before:.1%}")

    # Save for control experiment
    save_path = Path("/tmp/sleep_brain")
    save_brain(brain, save_path)

    # --- Sleep WITH replay ---
    print(f"\n  Sleep phase WITH replay ({SLEEP_STEPS} steps)...")
    t1 = time.time()
    sleep_phase(brain, SLEEP_STEPS, enable_replay=True)
    print(f"    done in {time.time() - t1:.1f}s")

    acc_with_replay = eval_snn(brain, X_readout, y_readout, X_test, y_test)
    print(f"    Accuracy after sleep+replay: {acc_with_replay:.1%}")

    # --- Sleep WITHOUT replay (control) ---
    print(f"\n  Sleep phase WITHOUT replay ({SLEEP_STEPS} steps, control)...")
    brain_ctrl = load_brain(save_path)
    t2 = time.time()
    sleep_phase(brain_ctrl, SLEEP_STEPS, enable_replay=False)
    print(f"    done in {time.time() - t2:.1f}s")

    acc_no_replay = eval_snn(brain_ctrl, X_readout, y_readout, X_test, y_test)
    print(f"    Accuracy after sleep (no replay): {acc_no_replay:.1%}")

    # --- Summary ---
    print("\n" + "=" * 68)
    print("  Summary")
    print("=" * 68)
    print(f"  Before sleep:          {acc_before:.1%}")
    print(f"  After sleep+replay:    {acc_with_replay:.1%}  (delta: {acc_with_replay - acc_before:+.1%})")
    print(f"  After sleep (control): {acc_no_replay:.1%}  (delta: {acc_no_replay - acc_before:+.1%})")
    replay_benefit = acc_with_replay - acc_no_replay
    print(f"\n  Replay benefit: {replay_benefit:+.1%}")
    if replay_benefit > 0.01:
        print("  -> Replay consolidation improves retention.")
    elif replay_benefit > -0.01:
        print("  -> No significant replay effect detected.")
    else:
        print("  -> Replay appears to hurt (possible interference).")
    print("=" * 68)


if __name__ == "__main__":
    main()
