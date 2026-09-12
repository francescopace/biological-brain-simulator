from __future__ import annotations

import copy
import numpy as np

from src.brain import Brain, DISABLED_GROWTH_INTERVAL


def inference_brain(brain: Brain, *, independent: bool = True) -> Brain:
    """Copy a fixed network for readout fitting or held-out evaluation."""
    snapshot = copy.deepcopy(brain)
    snapshot.freeze_structural_plasticity()
    snapshot.freeze_plasticity()
    snapshot.freeze_homeostatic_scaling()
    snapshot.freeze_adaptive_thresholds()
    snapshot.disable_reward_modulated_plasticity()
    snapshot.disable_memory()
    if independent:
        snapshot.disable_oscillations()
    snapshot.encoder.noise_level = 0.0
    return snapshot


def can_reuse_independent_inference(brain: Brain) -> bool:
    """Identical independent presentations require a deterministic frozen model."""
    return (
        brain.encoder.noise_level == 0.0 and not brain.oscillators.enabled
        and not brain.memory.enabled and not brain.reward_stdp.enabled
        and not brain.homeostasis.theta_enabled and not brain.homeostasis.scaling_enabled
        and not brain.metaplasticity_enabled
        and brain.growth.growth_interval == DISABLED_GROWTH_INTERVAL
        and all(not target.plasticity_enabled
                for target in [*brain.regions.values(), *brain.projections])
    )


def reset_independent_state(brain: Brain) -> None:
    """Reset electrical/STP history, preserving learned parameters and counters."""
    if brain.oscillators.enabled:
        raise ValueError("Independent inference requires oscillations disabled")
    for region in brain.regions.values():
        n = region.n_neurons
        region.v[:n] = -65.0
        region.u[:n] = region.b[:n] * region.v[:n]
        region.current[:n] = 0.0
        region.fired[:n] = False
        region.activity[:n] = 0.0
        region.last_spike_time[:n] = -float("inf")
        region.spike_buffer.zero_()
    for target in [*brain.regions.values(), *brain.projections]:
        ns = target.n_synapses
        target.syn_resource[:ns] = 1.0
        target.syn_facilitation[:ns] = 0.0
    brain.reset_traces()


def quiet_steps(brain: Brain, n_steps: int) -> None:
    for _ in range(n_steps):
        brain.step()


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64)
    cm = np.zeros((len(label_arr), len(label_arr)), dtype=np.int64)
    label_to_idx = {int(label): idx for idx, label in enumerate(label_arr)}
    for true_label, pred_label in zip(y_true, y_pred):
        pred_int = int(pred_label)
        if pred_int < 0:
            continue
        cm[label_to_idx[int(true_label)], label_to_idx[pred_int]] += 1
    return cm
