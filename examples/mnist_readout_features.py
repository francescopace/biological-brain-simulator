"""Read-only feature transforms for labelled MNIST ridge readouts.

Voltage centering removes each image's mean across excitatory cortex neurons.
It does not change spike counts, neural state, or the SNN's learning rule.
"""

import numpy as np


FEATURE_MODES = ("standard", "voltage_centered")


def readout_features(responses, mode):
    """Per-image features; labels and other images never enter the transform."""
    if mode not in FEATURE_MODES:
        raise ValueError("Unknown readout feature mode")
    spikes = np.asarray(responses.spikes, dtype=np.float64)
    voltage = np.asarray(responses.voltages, dtype=np.float64)
    if (spikes.ndim != 2 or spikes.shape != voltage.shape or spikes.shape[1] == 0
            or spikes.shape[1] != len(responses.exc_indices)):
        raise ValueError("Readout requires matching sample-by-neuron arrays")
    if not np.isfinite(spikes).all() or not np.isfinite(voltage).all() or np.any(spikes < 0):
        raise ValueError("Readout responses must be finite with nonnegative spike counts")
    if mode == "voltage_centered":
        voltage = voltage - voltage.mean(axis=1, keepdims=True)
    return np.concatenate((spikes, voltage), axis=1)
