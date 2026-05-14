"""
Biological neuron type definitions and Izhikevich parameter tables.

The Izhikevich model captures the essential behavior of biological neurons
(regular spiking, bursting, chattering, fast spiking, etc.) with just two
differential equations:

    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)

    if v >= 30 mV then v = c, u = u + d

Parameters a, b, c, d define the neuron type and firing pattern.
Actual neuron state is stored as dense NumPy arrays in Region.
"""

from __future__ import annotations

import enum


class NeuronType(enum.IntEnum):
    EXCITATORY = 0
    INHIBITORY = 1


class FiringPattern(enum.Enum):
    """Biologically observed firing patterns reproducible by the Izhikevich model."""
    REGULAR_SPIKING = "RS"
    INTRINSICALLY_BURSTING = "IB"
    CHATTERING = "CH"
    FAST_SPIKING = "FS"
    LOW_THRESHOLD_SPIKING = "LTS"
    THALAMO_CORTICAL = "TC"
    RESONATOR = "RZ"


# Izhikevich parameters (a, b, c, d) for each firing pattern
PATTERN_PARAMS: dict[FiringPattern, tuple[float, float, float, float]] = {
    FiringPattern.REGULAR_SPIKING:       (0.02,  0.20, -65.0,  8.0),
    FiringPattern.INTRINSICALLY_BURSTING:(0.02,  0.20, -55.0,  4.0),
    FiringPattern.CHATTERING:            (0.02,  0.20, -50.0,  2.0),
    FiringPattern.FAST_SPIKING:          (0.10,  0.20, -65.0,  2.0),
    FiringPattern.LOW_THRESHOLD_SPIKING: (0.02,  0.25, -65.0,  2.0),
    FiringPattern.THALAMO_CORTICAL:      (0.02,  0.25, -65.0,  0.05),
    FiringPattern.RESONATOR:             (0.10,  0.26, -65.0,  2.0),
}

EXCITATORY_RATIO = 0.8
