"""
Synapse type definitions and neurotransmitter properties.

Actual synaptic state (weights, delays, STP variables) is stored as
dense NumPy arrays in Region and Projection objects.
"""

from __future__ import annotations

import enum


class NeurotransmitterType(enum.Enum):
    GLUTAMATE = "glutamate"
    GABA = "gaba"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"


# (sign, modulation_factor) per neurotransmitter
NT_PROPERTIES: dict[NeurotransmitterType, tuple[float, float]] = {
    NeurotransmitterType.GLUTAMATE:     (+1.0, 1.0),
    NeurotransmitterType.GABA:          (-1.0, 1.0),
    NeurotransmitterType.DOPAMINE:      (+1.0, 1.5),
    NeurotransmitterType.SEROTONIN:     (+0.3, 0.8),
    NeurotransmitterType.ACETYLCHOLINE: (+0.5, 1.2),
}
