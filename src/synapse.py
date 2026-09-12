"""
Synapse type definitions and neurotransmitter properties.

Actual synaptic state (weights, delays, STP variables) is stored as
dense PyTorch tensors in Region and Projection objects.
"""

from __future__ import annotations

import enum

import torch


class NeurotransmitterType(enum.Enum):
    GLUTAMATE = 1
    GABA = 2
    DOPAMINE = 3
    SEROTONIN = 4
    ACETYLCHOLINE = 5


# (sign, modulation_factor) per neurotransmitter
NT_PROPERTIES: dict[NeurotransmitterType, tuple[float, float]] = {
    NeurotransmitterType.GLUTAMATE:     (+1.0, 1.0),
    NeurotransmitterType.GABA:          (-1.0, 1.0),
    NeurotransmitterType.DOPAMINE:      (+1.0, 1.5),
    NeurotransmitterType.SEROTONIN:     (+0.3, 0.8),
    NeurotransmitterType.ACETYLCHOLINE: (+0.5, 1.2),
}


def advance_synapse_state(target, ns: int, recovery_rate: float, transmission_decay: float) -> None:
    """Recover resources and advance traces/counters without temporary state arrays.

    Public tensors may use nonstandard dtypes or autograd. Keep the original
    assignment/graph behavior there; ordinary floating state uses native in-place
    operations, preserving the recovery, facilitation, age, recent-trace order.
    """
    arrays = (target.syn_resource, target.syn_facilitation, target.syn_age, target.syn_recent)
    s = slice(0, ns)
    if not arrays[0].is_floating_point() or any(a.requires_grad for a in arrays):
        target.syn_resource[s] = torch.clamp(target.syn_resource[s] + recovery_rate, max=1.0)
        target.syn_facilitation[s] *= .98
        target.syn_age[s] += 1
        target.syn_recent[s] *= transmission_decay
        return
    target.syn_resource[s].add_(recovery_rate).clamp_(max=1.0)
    target.syn_facilitation[s].mul_(.98)
    target.syn_age[s].add_(1)
    target.syn_recent[s].mul_(transmission_decay)
