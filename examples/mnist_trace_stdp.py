"""Experimental additive, post-triggered trace STDP for the MNIST learning check.

At each presynaptic arrival, x += 1; between steps x decays with time constant
tau. On a postsynaptic spike, dw = A_plus * learning_rate * (x - target).
There is no presynaptic-triggered weight update and A_minus is not used.

The trace and arrival history belong to ONE image presentation: they start at
zero and are discarded before rest. Neural state and queued currents are not
reset here. This boundary policy differs from the existing pair rule, which
retains last-spike times. Only completed-image checkpoints can resume this rule.
The projection topology and simulation dt must stay fixed during a presentation.
"""

from __future__ import annotations

import math

import torch

from src.region import MAX_DELAY_STEPS


def validate_learning_rule(rule, tau, target):
    if rule not in ("pair", "post_trace"):
        raise ValueError("Unknown MNIST learning rule")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Trace tau must be finite and positive")
    if not math.isfinite(target) or target < 0:
        raise ValueError("Trace target must be finite and nonnegative")


class PostTraceSTDP:
    """Call after every brain.step() of one presentation, including silent steps."""

    def __init__(self, brain, *, tau=20.0, target=0.2):
        validate_learning_rule("post_trace", tau, target)
        self.brain = brain
        self.projection = brain.get_projection("input", "cortex")
        self.source = brain.regions["input"]
        self.target_region = brain.regions["cortex"]
        self.ns = self.projection.n_synapses
        self.n_pre, self.n_post = self.source.n_neurons, self.target_region.n_neurons
        self.dt = brain.dt
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Trace STDP requires a finite positive dt")
        self.decay = math.exp(-self.dt / tau)
        self.target = target
        self.last_step = brain.step_count
        self.delays = self.projection.syn_delay[:self.ns].to(torch.int64).clone()
        if bool(((self.delays < 1) | (self.delays >= MAX_DELAY_STEPS)).any()):
            raise ValueError("Trace STDP requires valid positive synaptic delays")
        self.pre = self.projection.syn_pre[:self.ns].to(torch.int64).clone()
        self.ring_size = int(self.delays.max()) + 1 if self.ns else 1
        weights = self.projection.syn_weight[:self.ns]
        self.trace = torch.zeros_like(weights)
        self.history = torch.zeros((self.ring_size, self.n_pre), dtype=torch.bool,
                                   device=weights.device)
        self.cursor = 0

    def step(self):
        proj, brain = self.projection, self.brain
        if (proj.n_synapses != self.ns or self.source.n_neurons != self.n_pre
                or self.target_region.n_neurons != self.n_post or brain.dt != self.dt):
            raise ValueError("Trace STDP requires fixed topology sizes and dt within a presentation")
        if brain.step_count != self.last_step + 1:
            raise ValueError("Trace STDP must follow every brain step exactly once")
        self.last_step = brain.step_count
        # A source spike emitted now arrives only after its positive delay.
        self.history[self.cursor].copy_(self.source.fired[:self.n_pre])
        arrivals = self.history[(self.cursor - self.delays) % self.ring_size, self.pre]
        self.trace.mul_(self.decay).add_(arrivals)
        self.cursor = (self.cursor + 1) % self.ring_size
        if not proj.plasticity_enabled or self.ns == 0:
            return 0
        selected = proj._post_events.select(self.target_region.fired[:self.n_post],
                                            proj.syn_post[:self.ns], proj.syn_alive[:self.ns])
        if not selected.numel():
            return 0
        delta = (proj.syn_A_plus[selected] * brain.stdp.learning_rate
                 * (self.trace[selected] - self.target))
        proj.syn_weight[selected] = torch.clamp(proj.syn_weight[selected] + delta,
                                                proj.syn_min_weight[selected],
                                                proj.syn_max_weight[selected])
        return selected.numel()
