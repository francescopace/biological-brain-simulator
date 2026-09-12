"""Pre-empty-path Region.step, retained only for equivalence and timing checks."""

import torch

from src.device import DEVICE
from src.integration import integrate
from src.region import (
    MAX_DELAY_STEPS, _ACTIVITY_DECAY, _DEPLETION_RATE, _RECOVERY_RATE, _TRANSMISSION_DECAY,
)


def dense_step(self, time, step_count):
    n, ns, dt = self.n_neurons, self.n_synapses, self.dt
    if n == 0:
        return torch.tensor([], dtype=torch.int32, device=DEVICE)
    step = torch.tensor(step_count, dtype=torch.int64, device=DEVICE)
    s = slice(0, ns)
    self.syn_resource[s] = torch.clamp(self.syn_resource[s] + _RECOVERY_RATE, max=1.0)
    self.syn_facilitation[s] *= 0.98
    self.syn_age[s] += 1
    self.syn_recent[s] *= _TRANSMISSION_DECAY
    slot = step % MAX_DELAY_STEPS
    self.current[:n] += self.spike_buffer[slot, :n]
    self.spike_buffer[slot, :n] = 0.0
    current = self.current[:n] - self.theta[:n]
    v, u, crossed = integrate(self.v[:n], self.u[:n], current, self.a[:n], self.b[:n], dt,
                              method=self.integration_method, max_step=self.integration_max_step)
    fired = crossed & self.neuron_alive[:n]
    v = torch.where(fired, self.c[:n], v)
    u = torch.where(fired, u + self.d[:n], u)
    self.v[:n], self.u[:n] = v, u
    fired_idx = torch.where(fired)[0]
    self.last_spike_time[fired_idx] = time
    self.total_spikes[fired_idx] += 1
    self.fired[:n] = fired
    self.activity[:n] = self.activity[:n] * _ACTIVITY_DECAY + fired.to(torch.float32)
    self.neuron_age[:n] += 1
    self.current[:n] = 0.0
    active = self._pre_events.select(fired, self.syn_pre[:ns], self.syn_alive[:ns])
    resource, facilitation = self.syn_resource[active], self.syn_facilitation[active]
    effective = (self.syn_weight[active] * resource * (1.0 + facilitation)
                 * self.syn_modulation[active] * self.syn_attenuation[active])
    self.syn_resource[active] = torch.clamp(resource - _DEPLETION_RATE, min=0.0)
    self.syn_facilitation[active] += 0.05
    self.syn_total_tx[active] += 1
    self.syn_recent[active] += 1.0
    target_slots = (step + self.syn_delay[active]) % MAX_DELAY_STEPS
    target_neurons = self.syn_post[active].to(torch.int64)
    flat = target_slots.to(torch.int64) * self.spike_buffer.size(1) + target_neurons
    self.spike_buffer.view(-1).index_add_(0, flat, effective)
    return fired_idx
