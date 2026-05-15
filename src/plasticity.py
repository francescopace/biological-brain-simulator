"""
Plasticity rules operating on vectorized synapse arrays.

- STDP (Spike-Timing Dependent Plasticity): event-driven. dw is computed
  *only* on the step where pre or post just fired — not every step.
- RewardModulatedSTDP: three-factor rule (STDP × dopamine). Dopamine
  is per-target (region or projection), so reward signals can be
  targeted at a specific pathway instead of flooding the whole brain.
- HomeostaticPlasticity: synaptic scaling to maintain target firing rates.
- Metaplasticity (BCM-inspired): shifts STDP thresholds with activity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .region import Region


class STDP:
    """Event-driven, vectorized STDP across arrays of synapses.

    Nearest-neighbor implementation: at each call, weights (or eligibility)
    are updated *only* for synapses whose pre or post neuron fired this
    step. A pre→post spike pair contributes exactly once to LTP and once
    to LTD as the events occur — not 50 times across the STDP window.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        stdp_window: float = 50.0,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def apply_event(
        self,
        fired_pre: torch.Tensor,            # bool [n_neurons_src]
        fired_post: torch.Tensor,           # bool [n_neurons_dst]
        syn_pre: torch.Tensor,              # int32 [n_syn]
        syn_post: torch.Tensor,             # int32 [n_syn]
        pre_last_spike_arr: torch.Tensor,   # float32 [n_neurons_src]
        post_last_spike_arr: torch.Tensor,  # float32 [n_neurons_dst]
        A_plus: torch.Tensor,
        A_minus: torch.Tensor,
        alive: torch.Tensor,
        weights: torch.Tensor,
        min_weight: torch.Tensor,
        max_weight: torch.Tensor,
        current_time: float,
        eligibility: torch.Tensor | None = None,
    ) -> int:
        """
        Apply nearest-neighbor STDP (sync-free for GPU performance).

        Computes LTP/LTD without CPU-GPU synchronization points
        (no torch.any() guards). Operations on empty index sets are near-zero cost.
        """
        window = self.stdp_window
        pre_i64 = syn_pre.to(torch.int64)
        post_i64 = syn_post.to(torch.int64)

        # ── LTP arm: post fired this step ────────────────────────────
        post_fired_syn = fired_post[post_i64] & alive
        idx_ltp = torch.where(post_fired_syn)[0]
        if idx_ltp.numel() > 0:
            dt = current_time - pre_last_spike_arr[pre_i64[idx_ltp]]
            within = (dt >= 0.0) & (dt < window)
            sel = idx_ltp[within]
            if sel.numel() > 0:
                dw = A_plus[sel] * torch.exp(-dt[within] / self.tau_plus) * self.learning_rate
                if eligibility is not None:
                    eligibility[sel] += dw
                else:
                    weights[sel] = torch.clamp(
                        weights[sel] + dw, min_weight[sel], max_weight[sel],
                    )

        # ── LTD arm: pre fired this step ─────────────────────────────
        pre_fired_syn = fired_pre[pre_i64] & alive
        idx_ltd = torch.where(pre_fired_syn)[0]
        if idx_ltd.numel() > 0:
            dt = current_time - post_last_spike_arr[post_i64[idx_ltd]]
            within = (dt >= 0.0) & (dt < window)
            sel = idx_ltd[within]
            if sel.numel() > 0:
                dw = -A_minus[sel] * torch.exp(-dt[within] / self.tau_minus) * self.learning_rate
                if eligibility is not None:
                    eligibility[sel] += dw
                else:
                    weights[sel] = torch.clamp(
                        weights[sel] + dw, min_weight[sel], max_weight[sel],
                    )

        return 0


class RewardModulatedSTDP:
    """
    Three-factor learning rule with per-target dopamine.

    STDP computes a candidate weight change (dw) which is accumulated into
    an eligibility trace per synapse. When a reward (positive dopamine) or
    punishment (negative dopamine) is delivered to a *specific target*
    (a region's internal synapses, or one projection), the actual weight
    change there is: Δw = eligibility × dopamine[target].

    Targets are strings:
      - "region:<name>"             for a region's internal synapses
      - "proj:<src>-><dst>"         for an inter-region projection

    No global dopamine: reward signals must specify a target so credit
    assignment stays sample-local.
    """

    def __init__(
        self,
        tau_eligibility: float = 200.0,
        dopamine_decay: float = 0.5,
        baseline_dopamine: float = 0.0,
    ):
        self.tau_eligibility = tau_eligibility
        self.dopamine_decay = dopamine_decay
        self.baseline_dopamine = baseline_dopamine
        self.dopamine: dict[str, float] = {}
        self._stdp = STDP()

    # ── External reinforcement signals ───────────────────────────────

    def reward(self, amount: float, target: str) -> None:
        """Deliver a positive dopamine pulse to `target`."""
        if not target:
            raise ValueError(
                "reward() requires an explicit target "
                "(e.g. 'region:motor' or 'proj:cortex->motor')"
            )
        self.dopamine[target] = (
            self.dopamine.get(target, self.baseline_dopamine) + amount
        )

    def punish(self, amount: float, target: str) -> None:
        """Deliver a negative dopamine dip to `target`."""
        if not target:
            raise ValueError(
                "punish() requires an explicit target "
                "(e.g. 'region:motor' or 'proj:cortex->motor')"
            )
        self.dopamine[target] = (
            self.dopamine.get(target, self.baseline_dopamine) - amount
        )

    def get(self, target: str) -> float:
        """Current dopamine level at `target` (baseline if unseen)."""
        return self.dopamine.get(target, self.baseline_dopamine)

    # ── Per-step plasticity application ──────────────────────────────

    def apply_target(
        self,
        target: str,
        fired_pre: torch.Tensor,
        fired_post: torch.Tensor,
        syn_pre: torch.Tensor,
        syn_post: torch.Tensor,
        pre_last_spike_arr: torch.Tensor,
        post_last_spike_arr: torch.Tensor,
        weights: torch.Tensor,
        A_plus: torch.Tensor,
        A_minus: torch.Tensor,
        alive: torch.Tensor,
        min_weight: torch.Tensor,
        max_weight: torch.Tensor,
        eligibility: torch.Tensor,
        current_time: float,
    ) -> int:
        """
        Single-step plasticity for a set of synapses identified by `target`:
          1. Event-driven STDP → accumulate into eligibility
          2. Decay eligibility toward 0
          3. If |dopamine[target]| > eps: weights += eligibility × dopamine
          4. Decay dopamine[target] toward baseline
        """
        # 1. STDP into eligibility
        changes = self._stdp.apply_event(
            fired_pre, fired_post,
            syn_pre, syn_post,
            pre_last_spike_arr, post_last_spike_arr,
            A_plus, A_minus, alive,
            weights, min_weight, max_weight,
            current_time,
            eligibility=eligibility,
        )

        # 2. Decay eligibility (single vectorized multiply)
        eligibility *= torch.exp(torch.tensor(-1.0 / self.tau_eligibility, device=eligibility.device))

        # 3. Apply eligibility × dopamine if this target has non-baseline dopamine
        da = self.dopamine.get(target, self.baseline_dopamine)
        if abs(da - self.baseline_dopamine) > 1e-6:
            dw = eligibility * da
            mask = alive & (torch.abs(dw) > 1e-8)
            weights[mask] = torch.clamp(
                weights[mask] + dw[mask], min_weight[mask], max_weight[mask],
            )

        # 4. Decay dopamine[target] toward baseline
        if target in self.dopamine:
            new_level = (
                self.baseline_dopamine
                + (self.dopamine[target] - self.baseline_dopamine) * self.dopamine_decay
            )
            # Prune entries that have effectively settled — keeps the dict small.
            if abs(new_level - self.baseline_dopamine) < 1e-9:
                del self.dopamine[target]
            else:
                self.dopamine[target] = new_level

        return changes


class HomeostaticPlasticity:
    """Synaptic scaling and adaptive excitability threshold (theta) to maintain target firing rates."""

    def __init__(
        self,
        target_rate: float = 5.0,
        scaling_rate: float = 0.001,
        check_interval: int = 100,
        theta_plus: float = 0.10,
        theta_leak: float = 0.005,
    ):
        self.target_rate = target_rate
        self.scaling_rate = scaling_rate
        self.check_interval = check_interval
        self.theta_plus = theta_plus
        self.theta_leak = theta_leak
        self._step_counter = 0

    def apply(self, regions: list[Region], current_time: float) -> None:
        self._step_counter += 1

        # Update adaptive thresholds (theta) every step
        for region in regions:
            n = region.n_neurons
            if n > 0:
                fired = region.fired[:n].to(torch.float32)
                region.theta[:n] = region.theta[:n] * (1.0 - self.theta_leak) + self.theta_plus * fired

        if self._step_counter % self.check_interval != 0:
            return

        for region in regions:
            n = region.n_neurons
            ns = region.n_synapses
            if n == 0 or ns == 0:
                continue

            syn = slice(0, ns)
            post_idx = region.syn_post[syn].to(torch.int64)
            alive_syn = region.syn_alive[syn]
            alive_n = region.neuron_alive[:n]
            mask = alive_syn & (post_idx < n) & alive_n[post_idx]

            activity = region.activity[:n]
            target = self.target_rate * 0.001
            weights = region.syn_weight[syn]
            min_weight = region.syn_min_weight[syn]
            max_weight = region.syn_max_weight[syn]
            post_rate = torch.clamp(activity[post_idx], min=0.001)
            ratio = target / post_rate
            scale = torch.clamp(1.0 + self.scaling_rate * (ratio - 1.0), 0.95, 1.05)

            weights[mask] *= scale[mask]
            weights[mask] = torch.clamp(
                weights[mask],
                min_weight[mask],
                max_weight[mask],
            )


class Metaplasticity:
    """BCM-inspired: shift STDP thresholds based on recent activity."""

    def __init__(self, adaptation_rate: float = 0.0001):
        self.adaptation_rate = adaptation_rate

    def update_thresholds(self, regions: list[Region], current_time: float) -> None:
        for region in regions:
            n = region.n_neurons
            ns = region.n_synapses
            if n == 0 or ns == 0:
                continue

            syn = slice(0, ns)
            post_idx = region.syn_post[syn].to(torch.int64)
            alive_syn = region.syn_alive[syn]
            alive_post = region.neuron_alive[:n]
            mask = alive_syn & (post_idx < n) & alive_post[post_idx]

            rate = region.activity[:n][post_idx]
            A_plus = region.syn_A_plus[syn]
            A_minus = region.syn_A_minus[syn]
            A_plus[mask] = torch.clamp(
                0.01 - self.adaptation_rate * rate[mask],
                min=0.001,
            )
            A_minus[mask] = torch.clamp(
                0.012 + self.adaptation_rate * rate[mask] * 0.5,
                min=0.001,
            )
