"""
Plasticity rules operating on vectorized synapse arrays.

- STDP (Spike-Timing Dependent Plasticity)
- Reward-Modulated STDP (three-factor: STDP × dopamine)
- Homeostatic plasticity (synaptic scaling)
- Metaplasticity (BCM theory — plasticity of plasticity)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .region import Region


class STDP:
    """Vectorized STDP across arrays of synapses."""

    def __init__(self, learning_rate: float = 1.0):
        self.learning_rate = learning_rate
        self.stdp_window: float = 50.0
        self.tau_plus: float = 20.0
        self.tau_minus: float = 20.0

    def apply_to_arrays(
        self,
        pre_last_spike: np.ndarray,
        post_last_spike: np.ndarray,
        weights: np.ndarray,
        A_plus: np.ndarray,
        A_minus: np.ndarray,
        alive: np.ndarray,
        min_weight: np.ndarray,
        max_weight: np.ndarray,
        current_time: float,
        eligibility: np.ndarray | None = None,
    ) -> int:
        """
        Apply nearest-neighbor STDP to synapse arrays.

        If eligibility is provided, dw is accumulated into the eligibility
        trace instead of being applied directly (for reward-modulated STDP).
        """
        t_min = current_time - self.stdp_window
        has_pre = pre_last_spike > t_min
        has_post = post_last_spike > t_min
        elig_mask = has_pre & has_post & alive

        if not np.any(elig_mask):
            return 0

        dt = post_last_spike[elig_mask] - pre_last_spike[elig_mask]
        within = np.abs(dt) < self.stdp_window
        if not np.any(within):
            return 0

        idx = np.where(elig_mask)[0][within]
        dt_w = dt[within] * self.learning_rate

        dw = np.zeros(len(dt_w))
        ltp = dt_w > 0
        ltd = dt_w < 0
        dw[ltp] = A_plus[idx[ltp]] * np.exp(-dt_w[ltp] / self.tau_plus)
        dw[ltd] = -A_minus[idx[ltd]] * np.exp(dt_w[ltd] / self.tau_minus)

        if eligibility is not None:
            # Three-factor rule: accumulate into eligibility trace
            eligibility[idx] += dw
        else:
            weights[idx] += dw
            weights[idx] = np.clip(weights[idx], min_weight[idx], max_weight[idx])

        return int(np.count_nonzero(dw))


class RewardModulatedSTDP:
    """
    Three-factor learning rule: STDP × neuromodulation.

    STDP computes a candidate weight change (dw) which is accumulated
    into an eligibility trace per synapse. The trace decays over time.
    When a reward (dopamine) or punishment signal arrives, the actual
    weight change is: Δw = eligibility × dopamine_level.

    This implements the biological principle: "neurons that fire together
    AND are rewarded, wire together."
    """

    def __init__(
        self,
        tau_eligibility: float = 1000.0,
        dopamine_decay: float = 0.995,
        baseline_dopamine: float = 0.0,
    ):
        self.tau_eligibility = tau_eligibility
        self.dopamine_decay = dopamine_decay
        self.baseline_dopamine = baseline_dopamine
        self.dopamine_level: float = baseline_dopamine

        self._stdp = STDP()

    def reward(self, amount: float = 1.0) -> None:
        """Deliver a reward signal (positive dopamine burst)."""
        self.dopamine_level += amount

    def punish(self, amount: float = 0.5) -> None:
        """Deliver a punishment signal (dopamine dip below baseline)."""
        self.dopamine_level -= amount

    def apply_to_arrays(
        self,
        pre_last_spike: np.ndarray,
        post_last_spike: np.ndarray,
        weights: np.ndarray,
        A_plus: np.ndarray,
        A_minus: np.ndarray,
        alive: np.ndarray,
        min_weight: np.ndarray,
        max_weight: np.ndarray,
        eligibility: np.ndarray,
        current_time: float,
    ) -> int:
        """
        1. Compute STDP dw and accumulate into eligibility trace
        2. Decay eligibility trace
        3. Apply eligibility × dopamine to weights
        4. Decay dopamine toward baseline
        """
        n = len(weights)

        # Step 1: STDP → eligibility (not directly to weights)
        changes = self._stdp.apply_to_arrays(
            pre_last_spike, post_last_spike,
            weights, A_plus, A_minus, alive,
            min_weight, max_weight, current_time,
            eligibility=eligibility,
        )

        # Step 2: Decay eligibility trace
        decay = np.exp(-1.0 / self.tau_eligibility)
        eligibility[:n] *= decay

        # Step 3: Apply eligibility × dopamine to weights
        if abs(self.dopamine_level) > 1e-6:
            dw = eligibility[:n] * self.dopamine_level
            mask = alive & (np.abs(dw) > 1e-8)
            if np.any(mask):
                weights[mask] += dw[mask]
                weights[:n] = np.clip(weights[:n], min_weight[:n], max_weight[:n])

        # Step 4: Decay dopamine toward baseline
        self.dopamine_level = (
            self.baseline_dopamine
            + (self.dopamine_level - self.baseline_dopamine) * self.dopamine_decay
        )

        return changes


class HomeostaticPlasticity:
    """Synaptic scaling to maintain target firing rates."""

    def __init__(
        self,
        target_rate: float = 5.0,
        scaling_rate: float = 0.001,
        check_interval: int = 100,
    ):
        self.target_rate = target_rate
        self.scaling_rate = scaling_rate
        self.check_interval = check_interval
        self._step_counter = 0

    def apply(self, regions: list[Region], current_time: float) -> None:
        self._step_counter += 1
        if self._step_counter % self.check_interval != 0:
            return

        for region in regions:
            n = region.n_neurons
            ns = region.n_synapses
            if n == 0 or ns == 0:
                continue

            activity = region.activity[:n]
            alive_n = region.neuron_alive[:n]
            target = self.target_rate * 0.001

            for i in range(n):
                if not alive_n[i]:
                    continue
                rate = activity[i]
                ratio = target / max(rate, 0.001)
                scale = 1.0 + self.scaling_rate * (ratio - 1.0)
                scale = np.clip(scale, 0.95, 1.05)

                mask = (region.syn_post[:ns] == i) & region.syn_alive[:ns]
                region.syn_weight[:ns][mask] *= scale
                region.syn_weight[:ns][mask] = np.clip(
                    region.syn_weight[:ns][mask],
                    region.syn_min_weight[:ns][mask],
                    region.syn_max_weight[:ns][mask],
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

            activity = region.activity[:n]

            for i in range(n):
                if not region.neuron_alive[i]:
                    continue
                rate = activity[i]
                mask = (region.syn_post[:ns] == i) & region.syn_alive[:ns]
                if not np.any(mask):
                    continue
                region.syn_A_plus[:ns][mask] = np.maximum(
                    0.001, 0.01 - self.adaptation_rate * rate
                )
                region.syn_A_minus[:ns][mask] = np.maximum(
                    0.001, 0.012 + self.adaptation_rate * rate * 0.5
                )
