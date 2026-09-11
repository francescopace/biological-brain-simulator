"""Tests for plasticity rules: STDP, R-STDP, homeostasis, metaplasticity."""

from contextlib import ExitStack
import math

import torch
import pytest

from src.plasticity import STDP, RewardModulatedSTDP, HomeostaticPlasticity, Metaplasticity
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


class TestSTDP:
    def _make_synapse_arrays(self, n_syn=1, n_neurons=3):
        """Helper: minimal synapse arrays for a pre->post pair."""
        return dict(
            syn_pre=torch.tensor([0] * n_syn, dtype=torch.int32),
            syn_post=torch.tensor([1] * n_syn, dtype=torch.int32),
            A_plus=torch.full((n_syn,), 0.01),
            A_minus=torch.full((n_syn,), 0.012),
            alive=torch.ones(n_syn, dtype=torch.bool),
            weights=torch.full((n_syn,), 0.5),
            min_weight=torch.zeros(n_syn),
            max_weight=torch.full((n_syn,), 10.0),
        )

    def test_ltp_post_fires_after_pre(self):
        """Post fires at t=20 after pre fired at t=10 → LTP (weight increase)."""
        stdp = STDP()
        arrs = self._make_synapse_arrays()
        pre_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])
        post_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])

        # Step 1: pre fires at t=10
        fired_pre = torch.tensor([True, False, False])
        fired_post = torch.tensor([False, False, False])
        pre_last[0] = 10.0
        stdp.apply_event(fired_pre, fired_post, current_time=10.0,
                         pre_last_spike_arr=pre_last, post_last_spike_arr=post_last,
                         **arrs)
        w_after_pre = arrs["weights"][0].item()

        # Step 2: post fires at t=20
        fired_pre = torch.tensor([False, False, False])
        fired_post = torch.tensor([False, True, False])
        post_last[1] = 20.0
        stdp.apply_event(fired_pre, fired_post, current_time=20.0,
                         pre_last_spike_arr=pre_last, post_last_spike_arr=post_last,
                         **arrs)
        w_after_post = arrs["weights"][0].item()

        assert w_after_post > w_after_pre  # LTP

    def test_ltd_pre_fires_after_post(self):
        """Pre fires at t=20 after post fired at t=10 → LTD (weight decrease)."""
        stdp = STDP()
        arrs = self._make_synapse_arrays()
        pre_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])
        post_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])

        # Post fires at t=10
        fired_post = torch.tensor([False, True, False])
        post_last[1] = 10.0
        stdp.apply_event(
            torch.tensor([False, False, False]), fired_post, current_time=10.0,
            pre_last_spike_arr=pre_last, post_last_spike_arr=post_last, **arrs,
        )
        w_after_post = arrs["weights"][0].item()

        # Pre fires at t=20
        fired_pre = torch.tensor([True, False, False])
        pre_last[0] = 20.0
        stdp.apply_event(
            fired_pre, torch.tensor([False, False, False]), current_time=20.0,
            pre_last_spike_arr=pre_last, post_last_spike_arr=post_last, **arrs,
        )
        w_after_pre = arrs["weights"][0].item()

        assert w_after_pre < w_after_post  # LTD

    def test_weights_clamped(self):
        stdp = STDP(learning_rate=100.0)
        arrs = self._make_synapse_arrays()
        arrs["weights"][0] = 9.99
        arrs["max_weight"][0] = 10.0
        pre_last = torch.tensor([5.0, -float('inf'), -float('inf')])
        post_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])

        fired_post = torch.tensor([False, True, False])
        stdp.apply_event(
            torch.tensor([False, False, False]), fired_post, current_time=6.0,
            pre_last_spike_arr=pre_last, post_last_spike_arr=post_last, **arrs,
        )
        assert arrs["weights"][0].item() <= 10.0

    def test_dead_synapses_ignored(self):
        stdp = STDP()
        arrs = self._make_synapse_arrays()
        arrs["alive"][0] = False
        w_before = arrs["weights"][0].item()

        pre_last = torch.tensor([5.0, -float('inf'), -float('inf')])
        post_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])
        stdp.apply_event(
            torch.tensor([False, False, False]),
            torch.tensor([False, True, False]),
            current_time=6.0,
            pre_last_spike_arr=pre_last, post_last_spike_arr=post_last, **arrs,
        )
        assert arrs["weights"][0].item() == w_before

    def test_eligibility_mode(self):
        stdp = STDP()
        arrs = self._make_synapse_arrays()
        eligibility = torch.zeros(1)
        pre_last = torch.tensor([5.0, -float('inf'), -float('inf')])
        post_last = torch.tensor([-float('inf'), -float('inf'), -float('inf')])
        w_before = arrs["weights"][0].item()

        stdp.apply_event(
            torch.tensor([False, False, False]),
            torch.tensor([False, True, False]),
            current_time=6.0,
            pre_last_spike_arr=pre_last, post_last_spike_arr=post_last,
            eligibility=eligibility, **arrs,
        )
        # Weight unchanged, eligibility updated
        assert arrs["weights"][0].item() == w_before
        assert eligibility[0].item() != 0.0


class TestRewardModulatedSTDP:
    @staticmethod
    def _two_posts():
        return dict(
            fired_pre=torch.tensor([False]),
            fired_post=torch.tensor([True, True]),
            syn_pre=torch.tensor([0, 0], dtype=torch.int32),
            syn_post=torch.tensor([0, 1], dtype=torch.int32),
            pre_last_spike_arr=torch.tensor([0.0]),
            post_last_spike_arr=torch.tensor([1.0, 1.0]),
            weights=torch.tensor([0.5, 0.5]),
            A_plus=torch.tensor([0.01, 0.01]),
            A_minus=torch.tensor([0.012, 0.012]),
            alive=torch.tensor([True, True]),
            min_weight=torch.zeros(2),
            max_weight=torch.full((2,), 10.0),
            eligibility=torch.ones(2),
            current_time=1.0,
        )

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_eligibility_decay_depends_on_elapsed_milliseconds(self, dt):
        rstdp = RewardModulatedSTDP(tau_eligibility=200.0)
        arrays = self._two_posts()
        arrays["fired_post"].zero_()

        for _ in range(round(100.0 / dt)):
            rstdp.apply_target("test", **arrays, dt=dt)

        assert arrays["eligibility"].tolist() == pytest.approx(
            [math.exp(-100.0 / 200.0)] * 2, rel=1e-5,
        )

    def test_restriction_blocks_new_spikes_and_existing_traces_on_other_posts(self):
        rstdp = RewardModulatedSTDP()
        arrays = self._two_posts()
        rstdp.reward(7.0, "test")

        with rstdp.restrict_to_posts("test", [0]):
            assert rstdp.get("test") == rstdp.baseline_dopamine
            rstdp.reward(1.0, "test")
            # Both posts spike on each step while dopamine is active.
            for _ in range(3):
                rstdp.apply_target("test", **arrays)

        assert arrays["weights"][0] > 0.5
        assert arrays["weights"][1].item() == pytest.approx(0.5)
        assert arrays["eligibility"][1].item() == 0.0
        assert rstdp.get("test") == rstdp.baseline_dopamine

        before = arrays["weights"].clone()
        rstdp.apply_target("test", **arrays)
        assert arrays["eligibility"][1] > 0.0  # restriction has ended
        assert torch.equal(arrays["weights"], before)  # no pulse spills over

    def test_restriction_cleanup_on_exception(self):
        rstdp = RewardModulatedSTDP(baseline_dopamine=0.1)
        arrays = self._two_posts()

        with pytest.raises(RuntimeError, match="interrupted"):
            with rstdp.restrict_to_posts("test", [0]):
                rstdp.reward(1.0, "test")
                rstdp.apply_target("test", **arrays)
                raise RuntimeError("interrupted")

        assert rstdp.get("test") == rstdp.baseline_dopamine
        before = arrays["weights"].clone()
        rstdp.apply_target("test", **arrays)
        assert arrays["eligibility"][1] > 0.0
        assert torch.equal(arrays["weights"], before)

    def test_distinct_targets_can_be_restricted_together(self):
        rstdp = RewardModulatedSTDP()
        first, second = self._two_posts(), self._two_posts()
        rstdp.reward(2.0, "unrelated")

        with ExitStack() as stack:
            stack.enter_context(rstdp.restrict_to_posts("first", [0]))
            stack.enter_context(rstdp.restrict_to_posts("second", [1]))
            rstdp.reward(1.0, "first")
            rstdp.reward(1.0, "second")
            rstdp.apply_target("first", **first)
            rstdp.apply_target("second", **second)

        assert first["weights"][0] > 0.5
        assert first["weights"][1].item() == pytest.approx(0.5)
        assert second["weights"][0].item() == pytest.approx(0.5)
        assert second["weights"][1] > 0.5
        assert rstdp.get("first") == rstdp.get("second") == 0.0
        assert rstdp.get("unrelated") == 2.0

    def test_nested_restrictions_on_same_target_are_rejected(self):
        rstdp = RewardModulatedSTDP()
        with rstdp.restrict_to_posts("test", [0]):
            with pytest.raises(ValueError, match="already restricted"):
                with rstdp.restrict_to_posts("test", [1]):
                    pass

    def test_disabled_rule_does_not_update_eligibility_or_weights(self):
        rstdp = RewardModulatedSTDP()
        rstdp.enabled = False
        weights = torch.tensor([0.5])
        eligibility = torch.tensor([0.25])

        changes = rstdp.apply_target(
            target="test",
            fired_pre=torch.tensor([True, False]),
            fired_post=torch.tensor([False, True]),
            syn_pre=torch.tensor([0], dtype=torch.int32),
            syn_post=torch.tensor([1], dtype=torch.int32),
            pre_last_spike_arr=torch.tensor([1.0, -1000.0]),
            post_last_spike_arr=torch.tensor([-1000.0, 2.0]),
            weights=weights,
            A_plus=torch.tensor([0.01]),
            A_minus=torch.tensor([0.012]),
            alive=torch.tensor([True]),
            min_weight=torch.tensor([0.0]),
            max_weight=torch.tensor([10.0]),
            eligibility=eligibility,
            current_time=2.0,
        )

        assert changes == 0
        assert weights.item() == pytest.approx(0.5)
        assert eligibility.item() == pytest.approx(0.25)

    def test_reward_and_punish(self):
        rstdp = RewardModulatedSTDP()
        rstdp.reward(1.0, "region:motor")
        assert rstdp.get("region:motor") > 0
        rstdp.punish(2.0, "region:motor")
        assert rstdp.get("region:motor") < 0

    def test_dopamine_decays_on_apply(self):
        rstdp = RewardModulatedSTDP()
        rstdp.reward(1.0, "test")
        initial = rstdp.get("test")

        # Run apply_target with minimal arrays
        n = 3
        rstdp.apply_target(
            target="test",
            fired_pre=torch.zeros(n, dtype=torch.bool),
            fired_post=torch.zeros(n, dtype=torch.bool),
            syn_pre=torch.tensor([0], dtype=torch.int32),
            syn_post=torch.tensor([1], dtype=torch.int32),
            pre_last_spike_arr=torch.full((n,), -1000.0),
            post_last_spike_arr=torch.full((n,), -1000.0),
            weights=torch.tensor([0.5]),
            A_plus=torch.tensor([0.01]),
            A_minus=torch.tensor([0.012]),
            alive=torch.tensor([True]),
            min_weight=torch.tensor([0.0]),
            max_weight=torch.tensor([10.0]),
            eligibility=torch.tensor([0.0]),
            current_time=1.0,
        )
        assert rstdp.get("test") < initial  # dopamine decayed


class TestHomeostaticPlasticity:
    def test_activity_trace_is_converted_to_hz(self):
        r = Region("test", RegionType.SENSORY, max_neurons=2, dt=1.0)
        r.populate(2, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=2.0)
        # For decay=0.995, a raw EMA count of 1.0 represents 5 Hz.
        r.activity[1] = 1.0
        before = r.syn_weight[0].item()

        hp = HomeostaticPlasticity(
            target_rate=5.0,
            check_interval=1,
            scaling_rate=0.1,
            activity_decay=0.995,
        )
        hp.apply([r], 1.0)

        assert r.syn_weight[0].item() == pytest.approx(before)

    def test_underactive_target_reduces_inhibitory_magnitude(self):
        r = Region("test", RegionType.SENSORY, max_neurons=5)
        r.populate(2, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=2.0, nt=NeurotransmitterType.GABA)
        r.activity[1] = 0.001
        before = r.syn_weight[0].item()

        hp = HomeostaticPlasticity(check_interval=1, scaling_rate=0.1)
        hp.apply([r], 1.0)

        assert before < 0.0
        assert r.syn_weight[0].item() > before

    def test_theta_updates_on_fire(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.fired[0] = True
        hp = HomeostaticPlasticity(theta_plus=0.10, theta_leak=0.005)
        hp.apply([r], 1.0)
        assert r.theta[0].item() > 0.0
        assert r.theta[1].item() == pytest.approx(0.0)  # didn't fire

    def test_theta_leaks_without_firing(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.theta[0] = 1.0
        r.fired[0] = False
        hp = HomeostaticPlasticity(theta_plus=0.10, theta_leak=0.005)
        hp.apply([r], 1.0)
        assert r.theta[0].item() < 1.0

    def test_scaling_can_be_disabled_without_disabling_theta(self):
        r = Region("test", RegionType.SENSORY, max_neurons=2)
        r.populate(2, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=2.0)
        r.activity[1] = 10.0
        r.fired[0] = True
        before = r.syn_weight[0].item()

        hp = HomeostaticPlasticity(check_interval=1, scaling_rate=0.1)
        hp.scaling_enabled = False
        hp.apply([r], 1.0)

        assert r.syn_weight[0].item() == pytest.approx(before)
        assert r.theta[0].item() > 0.0

    def test_theta_can_be_disabled_independently(self):
        r = Region("test", RegionType.SENSORY, max_neurons=2)
        r.populate(2, connectivity=0.0)
        r.fired[0] = True

        hp = HomeostaticPlasticity()
        hp.theta_enabled = False
        hp.apply([r], 1.0)

        assert r.theta[0].item() == pytest.approx(0.0)


class TestMetaplasticity:
    def test_frozen_region_keeps_zero_learning_coefficients(self):
        r = Region("test", RegionType.SENSORY, max_neurons=2)
        r.populate(2, connectivity=0.0)
        r.plasticity_enabled = False
        r.add_one_synapse(0, 1)

        Metaplasticity().update_thresholds([r], 1.0)

        assert r.syn_A_plus[0].item() == 0.0
        assert r.syn_A_minus[0].item() == 0.0

    def test_high_activity_reduces_ltp(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.1)
        r._rng.manual_seed(0)
        if r.n_synapses == 0:
            r.add_one_synapse(0, 1)
        r.activity[1] = 0.5  # high post activity
        meta = Metaplasticity(adaptation_rate=0.01)
        meta.update_thresholds([r], 1.0)
        # A_plus should be reduced for synapses onto high-activity neurons
        post_idx = r.syn_post[0].item()
        if r.activity[post_idx].item() > 0:
            assert r.syn_A_plus[0].item() < 0.01
