"""Tests for plasticity rules: STDP, R-STDP, homeostasis, metaplasticity."""

import torch
import pytest

from src.plasticity import STDP, RewardModulatedSTDP, HomeostaticPlasticity, Metaplasticity
from src.region import Region, RegionType


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


class TestMetaplasticity:
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
