"""Tests for Brain: region management, projections, step orchestration, reward/punish."""

import math

import torch
import pytest

from src.brain import Brain, Projection
from src.region import RegionType
from src.synapse import NeurotransmitterType


class TestRegionManagement:
    def test_add_region(self, brain):
        assert "input" in brain.regions
        assert "output" in brain.regions
        assert brain.regions["input"].n_neurons == 10
        assert brain.regions["output"].n_neurons == 10

    def test_add_region_returns_region(self, seed):
        b = Brain(seed=seed)
        r = b.add_region("test", RegionType.SENSORY, n_neurons=5)
        assert r is b.regions["test"]
        assert r.n_neurons == 5

    def test_duplicate_region_name_is_rejected(self, seed):
        b = Brain(seed=seed)
        b.add_region("test", RegionType.SENSORY, n_neurons=5)
        with pytest.raises(ValueError, match="already exists"):
            b.add_region("test", RegionType.MOTOR, n_neurons=5)

    def test_region_types_set_correctly(self, brain):
        assert brain.regions["input"].region_type == RegionType.SENSORY
        assert brain.regions["output"].region_type == RegionType.MOTOR


class TestProjections:
    def test_connect_creates_projection(self, brain):
        assert len(brain.projections) == 1
        proj = brain.projections[0]
        assert proj.source_name == "input"
        assert proj.target_name == "output"
        assert proj.n_synapses > 0

    def test_get_projection(self, brain):
        proj = brain.get_projection("input", "output")
        assert proj.source_name == "input"

    def test_get_projection_missing_raises(self, brain):
        with pytest.raises(KeyError):
            brain.get_projection("output", "input")

    def test_bidirectional_connect(self, seed):
        b = Brain(seed=seed)
        b.add_region("a", RegionType.ASSOCIATION, n_neurons=10, connectivity=0.0)
        b.add_region("b", RegionType.ASSOCIATION, n_neurons=10, connectivity=0.0)
        b.connect_regions("a", "b", density=0.5, bidirectional=True)
        assert len(b.projections) == 2
        names = {(p.source_name, p.target_name) for p in b.projections}
        assert ("a", "b") in names
        assert ("b", "a") in names

    def test_only_excitatory_neurons_project(self, brain):
        proj = brain.projections[0]
        src = brain.regions["input"]
        pre_types = src.neuron_type[proj.syn_pre[:proj.n_synapses].to(torch.int64)]
        # All presynaptic neurons in projection should be excitatory
        assert torch.all(pre_types == 0)  # EXCITATORY = 0


class TestStep:
    def test_step_advances_time(self, brain):
        assert brain.time == 0.0
        brain.step()
        assert brain.time == brain.dt

    def test_step_increments_count(self, brain):
        assert brain.step_count == 0
        brain.step()
        assert brain.step_count == 1

    def test_step_returns_fired_dict(self, brain):
        result = brain.step()
        assert isinstance(result, dict)
        assert "input" in result
        assert "output" in result

    def test_multiple_steps_stable(self, brain):
        for _ in range(100):
            brain.step()
        assert brain.step_count == 100
        assert brain.time == pytest.approx(100.0)

    def test_stimulate_injects_current(self, brain):
        brain.stimulate("input", [0.5] * 10)
        assert torch.any(brain.regions["input"].current[:10] > 0)

    def test_stimulated_region_fires(self, brain):
        any_fired = False
        for _ in range(20):
            brain.stimulate("input", [1.0] * 10)
            result = brain.step()
            if sum(len(v) for v in result.values()) > 0:
                any_fired = True
                break
        assert any_fired


class TestReward:
    def test_reward_increases_dopamine(self, brain):
        target = Brain.projection_target("input", "output")
        brain.reward(1.0, target)
        assert brain.dopamine(target) > 0

    def test_punish_decreases_dopamine(self, brain):
        target = Brain.region_target("output")
        brain.punish(1.0, target)
        assert brain.dopamine(target) < 0

    def test_dopamine_baseline_for_unseen_target(self, brain):
        assert brain.dopamine("region:nonexistent") == 0.0

    def test_reward_requires_target(self, brain):
        with pytest.raises(ValueError):
            brain.reward(1.0, "")

    def test_target_string_helpers(self):
        assert Brain.region_target("motor") == "region:motor"
        assert Brain.projection_target("a", "b") == "proj:a->b"


class TestCurrentInjection:
    def test_inject_current(self, brain):
        brain.inject_current("input", [0, 1, 2], 10.0)
        r = brain.regions["input"]
        assert r.current[0].item() == pytest.approx(10.0)
        assert r.current[1].item() == pytest.approx(10.0)
        assert r.current[2].item() == pytest.approx(10.0)
        assert r.current[3].item() == pytest.approx(0.0)

    def test_inject_current_out_of_range_ignored(self, brain):
        brain.inject_current("input", [999], 10.0)  # should not crash


class TestPlasticityControl:
    def test_freeze_plasticity(self, brain):
        brain.freeze_plasticity()
        for proj in brain.projections:
            ns = proj.n_synapses
            assert not proj.plasticity_enabled
            assert torch.all(proj.syn_A_plus[:ns] == 0)
            assert torch.all(proj.syn_A_minus[:ns] == 0)

    def test_freeze_discards_pending_learning_and_survives_metaplasticity(self, brain):
        brain.disable_oscillations()
        brain.freeze_homeostatic_scaling()
        region = brain.regions["input"]
        if region.n_synapses == 0:
            region.add_one_synapse(0, 1)
        proj = brain.projections[0]
        region.syn_eligibility[:region.n_synapses] = 0.5
        proj.syn_eligibility[:proj.n_synapses] = 0.5
        region_target = Brain.region_target(region.name)
        proj_target = Brain.projection_target(proj.source_name, proj.target_name)
        brain.reward(1.0, region_target)
        brain.reward(1.0, proj_target)

        brain.freeze_plasticity()
        assert brain.dopamine(region_target) == brain.dopamine(proj_target) == 0.0
        assert not torch.any(region.syn_eligibility[:region.n_synapses])
        assert not torch.any(proj.syn_eligibility[:proj.n_synapses])
        region_before = region.syn_weight[:region.n_synapses].clone()
        proj_before = proj.syn_weight[:proj.n_synapses].clone()

        for _ in range(5):
            brain.reward(1.0, region_target)
            brain.reward(1.0, proj_target)
            brain.inject_current("input", [0, 1], 30.0)
            brain.step()

        assert brain.metaplasticity_enabled
        assert not region.plasticity_enabled
        assert torch.all(region.syn_A_plus[:region.n_synapses] == 0.0)
        assert torch.all(region.syn_A_minus[:region.n_synapses] == 0.0)
        assert torch.equal(region.syn_weight[:region.n_synapses], region_before)
        assert torch.equal(proj.syn_weight[:proj.n_synapses], proj_before)
        assert brain.dopamine(region_target) == brain.dopamine(proj_target) == 0.0

    def test_synapses_added_to_frozen_region_remain_frozen(self, brain):
        brain.freeze_plasticity()
        region = brain.regions["input"]
        new_index = region.n_synapses
        region.add_one_synapse(0, 1)
        brain.step()

        assert region.syn_A_plus[new_index].item() == 0.0
        assert region.syn_A_minus[new_index].item() == 0.0
        assert region.syn_eligibility[new_index].item() == 0.0

    def test_enable_projection_plasticity(self, brain):
        brain.freeze_plasticity()
        proj = brain.enable_projection_plasticity("input", "output", A_plus=0.05, A_minus=0.06)
        ns = proj.n_synapses
        assert proj.plasticity_enabled
        assert not brain.regions["input"].plasticity_enabled
        assert torch.allclose(proj.syn_A_plus[:ns], torch.tensor(0.05))

    def test_reenabled_projection_learns_from_new_reward_only(self, brain):
        brain.disable_oscillations()
        brain.freeze_homeostatic_scaling()
        brain.freeze_plasticity()
        proj = brain.projections[0]
        ns = proj.n_synapses
        target = Brain.projection_target(proj.source_name, proj.target_name)
        proj.syn_eligibility[:ns] = 1.0
        brain.reward(1.0, target)
        brain.enable_projection_plasticity("input", "output")
        assert brain.dopamine(target) == 0.0
        assert not torch.any(proj.syn_eligibility[:ns])

        pre, post = int(proj.syn_pre[0]), int(proj.syn_post[0])
        proj.syn_weight[0] = 1.0
        brain.regions["input"].last_spike_time[pre] = brain.time
        brain.regions["output"].v[post] = 40.0
        brain.reward(1.0, target)
        brain.step()

        assert proj.syn_eligibility[0] > 0.0
        assert proj.syn_weight[0] > 1.0

    def test_brain_passes_elapsed_time_to_region_and_projection_eligibility(self):
        brain = Brain(dt=2.0, seed=42)
        region = brain.add_region("input", RegionType.SENSORY, n_neurons=2, connectivity=0.0)
        brain.add_region("output", RegionType.MOTOR, n_neurons=2, connectivity=0.0)
        region.add_one_synapse(0, 1)
        brain.connect_regions("input", "output", density=1.0)
        brain.disable_oscillations()
        brain.freeze_homeostatic_scaling()
        brain.freeze_structural_plasticity()
        proj = brain.projections[0]
        region.syn_eligibility[0] = 1.0
        proj.syn_eligibility[:proj.n_synapses] = 1.0

        for _ in range(50):
            brain.step()

        expected = math.exp(-100.0 / brain.reward_stdp.tau_eligibility)
        assert region.syn_eligibility[0].item() == pytest.approx(expected, rel=1e-5)
        assert proj.syn_eligibility[:proj.n_synapses].tolist() == pytest.approx(
            [expected] * proj.n_synapses, rel=1e-5,
        )

    def test_reset_traces(self, brain):
        brain.reward(1.0, Brain.projection_target("input", "output"))
        brain.reset_traces()
        assert brain.dopamine(Brain.projection_target("input", "output")) == 0.0


class TestRun:
    def test_run_completes(self, brain):
        brain.run(50, verbose=False)
        assert brain.step_count == 50

    def test_run_with_stimulus_fn(self, brain):
        calls = []
        def stim(b, i):
            calls.append(i)
            b.stimulate("input", [0.5] * 10)
        brain.run(10, stimulus_fn=stim, verbose=False)
        assert len(calls) == 10


class TestInspection:
    def test_summary_string(self, brain):
        s = brain.summary()
        assert "Brain Summary" in s
        assert "input" in s
        assert "output" in s

    def test_repr(self, brain):
        s = repr(brain)
        assert "Brain(" in s
        assert "regions=2" in s

    def test_snapshot_counts(self, brain):
        snap = brain._snapshot()
        assert snap.total_neurons == 20
        assert snap.total_synapses > 0
