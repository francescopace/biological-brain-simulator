"""Checked event indices must preserve selection, update order and simulation state."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from examples.mnist_optimization_check import simulation_digest
from src.brain import Brain
from src.persistence import load_brain, save_brain
from src.plasticity import STDP, RewardModulatedSTDP
from src.region import RegionType
from src.synaptic_events import SynapseEventIndex


@pytest.mark.parametrize("count", [0, 5, 3000])
@pytest.mark.parametrize("fraction", [0.0, 0.05, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("grouped", [False, True])
def test_selection_matches_scan_including_order(count, fraction, grouped):
    rng = np.random.default_rng(42)
    endpoints = torch.from_numpy(rng.integers(0, 64, size=count, dtype=np.int32))
    if grouped:
        endpoints = endpoints.sort().values
    alive = torch.from_numpy(rng.random(count) > 0.15)
    fired = torch.from_numpy(rng.random(64) < fraction)
    index = SynapseEventIndex()
    for _ in range(2):
        expected = torch.where(fired[endpoints] & alive)[0]
        actual = index.select(fired, endpoints, alive)
        assert actual.dtype == torch.int64
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("mutation", ["torch", "numpy", "data", "replacement", "pruning",
                                      "resize", "neuron_count", "noncontiguous"])
def test_mutable_endpoints_and_liveness_never_leave_stale_indices(mutation):
    endpoints = torch.arange(4096, dtype=torch.int32) % 64
    alive = torch.ones(4096, dtype=torch.bool)
    fired = torch.zeros(64, dtype=torch.bool)
    fired[3] = True
    index = SynapseEventIndex()
    index.select(fired, endpoints, alive)
    assert index._endpoints is not None
    if mutation == "torch":
        endpoints[::5] = 3
    elif mutation == "numpy":
        endpoints.numpy()[::5] = 3
    elif mutation == "data":
        endpoints.data[::5] = 3
    elif mutation == "replacement":
        endpoints = (endpoints + 1) % 64
    elif mutation == "pruning":
        alive.numpy()[::2] = False
    elif mutation == "resize":
        endpoints = torch.cat((endpoints, torch.full((300,), 3, dtype=torch.int32)))
        alive = torch.ones(len(endpoints), dtype=torch.bool)
    elif mutation == "neuron_count":
        fired = torch.cat((fired, torch.tensor([True])))
        endpoints[::5] = 64
    else:
        endpoints = endpoints[::2]
        alive = alive[::2]
    assert torch.equal(index.select(fired, endpoints, alive), torch.where(fired[endpoints] & alive)[0])


def test_negative_indices_and_invalid_dead_endpoints_preserve_scan_behavior():
    index = SynapseEventIndex()
    fired = torch.zeros(64, dtype=torch.bool)
    fired[-1] = True
    endpoints = torch.full((3000,), -1, dtype=torch.int64)
    alive = torch.ones(3000, dtype=torch.bool)
    assert torch.equal(index.select(fired, endpoints, alive), torch.where(fired[endpoints] & alive)[0])
    endpoints[0] = 64
    alive[0] = False
    with pytest.raises(IndexError):
        index.select(fired, endpoints, alive)


def test_inference_tensors_do_not_need_version_counters():
    with torch.inference_mode():
        endpoints = torch.arange(3000, dtype=torch.int32) % 64
        fired, alive = torch.zeros(64, dtype=torch.bool), torch.ones(3000, dtype=torch.bool)
        fired[7] = True
        index = SynapseEventIndex()
        index.select(fired, endpoints, alive)
        endpoints[0] = 7
        assert torch.equal(index.select(fired, endpoints, alive), torch.where(fired[endpoints] & alive)[0])


def test_capacity_and_active_neuron_views_share_the_same_cache():
    index = SynapseEventIndex()
    fired = torch.zeros(128, dtype=torch.bool)
    fired[3] = True
    endpoints, alive = torch.arange(3000, dtype=torch.int32) % 64, torch.ones(3000, dtype=torch.bool)
    expected = index.select(fired, endpoints, alive)
    pointers = index._pointers
    for length in (64, 128, 100, 64):
        assert torch.equal(index.select(fired[:length], endpoints, alive), expected)
        assert index._pointers is pointers
    # A shorter view that no longer covers an endpoint must still raise.
    with pytest.raises(IndexError):
        index.select(fired[:32], endpoints, alive)


def test_non_cpu_device_uses_torch_fallback_without_numpy(monkeypatch):
    fired = torch.empty(64, dtype=torch.bool, device="meta")
    endpoints = torch.empty(3000, dtype=torch.int64, device="meta")
    alive = torch.empty(3000, dtype=torch.bool, device="meta")
    marker = object()
    monkeypatch.setattr(SynapseEventIndex, "_scan", staticmethod(lambda *args: marker))
    assert SynapseEventIndex().select(fired, endpoints, alive) is marker


@pytest.mark.parametrize("eligibility_mode", [False, True])
def test_indexed_stdp_matches_reference_with_clipping_and_dead_synapses(eligibility_mode):
    rng = torch.Generator().manual_seed(15)
    n, ns = 64, 3000
    endpoints = {key: torch.randint(n, (ns,), dtype=torch.int32, generator=rng)
                 for key in ("syn_pre", "syn_post")}
    arrays = dict(
        **endpoints, A_plus=torch.rand(ns, generator=rng) * 0.1,
        A_minus=torch.rand(ns, generator=rng) * 0.12,
        alive=torch.rand(ns, generator=rng) > 0.2,
        weights=torch.rand(ns, generator=rng),
        min_weight=torch.zeros(ns), max_weight=torch.ones(ns),
        eligibility=torch.zeros(ns) if eligibility_mode else None,
    )
    reference, indexed = copy.deepcopy(arrays), copy.deepcopy(arrays)
    pre_last, post_last = torch.full((n,), -float("inf")), torch.full((n,), -float("inf"))
    rule = STDP(learning_rate=10.0)
    indices = (SynapseEventIndex(), SynapseEventIndex())
    for time in range(100):
        pre = torch.rand(n, generator=rng) < 0.08
        post = torch.rand(n, generator=rng) < 0.08
        pre_last[pre], post_last[post] = float(time), float(time)
        for arr, caches in ((reference, None), (indexed, indices)):
            rule.apply_event(pre, post, pre_last_spike_arr=pre_last, post_last_spike_arr=post_last,
                             current_time=float(time), event_indices=caches, **arr)
        assert torch.equal(reference["weights"], indexed["weights"])
        if eligibility_mode:
            assert torch.equal(reference["eligibility"], indexed["eligibility"])


def test_indexed_reward_restriction_matches_reference():
    n, ns = 64, 4096
    pre = torch.arange(ns, dtype=torch.int32) % n
    post = torch.arange(ns, dtype=torch.int32) // n
    arrays = dict(
        syn_pre=pre, syn_post=post, alive=torch.ones(ns, dtype=torch.bool),
        weights=torch.full((ns,), 0.5), A_plus=torch.full((ns,), 0.1),
        A_minus=torch.full((ns,), 0.12), min_weight=torch.zeros(ns), max_weight=torch.ones(ns),
        eligibility=torch.full((ns,), 0.01),
        pre_last_spike_arr=torch.zeros(n), post_last_spike_arr=torch.zeros(n),
    )
    fired = torch.zeros(n, dtype=torch.bool)
    fired[[1, 3, 5]] = True
    results = []
    for caches in (None, (SynapseEventIndex(), SynapseEventIndex())):
        arr = copy.deepcopy(arrays)
        rule = RewardModulatedSTDP()
        with rule.restrict_to_posts("test", [3]):
            rule.reward(0.3, "test")
            for time in range(5):
                rule.apply_target("test", fired, fired, **arr, current_time=float(time),
                                  event_indices=caches)
        results.append(arr)
    assert torch.equal(results[0]["weights"], results[1]["weights"])
    assert torch.equal(results[0]["eligibility"], results[1]["eligibility"])
    assert torch.all(results[1]["weights"][post != 3] == 0.5)


def test_simulation_growth_edits_and_save_load_match_dense_path(monkeypatch, tmp_path):
    monkeypatch.setattr(SynapseEventIndex, "min_synapses", 0)
    brain = Brain(seed=5)
    for name in ("input", "cortex"):
        brain.add_region(name, RegionType.ASSOCIATION, n_neurons=16,
                         connectivity=0.25, max_neurons=24)
    brain.connect_regions("input", "cortex", density=0.5)
    brain.growth.growth_interval = 10
    brain.growth.neurogenesis_threshold = -1.0  # Force births even in a quiet cortex.
    models = [copy.deepcopy(brain), copy.deepcopy(brain)]
    for step in range(100):
        for model, enabled in zip(models, (False, True)):
            with patch.object(SynapseEventIndex, "enabled", enabled):
                cortex = model.regions["cortex"]
                if step == 35:
                    cortex.syn_alive[:cortex.n_synapses:3] = False
                if step == 45:
                    if cortex.syn_pre.device.type == "cpu":
                        cortex.syn_pre.numpy()[0] = 3
                    else:
                        cortex.syn_pre[0] = 3
                    cortex.syn_post.data[1] = 4
                    cortex.add_one_synapse(2, 4, weight=3.0, delay_ms=3.0)
                model.stimulate("input", np.linspace(0.0, 1.0, 16))
                if step % 15 == 0:
                    model.reward(0.1, model.projection_target("input", "cortex"))
                model.step()
        assert simulation_digest(models[0]) == simulation_digest(models[1])
    assert models[1].growth.history
    cortex = models[1].regions["cortex"]
    if cortex.syn_pre.device.type == "cpu":
        assert cortex._pre_events._endpoints is not None
    else:
        assert cortex._pre_events._endpoints is None
    assert models[1].regions["cortex"].n_neurons > 16
    # Derived caches are not checkpoint state and are rebuilt on load/copy.
    saved = tmp_path / "brain"
    save_brain(models[1], saved)
    loaded = load_brain(saved)
    copied = copy.deepcopy(models[1])
    for model in (loaded, copied):
        assert model.regions["cortex"]._pre_events._endpoints is None
        model.stimulate("input", np.linspace(0.0, 1.0, 16))
        model.step()
    with patch.object(SynapseEventIndex, "enabled", False):
        models[0].stimulate("input", np.linspace(0.0, 1.0, 16))
        models[0].step()
    assert simulation_digest(loaded) == simulation_digest(models[0])
    assert simulation_digest(copied) == simulation_digest(models[0])
