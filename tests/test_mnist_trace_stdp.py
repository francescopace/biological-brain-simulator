"""Arrival-timed, presentation-local STDP and preservation of the pair default."""

import copy
from dataclasses import asdict, replace
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig
from examples.mnist_optimization_check import reference_present, simulation_digest
from examples.mnist_trace_stdp import PostTraceSTDP
from src.region import MAX_DELAY_STEPS
from src.synaptic_events import SynapseEventIndex


def toy_brain(delays=(1, 3, 1), *, indexed=False):
    ns = len(delays)
    projection = SimpleNamespace(
        n_synapses=ns, syn_delay=torch.tensor(delays, dtype=torch.int32),
        syn_pre=torch.arange(ns, dtype=torch.int32), syn_post=torch.zeros(ns, dtype=torch.int32),
        syn_alive=torch.ones(ns, dtype=torch.bool), syn_weight=torch.ones(ns),
        syn_A_plus=torch.full((ns,), .1), syn_A_minus=torch.full((ns,), 999.),
        syn_min_weight=torch.zeros(ns), syn_max_weight=torch.full((ns,), 2.),
        plasticity_enabled=True, _post_events=SynapseEventIndex(),
    )
    if indexed:
        projection._post_events.min_synapses = 0
    source = SimpleNamespace(n_neurons=ns, fired=torch.zeros(ns, dtype=torch.bool))
    target = SimpleNamespace(n_neurons=8, fired=torch.zeros(8, dtype=torch.bool))
    return SimpleNamespace(dt=1., step_count=0, stdp=SimpleNamespace(learning_rate=.5),
                           regions={"input": source, "cortex": target},
                           get_projection=lambda *args: projection)


def tick(rule, pre=(), post=()):
    rule.source.fired.zero_()
    rule.source.fired[list(pre)] = True
    rule.target_region.fired.zero_()
    rule.target_region.fired[list(post)] = True
    rule.brain.step_count += 1
    return rule.step()


@pytest.mark.parametrize("indexed", [False, True])
def test_arrivals_accumulate_decay_and_only_post_spikes_update(indexed):
    brain = toy_brain(indexed=indexed)
    rule = PostTraceSTDP(brain, tau=2., target=.2)
    weights = rule.projection.syn_weight
    tick(rule, pre=(0, 1))
    torch.testing.assert_close(rule.trace, torch.zeros(3), rtol=0, atol=0)
    tick(rule, pre=(0,))
    torch.testing.assert_close(rule.trace, torch.tensor([1., 0., 0.]), rtol=0, atol=0)
    assert torch.equal(weights, torch.ones(3))  # No update on presynaptic events.
    tick(rule)
    torch.testing.assert_close(rule.trace, torch.tensor([1. + math.exp(-.5), 0., 0.]))
    tick(rule, post=(0,))
    expected_trace = torch.tensor([(1. + math.exp(-.5)) * math.exp(-.5), 1., 0.])
    torch.testing.assert_close(rule.trace, expected_trace)
    torch.testing.assert_close(weights, 1. + .05 * (expected_trace - .2))
    assert weights[2] < 1.  # Inactive input is depressed when the target fires.


def test_ring_wrap_and_maximum_delay_match_explicit_event_history():
    rule = PostTraceSTDP(toy_brain(delays=(1, 2, MAX_DELAY_STEPS - 1)), tau=7.)
    expected = torch.zeros(3)
    emissions = []
    rng = np.random.default_rng(19)
    for step in range(100):
        fired = rng.random(3) < .3
        emissions.append(fired)
        tick(rule, pre=tuple(np.flatnonzero(fired)))
        arrivals = torch.tensor([step >= delay and emissions[step - delay][i]
                                 for i, delay in enumerate((1, 2, MAX_DELAY_STEPS - 1))])
        expected = expected * math.exp(-1. / 7.) + arrivals
        assert torch.equal(rule.trace, expected)


def test_dead_disabled_zero_amplitude_and_bounds_are_respected():
    rule = PostTraceSTDP(toy_brain(delays=(1,) * 4), target=10.)
    proj = rule.projection
    proj.syn_alive[1] = False
    proj.syn_A_plus[2] = 0.
    proj.syn_weight[0] = .01
    proj.syn_weight[3] = 1.99
    proj.syn_A_plus[3] = 10.
    tick(rule, pre=(3,))
    rule.target = .2
    tick(rule, post=(0,))
    torch.testing.assert_close(proj.syn_weight, torch.tensor([0., 1., 1., 2.]), atol=1e-8, rtol=0)
    before = proj.syn_weight.clone()
    proj.plasticity_enabled = False
    tick(rule, post=(0,))
    assert torch.equal(proj.syn_weight, before)


def test_only_fired_postsynaptic_neurons_are_updated():
    rule = PostTraceSTDP(toy_brain())
    rule.projection.syn_post[:] = torch.tensor([0, 1, 2])
    tick(rule, post=(1,))
    torch.testing.assert_close(rule.projection.syn_weight, torch.tensor([1., .99, 1.]))


def test_trace_state_is_not_shared_between_presentations():
    brain = toy_brain()
    first = PostTraceSTDP(brain)
    tick(first, pre=(0, 1))
    tick(first)
    second = PostTraceSTDP(brain)
    assert torch.count_nonzero(first.trace) == 1
    assert torch.count_nonzero(second.trace) == 0
    assert torch.count_nonzero(second.history) == 0
    tick(second, post=(0,))
    torch.testing.assert_close(second.projection.syn_weight, torch.full((3,), .99))


@pytest.mark.parametrize("delay", [0, -1, MAX_DELAY_STEPS])
def test_invalid_delay_is_rejected_before_changing_brain(delay):
    brain = toy_brain(delays=(delay,))
    with pytest.raises(ValueError, match="synaptic delays"):
        PostTraceSTDP(brain)
    assert brain.step_count == 0
    assert brain.get_projection().syn_weight.item() == 1.


def test_duplicate_missing_steps_or_changed_sizes_are_rejected():
    brain = toy_brain()
    rule = PostTraceSTDP(brain)
    with pytest.raises(ValueError, match="every brain step"):
        rule.step()
    brain.step_count += 2
    with pytest.raises(ValueError, match="every brain step"):
        rule.step()
    rule.projection.n_synapses += 1
    with pytest.raises(ValueError, match="fixed topology"):
        rule.step()


def test_empty_projection_is_supported():
    rule = PostTraceSTDP(toy_brain(delays=()))
    assert tick(rule, post=(0,)) == 0


@pytest.fixture
def small_brain(monkeypatch):
    for name, value in {"N_INPUT": 4, "N_CORTEX_EXC": 8, "N_CORTEX_INH": 8,
                        "INPUT_TO_CORTEX_DENSITY": .8}.items():
        monkeypatch.setattr(mn, name, value)
    return mn.build_brain(seed=5)


@pytest.mark.parametrize("field,value", [("learning_rule", "unknown"), ("trace_tau", 0.),
    ("trace_tau", -1.), ("trace_tau", float("nan")), ("trace_tau", float("inf")),
    ("trace_target", -.1), ("trace_target", float("nan")), ("trace_target", float("inf"))])
def test_invalid_parameters_fail_before_simulation(small_brain, field, value):
    before = simulation_digest(small_brain)
    with pytest.raises(ValueError):
        replace(LearningConfig(), **{field: value}).validate()
    with pytest.raises(ValueError):
        mn.present_sample(small_brain, np.ones(4), 5, learn=True, **{field: value})
    assert simulation_digest(small_brain) == before


def test_pair_default_matches_original_loop_state_rng_and_responses(small_brain):
    reference = copy.deepcopy(small_brain)
    explicit = copy.deepcopy(small_brain)
    image = np.ones(4)
    expected = reference_present(reference, image, 50)
    default = mn.present_sample(small_brain, image, 50, learn=True)
    selected = mn.present_sample(explicit, image, 50, learn=True, learning_rule="pair")
    assert simulation_digest(reference) == simulation_digest(small_brain) == simulation_digest(explicit)
    for actual in (default, selected):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


def test_inference_never_constructs_trace_rule_or_changes_default(small_brain, monkeypatch):
    reference = copy.deepcopy(small_brain)
    monkeypatch.setattr(mn, "PostTraceSTDP", lambda *args, **kwargs: pytest.fail("unexpected trace allocation"))
    expected = mn.present_sample(reference, np.ones(4), 30)
    actual = mn.present_sample(small_brain, np.ones(4), 30, learning_rule="post_trace")
    assert simulation_digest(reference) == simulation_digest(small_brain)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_presentation_constructs_one_fresh_trace_and_never_calls_pair(small_brain, monkeypatch):
    created = []

    def construct(brain, **kwargs):
        rule = PostTraceSTDP(brain, **kwargs)
        created.append(rule)
        return rule

    monkeypatch.setattr(mn, "PostTraceSTDP", construct)
    monkeypatch.setattr(mn, "apply_feedforward_stdp", lambda *args: pytest.fail("pair applied"))
    for _ in range(2):
        mn.present_sample(small_brain, np.ones(4), 35, learn=True,
                          learning_rule="post_trace", collect_responses=False)
    assert len(created) == 2
    assert created[0].last_step == 35 and created[1].last_step == 70
    assert created[0].trace.data_ptr() != created[1].trace.data_ptr()
    assert torch.count_nonzero(created[0].trace) > 0


def test_trace_arrival_matches_real_delayed_projection_delivery(small_brain, monkeypatch):
    import src.region as region_module
    proj = small_brain.get_projection("input", "cortex")
    proj.syn_alive[:] = False
    proj.syn_alive[0] = True
    proj.syn_delay[0] = 3
    pre, post = int(proj.syn_pre[0]), int(proj.syn_post[0])
    small_brain.regions["input"].v[pre] = 35.
    rule = PostTraceSTDP(small_brain)
    recorded = []
    integrate = region_module.integrate

    def capture(v, u, current, *args, **kwargs):
        if len(v) == small_brain.regions["cortex"].n_neurons:
            recorded.append(current.clone())
        return integrate(v, u, current, *args, **kwargs)

    monkeypatch.setattr(region_module, "integrate", capture)
    for step in range(1, 5):
        small_brain.step()
        rule.step()
        assert float(rule.trace[0]) == (1. if step == 4 else 0.)
    assert all(float(current[post]) == 0. for current in recorded[:3])
    assert recorded[3][post] > 0


def test_update_changes_only_feedforward_weights_not_other_state_or_rng(small_brain):
    small_brain.regions["cortex"].v[0] = 35.
    rule = PostTraceSTDP(small_brain)
    small_brain.step()
    before = simulation_digest(small_brain)
    weights = rule.projection.syn_weight.clone()
    assert rule.step() > 0
    assert not torch.equal(weights, rule.projection.syn_weight)
    rule.projection.syn_weight.copy_(weights)
    assert simulation_digest(small_brain) == before


@pytest.mark.parametrize("module", ["mnist_boundary_audit", "mnist_plasticity_audit"])
def test_pair_specific_audits_reject_trace_studies_before_any_simulation(small_brain, tmp_path, monkeypatch, module):
    audit_module = importlib.import_module("examples." + module)
    before = simulation_digest(small_brain)
    with pytest.raises(ValueError, match="only the pair"):
        audit_module.audit(small_brain, np.ones((1, 4)), train_steps=2, rest_steps=0,
                           learning_rule="post_trace")
    assert simulation_digest(small_brain) == before
    study = tmp_path / "study"
    study.mkdir()
    summary = {"complete": True, "signature": {"seeds": [101],
               "config": asdict(LearningConfig(learning_rule="post_trace"))}}
    (study / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", [module, "--study", str(study), "--output", str(output)])
    monkeypatch.setattr(mn, "fetch_openml", lambda *args, **kwargs: pytest.fail("dataset accessed"))
    with pytest.raises(SystemExit) as error:
        audit_module.main()
    assert error.value.code == 2
    assert not output.exists()
