import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from examples import mnist_paper_theta_swap as swap
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork


def small_network():
    return ReferenceNetwork(replace(ReferenceConfig(dynamics_version=1, integration_substeps=1), n_input=8, n_exc=2, incoming_sum=4.,
                                    presentation_ms=20., rest_ms=5., input_delay_max=2.), seed=71)


def response(spikes, volts=None):
    spikes = np.asarray(spikes)
    return dict(spikes=spikes, voltages=np.asarray(volts, dtype=float) if volts is not None else spikes.astype(float),
                attempts=np.ones(len(spikes), dtype=int))


def test_hybrid_uses_only_donor_theta_and_does_not_alias_either_checkpoint():
    a = small_network()
    b = swap.training.training_copy(a, 4)
    b.theta += np.array([1., 2.])
    b.weights[:] = .8
    before = a.state_digest(), b.state_digest()
    mixed = swap.hybrid(a, b)
    np.testing.assert_array_equal(mixed.theta, b.theta)
    for field in ('weights', 'delays', 'v_e', 'v_i', 'last_pre', 'last_post'):
        np.testing.assert_array_equal(getattr(mixed, field), getattr(a, field))
    mixed.theta[:] = 0
    mixed.weights[:] = 0
    assert before == (a.state_digest(), b.state_digest())


@pytest.mark.parametrize('change', ['config', 'delay', 'theta'])
def test_hybrid_rejects_incompatible_donors(change):
    a = small_network()
    b = copy.deepcopy(a)
    if change == 'config':
        b.config = replace(b.config, tau_e=200.)
    elif change == 'delay':
        b.delays[0, 0] += 1
    else:
        b.theta[0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        swap.hybrid(a, b)


def test_factorial_metrics_keep_signed_interaction_and_endpoint_anchors():
    a = response([[2, 2]])
    b = response([[3, 1]])
    c = response([[2, 4]])
    d = response([[6, 4]])
    metrics = swap.factorial_metrics(a, b, c, d)
    assert metrics['weights_only']['relative_spike_l1'] == .5
    assert metrics['theta_only']['relative_spike_l1'] == .5
    assert metrics['joint']['relative_spike_l1'] == 1.5
    assert metrics['interaction']['relative_spike_l1'] == 1.
    assert metrics['weights_after_theta']['relative_spike_l1'] == 1.
    assert metrics['theta_after_weights']['relative_spike_l1'] == 1.5
    assert metrics['joint']['mean_total_spike_change'] == 6.


def test_additive_response_has_zero_interaction():
    a = response([[2, 2]])
    b = response([[3, 1]])
    c = response([[2, 4]])
    d = response([[3, 3]])
    metrics = swap.factorial_metrics(a, b, c, d)
    assert metrics['interaction']['relative_spike_l1'] == 0.
    assert metrics['interaction']['centered_voltage_rmse_mv'] == 0.


def test_fixed_weights_control_has_exact_zero_weight_terms_and_interaction():
    a = response([[2, 2]], [[-63.12345, -62.97654]])
    d = response([[1, 5]], [[-60.78632, -64.33179]])
    metrics = swap.factorial_metrics(a, copy.deepcopy(a), copy.deepcopy(d), d)
    for name in ('weights_only', 'weights_after_theta', 'interaction'):
        assert metrics[name]['relative_spike_l1'] == 0.
        assert metrics[name]['centered_voltage_rmse_mv'] == 0.
    assert metrics['joint'] == metrics['theta_only'] == metrics['theta_after_weights']


def test_silent_baseline_has_explicit_undefined_relative_norm_and_retains_absolute_metric():
    a = response([[0, 0]])
    d = response([[2, 0]])
    metrics = swap.factorial_metrics(a, a, d, d)
    assert metrics['joint']['relative_spike_l1'] is None
    assert metrics['joint']['mean_absolute_spike_difference'] == 2.


def test_voltage_metric_removes_common_neuron_offset():
    result = swap.delta_metrics(np.zeros((1,2)), np.full((1,2),12.), response([[1,1]]))
    assert result['centered_voltage_rmse_mv'] == 0.


def test_fixed_theta_makes_identical_weights_equivalent_on_common_grid():
    initial = small_network()
    models = {f: swap.training.training_copy(initial, f) for f in swap.FACTORS}
    for f, model in models.items():
        model.theta += f
    images, ids = np.full((1,8),255.), [71]
    responses = [swap.training.common_probe(swap.hybrid(models[f], models[2]), images, ids,
        seed=4, base_config=initial.config) for f in swap.FACTORS]
    for item in responses[1:]:
        for field in swap.FIELDS:
            np.testing.assert_array_equal(item[field], responses[0][field])


def test_replay_rejects_a_changed_diagonal_response():
    data = response([[1,2]])
    responses = {swap.key(a,w,t): copy.deepcopy(data) for a in swap.ARMS for w in swap.FACTORS for t in swap.FACTORS}
    baseline = {f'{f}__{a}__{k}':v for a in swap.ARMS for f in swap.FACTORS for k,v in data.items()}
    responses[swap.key(swap.ARMS[1],2,2)]['voltages'][0,0] += 1e-10
    with pytest.raises(AssertionError):
        swap.verify_replays(responses, baseline)


def test_worker_and_summary_verify_factorial_caches_without_fitting(tmp_path, monkeypatch):
    output = tmp_path/'study'
    output.mkdir()
    source = tmp_path/'source'
    source.mkdir()
    initial = small_network()
    models = {a:{f:swap.training.training_copy(initial,f) for f in swap.FACTORS} for a in swap.ARMS}
    for f in swap.FACTORS:
        models[swap.ARMS[1]][f].weights[:] = f / 10.
        for a in swap.ARMS:
            models[a][f].theta[:] = 20. + f
    def fake_probe(n, images, ids, **kwargs):
        w = round(float(n.weights.mean()) * 100)
        t = round(float(n.theta.mean()))
        return response([[w, t]], [[w*t, -w]])
    row = dict(seed=201)
    plan = dict(seeds=[201], rows=[row], source_root=str(source), probe_ids=[17],
                base_dt_ms=.5, probe_factor=4, poisson_seed_offset=100000)
    swap.atomic_json(output/'plan.json',plan)
    np.savez_compressed(source/'images.npz',probe_images=np.full((1,8),255.),probe_ids=[17])
    baseline = {f'{f}__{a}__{k}':v for a in swap.ARMS for f in swap.FACTORS
                for k,v in fake_probe(models[a][f],None,None).items()}
    monkeypatch.setattr(swap,'checked_plan',lambda p:plan)
    monkeypatch.setattr(swap,'load_models',lambda r:models)
    monkeypatch.setattr(swap,'baseline_arrays',lambda r,p:baseline)
    monkeypatch.setattr(swap.training,'common_probe',fake_probe)
    swap.run_seed(output,201)
    swap.summarize(output)
    result=json.loads((output/'summary.json').read_text())
    assert result['complete'] and result['source_unchanged']
    assert result['results'][0]['fixed_theta_control_exact']
    metrics=result['results'][0]['metrics'][swap.ARMS[0]]['2_to_4']
    assert metrics['weights_only']['relative_spike_l1'] == 0.
    assert metrics['joint'] == metrics['theta_only']
    with pytest.raises(FileExistsError):
        swap.run_seed(output,201)
    with pytest.raises(FileExistsError):
        swap.summarize(output)
