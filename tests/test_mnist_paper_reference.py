import copy
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from examples.mnist_paper_reference import (
    ReferenceConfig, ReferenceNetwork, class_average_predictions,
    conductance_step, frozen_responses, poisson_tape,
)


def small(**kwargs):
    return replace(ReferenceConfig(), n_input=8, n_exc=2, incoming_sum=4., **kwargs)


@pytest.mark.parametrize("kwargs", [{"dt": 0.}, {"theta_tau": -1.}, {"dt": .3},
    {"eta_post": -1.}, {"incoming_sum": 1000.}, {"n_exc": True}, {"rest_ms": float('nan')}])
def test_bad_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ReferenceNetwork(replace(ReferenceConfig(), **kwargs))


def test_passive_voltage_and_conductance_match_analytic_solution():
    v, ge, gi = conductance_step(np.array([-55.]), np.zeros(1), np.zeros(1),
        dt=.5, tau=100., rest=-65., reversal_e=0., reversal_i=-100.)
    np.testing.assert_allclose(v, -65 + 10 * math.exp(-.5 / 100), atol=1e-13)
    assert ge[0] == gi[0] == 0.


def test_conductance_integration_converges_to_independent_dop853():
    def rhs(t, state):
        v, ge, gi = state
        return [(-65-v + ge*(0-v) + gi*(-100-v))/100, -ge, -gi/2]
    reference = solve_ivp(rhs, [0, 8], [-65., 12., 3.], method='DOP853',
                          rtol=1e-12, atol=1e-13).y[:, -1]
    errors = []
    for dt in (.5, .25, .125):
        v, ge, gi = (np.array([x]) for x in (-65., 12., 3.))
        for _ in range(round(8 / dt)):
            v, ge, gi = conductance_step(v, ge, gi, dt=dt, tau=100., rest=-65.,
                                         reversal_e=0., reversal_i=-100.)
        errors.append(abs(v[0] - reference[0]))
        np.testing.assert_allclose([ge[0], gi[0]], reference[1:], atol=1e-10)
    assert errors[1] < errors[0] / 3.5 and errors[2] < errors[1] / 3.5
    assert errors[2] < .004


def test_poisson_draws_reproduce_expected_rate_and_retry_increase():
    config = small(presentation_ms=10000.)
    first = poisson_tape(np.full(8, 255.), config, seed=42)
    np.testing.assert_array_equal(first, poisson_tape(np.full(8, 255.), config, seed=42))
    assert abs(first.mean() - 63.75 * config.dt / 1000) < .002
    higher = poisson_tape(np.full(8, 255.), config, seed=42, attempt=1)
    assert np.all(higher[first]) and higher.sum() > first.sum()
    assert not poisson_tape(np.zeros(8), config, seed=42).any()


@pytest.mark.parametrize("image", [np.zeros(7), np.full(8, -1), np.full(8, 256), np.full(8, np.nan)])
def test_invalid_raw_pixels_rejected(image):
    with pytest.raises(ValueError):
        poisson_tape(image, small(), seed=42)


def test_heterogeneous_delays_are_delivered_across_advance_boundaries():
    network = ReferenceNetwork(small(input_delay_max=2.))
    network.delays[:] = 0
    network.delays[0] = [0, 3]
    tape = np.zeros((2, 8), dtype=bool)
    tape[0, 0] = True
    network.advance(tape)
    assert network.last_pre[0, 0] == 0.
    assert network.last_pre[0, 1] < 0
    assert network.pending_times.tolist() == [3]
    network.advance(np.zeros((2, 8), dtype=bool))
    assert network.last_pre[0, 1] == 1.5
    assert len(network.pending_times) == 0


def test_scheduled_arrivals_match_dense_reference():
    network = ReferenceNetwork(small(input_delay_max=2.))
    tape = np.random.default_rng(7).random((12, 8)) < .3
    original_delays = network.delays.copy()
    synapses, offsets = network._events(tape)
    for time in range(len(tape)):
        expected = []
        for pre, post in np.ndindex(network.weights.shape):
            emission = time - original_delays[pre, post]
            if emission >= 0 and tape[emission, pre]:
                expected.append(pre * network.config.n_exc + post)
        assert sorted(synapses[offsets[time]:offsets[time+1]].tolist()) == expected


def test_triplet_rule_uses_previous_post2_and_arrival_time():
    network = ReferenceNetwork(small())
    network.weights[:] = .5
    network.triplet_pre(np.array([0]), 10., learn=True)
    network.triplet_post(np.array([True, False]), 20., learn=True)
    assert network.weights[0, 0] == .5  # No earlier postsynaptic trace.
    network.triplet_pre(np.array([0]), 25., learn=True)
    expected = .5 - .0001 * math.exp(-5 / 20)
    assert network.weights[0, 0] == pytest.approx(expected)
    network.triplet_post(np.array([True, False]), 30., learn=True)
    expected += .01 * math.exp(-5 / 20) * math.exp(-10 / 40)
    assert network.weights[0, 0] == pytest.approx(expected)
    assert network.weights[1, 0] == .5  # No presynaptic trace at this synapse.


def test_triplet_trace_resets_instead_of_accumulating_and_clips():
    network = ReferenceNetwork(small())
    network.last_post[:] = 0
    network.triplet_pre(np.array([0]), 1., learn=False)
    network.triplet_pre(np.array([0]), 2., learn=False)
    network.weights[0, 0] = .9999
    network.triplet_post(np.array([True, False]), 3., learn=True)
    assert network.weights[0, 0] == 1.
    network.weights[0, 0] = 0.
    network.triplet_pre(np.array([0]), 4., learn=True)
    assert network.weights[0, 0] == 0.


def test_learning_off_retains_weights_and_frozen_theta():
    network = ReferenceNetwork(small())
    weights, theta = network.weights.copy(), network.theta.copy()
    network.advance(poisson_tape(np.full(8, 255.), small(), seed=8))
    np.testing.assert_array_equal(network.weights, weights)
    np.testing.assert_array_equal(network.theta, theta)


def test_theta_decays_on_reference_timescale_only_when_adapting():
    network = ReferenceNetwork(small())
    network.advance(np.zeros((100, 8), dtype=bool), adapt=True)
    np.testing.assert_allclose(network.theta, 20 * math.exp(-50 / 1e7), rtol=1e-13)


def test_inhibitory_partner_is_recruited_and_excludes_its_paired_exc_cell():
    network = ReferenceNetwork(small())
    network.reset_transients()
    network.pending_e[0] = True
    _, counts_i, _ = network.advance(np.zeros((12, 8), dtype=bool))
    assert counts_i[0] > 0 and counts_i[1] == 0
    assert network.gi_e[0] == 0 and network.gi_e[1] > 0


def test_refractory_prevents_immediate_refiring():
    network = ReferenceNetwork(small())
    network.reset_transients()
    network.ge_e[:] = 10000.
    tape = np.zeros((1, 8), dtype=bool)
    first, _, _ = network.advance(tape)
    assert first.tolist() == [1, 1]
    for _ in range(10):
        network.ge_e[:] = 10000.
        counts, _, _ = network.advance(tape)
        assert counts.sum() == 0


def test_normalization_matches_budget_and_rejects_zero_column():
    network = ReferenceNetwork(small())
    network.normalize()
    np.testing.assert_allclose(network.weights.sum(axis=0), 4.)
    assert np.all((network.weights >= 0) & (network.weights <= 1))
    network.weights[:, 1] = 0
    with pytest.raises(ValueError):
        network.normalize()


def test_full_state_roundtrip_resumes_pending_delays_exactly(tmp_path):
    network = ReferenceNetwork(small(input_delay_max=2.))
    network.delays[0] = 3
    first = np.zeros((2, 8), dtype=bool)
    first[1, 0] = True
    network.advance(first, learn=True, adapt=True)
    network.save(tmp_path / 'state', {'protocol': 1})
    loaded = ReferenceNetwork.load(tmp_path / 'state', {'protocol': 1})
    assert loaded.state_digest() == network.state_digest()
    rest = np.random.default_rng(4).random((20, 8)) < .2
    for a, b in zip(network.advance(rest, learn=True, adapt=True), loaded.advance(rest, learn=True, adapt=True)):
        np.testing.assert_array_equal(a, b)
    assert loaded.state_digest() == network.state_digest()
    with pytest.raises(FileExistsError):
        network.save(tmp_path / 'state', {})
    with pytest.raises(ValueError, match='protocol'):
        ReferenceNetwork.load(tmp_path / 'state', {})
    with (tmp_path / 'state' / 'arrays.npz').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        ReferenceNetwork.load(tmp_path / 'state', {'protocol': 1})


def test_class_average_excludes_silent_neurons_and_abstains():
    train = np.array([[3, 0, 0], [2, 0, 0], [0, 4, 0], [0, 2, 0]])
    valid = np.array([[1, 0, 100], [0, 2, 0], [0, 0, 0]])
    predictions, assignments, _ = class_average_predictions(train, [0, 0, 1, 1], valid, classes=(0, 1))
    assert assignments.tolist() == [0, 1, -1]
    assert predictions.tolist() == [0, 1, -1]
    with pytest.raises(ValueError, match='Each class'):
        class_average_predictions(train[:2], [0, 0], valid, classes=(0, 1))


def test_frozen_inference_is_order_invariant_and_source_unchanged():
    config = small(min_spikes=1)
    network = ReferenceNetwork(config)
    network.weights[:] = .9
    images, rows = np.array([[255.] * 8, [220.] * 8]), [100, 200]
    before = network.state_digest()
    first = frozen_responses(network, images, rows, seed=42)
    reverse = frozen_responses(network, images[::-1], rows[::-1], seed=42)
    for a, b in zip(first, reverse):
        np.testing.assert_array_equal(a, b[::-1])
    assert network.state_digest() == before


def test_retry_cap_does_not_silently_drop_blank_images():
    config = small(presentation_ms=1., rest_ms=0., max_attempts=2)
    with pytest.raises(RuntimeError, match='retry budget'):
        frozen_responses(ReferenceNetwork(config), np.zeros((1, 8)), [10], seed=42)
