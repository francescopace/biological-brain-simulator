import copy
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from examples import mnist_paper_timing_check as timing


@pytest.mark.parametrize('dt', timing.GRIDS)
def test_dyadic_refractory_release_matches_integer_step_budget(dt):
    result = timing.refractory_check(dt)
    for population in ('exc', 'inh'):
        assert result[population]['mismatched_intervals'] == 0
        assert all(v == 0 for v in result[population]['excess_steps'])


@pytest.mark.parametrize('dt', [.2, .1, .05])
def test_legacy_decimal_refractory_defect_remains_reproducible(dt):
    result = timing.refractory_check(dt)
    for population in ('exc', 'inh'):
        assert result[population]['mismatched_intervals'] > 0


def test_event_oracle_matches_closed_form_constant_conductance_and_refractory():
    source = timing.make_network(ge_e=(4., 0.))
    source.config = replace(source.config, tau_ge=1e20)
    result = timing.event_trace(source)
    c = source.config
    equilibrium = (c.rest_e + 4*c.reversal_e) / 5
    crossing = -c.tau_e/5 * math.log((c.threshold_e-equilibrium)/(c.rest_e-equilibrium))
    np.testing.assert_allclose(result['exc_spikes_ms'][0], [crossing, 2*crossing+c.refractory_e], rtol=0, atol=1e-9)
    assert result['exc_spikes_ms'][1] == []
    assert result['inh_spikes_ms'] == [[], []]


def test_grid_timestamps_quantize_constant_conductance_crossings_at_interval_end():
    for dt in timing.GRIDS:
        source = timing.make_network(dt, ge_e=(4., 0.))
        source.config = replace(source.config, tau_ge=1e20)
        c = source.config
        equilibrium = (c.rest_e + 4*c.reversal_e) / 5
        crossing = -c.tau_e/5 * math.log((c.threshold_e-equilibrium)/(c.rest_e-equilibrium))
        trace, _ = timing.grid_trace(source)
        first = math.ceil(crossing / dt) * dt
        np.testing.assert_allclose(trace['exc_spikes_ms'][0], [first, 2*first+c.refractory_e], atol=1e-12)


def test_recurrent_events_preserve_pair_symmetry_in_event_oracle():
    source = timing.make_network(ge_e=(80.,80.), recurrent=True)
    result = timing.event_trace(source)
    assert result['exc_spikes_ms'][0] == result['exc_spikes_ms'][1]
    assert result['inh_spikes_ms'][0] == result['inh_spikes_ms'][1]
    assert result['exc_spikes_ms'][0]
    assert result['inh_spikes_ms'][0][0] > result['exc_spikes_ms'][0][0]


def test_recurrent_conductance_delivery_occurs_at_emission_boundary_without_extra_step():
    for dt in timing.GRIDS:
        source = timing.make_network(dt, recurrent=True)
        source.pending_e[0] = True
        trace, result = timing.grid_trace(source, duration=dt)
        np.testing.assert_allclose(result.ge_i, [source.config.exc_to_inh * math.exp(-dt), 0.], atol=1e-13)
        assert source.pending_e[0]
        source.pending_e[:] = False
        source.pending_i[0] = True
        _, result = timing.grid_trace(source, duration=dt)
        np.testing.assert_allclose(result.gi_e, [0., source.config.inh_to_exc * math.exp(-dt/2)], atol=1e-13)


def test_observation_does_not_change_source_or_whole_chunk_trajectory():
    source = timing.make_network(ge_e=(80.,30.), recurrent=True)
    before = source.state_digest()
    trace, stepped = timing.grid_trace(source)
    whole = copy.deepcopy(source)
    exc, inh, _ = whole.advance(np.zeros((round(timing.DURATION_MS/source.config.dt),1),bool))
    assert source.state_digest() == before
    assert stepped.state_digest() == whole.state_digest()
    np.testing.assert_array_equal(exc, [len(v) for v in trace['exc_spikes_ms']])
    np.testing.assert_array_equal(inh, [len(v) for v in trace['inh_spikes_ms']])


def test_event_count_mismatch_does_not_report_misleading_matched_timing_error():
    a = dict(exc_spikes_ms=[[1.], []], inh_spikes_ms=[[], []])
    b = dict(exc_spikes_ms=[[1.1, 2.], []], inh_spikes_ms=[[], []])
    result = timing.compare_events(a, b)
    assert not result['same_per_cell_spike_counts']
    assert result['max_abs_spike_time_error_ms'] is None
    with pytest.raises(ValueError, match='neuron counts'):
        timing.compare_events(a, dict(exc_spikes_ms=[[1.]], inh_spikes_ms=[[]]))


def test_event_oracle_rejects_an_evolved_network():
    source = timing.make_network()
    source.step_count = 1
    with pytest.raises(ValueError, match='untouched'):
        timing.event_trace(source)


def test_legacy_grid_competition_characterizes_a_late_inhibition_extra_spike():
    source = timing.make_network(ge_e=(80.,35.), recurrent=True)
    reference = timing.event_trace(source)
    assert list(map(len,reference['exc_spikes_ms'])) == [1,0]
    for dt, expected in ((.5,[1,1]),(.25,[1,1]),(.125,[1,0]),(.0625,[1,0])):
        observed, _ = timing.grid_trace(timing.make_network(dt, ge_e=(80.,35.), recurrent=True))
        assert list(map(len,observed['exc_spikes_ms'])) == expected
    assert reference['inh_spikes_ms'][0][0] == pytest.approx(.8430013324, abs=1e-9)


def test_legacy_near_threshold_pulse_has_integration_bias_beyond_timestamp_ceiling():
    c = timing.make_network().config
    def rhs(t,y):
        return [(c.rest_e-y[0]+y[1]*(c.reversal_e-y[0]))/c.tau_e, -y[1]/c.tau_ge]
    exact = solve_ivp(rhs,(0,2.5),[c.rest_e,25.],method='DOP853',rtol=1e-12,atol=1e-13).y[0,-1]
    values = []
    for dt in (.5,.25,.125,.0625):
        n = timing.make_network(dt, ge_e=(25.,0.))
        n.config = replace(n.config,threshold_e=100.)  # Observe the unreset voltage.
        n.advance(np.zeros((round(2.5/dt),1),bool))
        values.append(n.v_e[0])
    assert values[0] < c.threshold_e < values[1] < values[2] < values[3] < exact
