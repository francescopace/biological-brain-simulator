import copy
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from examples import mnist_paper_dynamics_check as check
from examples import mnist_paper_timing_check as timing
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork, poisson_tape


def small(**kwargs):
    return replace(ReferenceConfig(), n_input=8, n_exc=2, incoming_sum=4.,
                   presentation_ms=40., rest_ms=10., input_delay_max=2., **kwargs)


@pytest.mark.parametrize("dt", timing.REFRACTORY_GRIDS)
@pytest.mark.parametrize("substeps", [1, 8])
def test_v2_refractory_releases_on_exact_integer_boundary(dt, substeps):
    result = check.refractory_check(dt, substeps)
    for population in result.values():
        assert len(population["observed_interval_ticks"]) > 1
        assert population["mismatched_intervals"] == 0


@pytest.mark.parametrize("kwargs", [dict(dynamics_version=3), dict(dynamics_version=True),
    dict(integration_substeps=0), dict(integration_substeps=-1), dict(integration_substeps=1.5),
    dict(integration_substeps=True), dict(dynamics_version=1, integration_substeps=8)])
def test_invalid_dynamics_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        ReferenceNetwork(small(**kwargs))


def test_defaults_are_versioned_and_input_draws_and_physical_delays_are_unchanged():
    corrected = ReferenceNetwork(small(), seed=71)
    legacy = ReferenceNetwork(replace(corrected.config, dynamics_version=1, integration_substeps=1), seed=71)
    assert corrected.config.dynamics_version == 2
    assert corrected.config.dt / corrected.config.integration_substeps == .0625
    for name in ("weights", "delays", "theta", "v_e", "v_i"):
        np.testing.assert_array_equal(getattr(corrected, name), getattr(legacy, name))
    for attempt in (0, 2):
        image = np.full(8, 255.)
        np.testing.assert_array_equal(poisson_tape(image, corrected.config, seed=73, attempt=attempt),
                                      poisson_tape(image, legacy.config, seed=73, attempt=attempt))
    assert corrected.state_digest() != legacy.state_digest()


@pytest.mark.parametrize("learn,adapt", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dt", [.5, .1])
def test_substeps_match_explicit_fine_grid_including_learning_delays_and_recurrence(learn, adapt, dt):
    coarse = ReferenceNetwork(small(dt=dt), seed=7)
    coarse.reset_transients()
    coarse.ge_e[:] = [80., 35.]
    before = coarse.weights.copy()
    fine = copy.deepcopy(coarse)
    factor = coarse.config.integration_substeps
    fine.config = replace(coarse.config, dt=coarse.config.dt / factor, integration_substeps=1)
    fine.delays *= factor
    tape = np.random.default_rng(51).random((80, 8)) < .4
    for chunk in (tape[:3], tape[3:9], tape[9:]):
        lifted = np.zeros((len(chunk) * factor, 8), bool)
        lifted[::factor] = chunk
        for left, right in zip(coarse.advance(chunk, learn=learn, adapt=adapt),
                               fine.advance(lifted, learn=learn, adapt=adapt)):
            np.testing.assert_array_equal(left, right)
        for name, value in vars(coarse).items():
            if isinstance(value, np.ndarray):
                expected = value * factor if name in ("delays", "pending_times") else value
                np.testing.assert_array_equal(getattr(fine, name), expected)
        assert fine.step_count == coarse.step_count * factor
    assert np.any(coarse.last_inh > 0)  # Recurrence was exercised.
    assert bool(np.any(coarse.weights != before)) == learn
    assert bool(np.any(coarse.theta != 20.)) == adapt


def test_corrected_competition_suppresses_extra_spike_and_reduces_inhibitory_delay():
    args = dict(ge_e=(80., 35.), recurrent=True)
    old = timing.make_network(**args)
    new = timing.make_network(**args, dynamics_version=2, integration_substeps=8)
    oracle = timing.event_trace(new)
    legacy, _ = check.spike_trace(old)
    corrected, state = check.spike_trace(new)
    assert list(map(len, legacy["exc_spikes_ms"])) == [1, 1]
    assert list(map(len, corrected["exc_spikes_ms"])) == [1, 0]
    comparison = timing.compare_events(oracle, corrected)
    assert comparison["same_per_cell_spike_counts"]
    assert comparison["max_abs_spike_time_error_ms"] < .1
    assert corrected["inh_spikes_ms"][0][0] == .9375
    # Recording one outer interval at a time must not alter state.
    whole = copy.deepcopy(new)
    whole.advance(np.zeros((40, 1), bool))
    assert state.state_digest() == whole.state_digest()


def test_near_threshold_voltage_bias_is_reduced_by_full_substep_integration():
    args = dict(ge_e=(25., 0.))
    old = timing.make_network(**args)
    new = timing.make_network(**args, dynamics_version=2, integration_substeps=8)
    oracle = timing.event_trace(new)["exc_spikes_ms"][0][0]
    old_events, _ = check.spike_trace(old)
    new_events, _ = check.spike_trace(new)
    assert old_events["exc_spikes_ms"][0] == [3.]
    assert new_events["exc_spikes_ms"][0] == [2.4375]
    assert 0 < new_events["exc_spikes_ms"][0][0] - oracle < .0625
    for network in (old, new):
        network.config = replace(network.config, threshold_e=100.)
        network.advance(np.zeros((5, 1), bool))
    exact_voltage = -51.893862500912
    assert abs(new.v_e[0] - exact_voltage) < .002
    assert abs(new.v_e[0] - exact_voltage) < abs(old.v_e[0] - exact_voltage) / 50


def test_multiple_internal_spikes_are_counted_and_recorded_in_one_outer_step():
    source = timing.make_network(ge_e=(10000., 0.), dynamics_version=2, integration_substeps=8)
    source.config = replace(source.config, refractory_e=0.)
    events, observed = check.spike_trace(source, duration=.5, adapt=True)
    whole = copy.deepcopy(source)
    exc, _, _ = whole.advance(np.zeros((1, 1), bool), adapt=True)
    assert exc[0] == 8 == len(events["exc_spikes_ms"][0])
    assert observed.state_digest() == whole.state_digest()
    assert whole.last_post[0] == .5


def test_v2_checkpoint_and_chunked_resume_preserve_release_ticks_and_pending_events(tmp_path):
    source = ReferenceNetwork(small(), seed=7)
    source.reset_transients()
    source.ge_e[:] = [80., 35.]
    source.delays[0] = 3
    tape = np.zeros((1, 8), bool)
    tape[0, 0] = True
    source.advance(tape, learn=True, adapt=True)
    assert source.pending_times.size and np.any(source.release_e > source.step_count * 8)
    source.pending_i[0] = True
    source.save(tmp_path / "v2", {"protocol": "corrected"})
    envelope = json.loads((tmp_path / "v2" / "metadata.json").read_text())
    assert envelope["version"] == envelope["config"]["dynamics_version"] == 2
    restored = ReferenceNetwork.load(tmp_path / "v2", {"protocol": "corrected"})
    assert restored.state_digest() == source.state_digest()
    tape = np.random.default_rng(19).random((80, 8)) < .4
    expected = source.advance(tape, learn=True, adapt=True)
    parts = [restored.advance(part, learn=True, adapt=True) for part in (tape[:7], tape[7:31], tape[31:])]
    for i in (0, 1):
        np.testing.assert_array_equal(expected[i], sum(part[i] for part in parts))
    np.testing.assert_allclose(expected[2], sum(p[2] * n for p, n in zip(parts, [7, 24, 49])) / 80, atol=1e-12)
    assert restored.state_digest() == source.state_digest()
    restored.reset_transients()
    assert not restored.release_e.any() and not restored.release_i.any()
    envelope["version"] = 1
    (tmp_path / "v2" / "metadata.json").write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="version mismatch"):
        ReferenceNetwork.load(tmp_path / "v2", {"protocol": "corrected"})


def test_unversioned_config_and_v1_checkpoint_replay_original_state_hashes(tmp_path):
    # Golden hashes from the unmodified 20260913 timing-audit source snapshot,
    # including decimal-grid refractory behavior, delayed arrivals and STDP.
    config = small(dt=.1, dynamics_version=1, integration_substeps=1)
    old_config = asdict(config)
    old_config.pop("dynamics_version")
    old_config.pop("integration_substeps")
    assert ReferenceConfig.from_dict(old_config) == config
    with pytest.raises(ValueError, match="explicit dynamics version"):
        ReferenceConfig.from_dict(dict(old_config, integration_substeps=8))
    source = ReferenceNetwork(config, seed=17)
    source.normalize()
    assert source.state_digest() == "e2706840d948ef5089d58ee4fee8ddc2a599b7f7d14a01e2cd2a94d44aac553d"
    source.reset_transients()
    source.ge_e[:] = [80., 35.]
    tape = np.zeros((400, 8), bool)
    tape[::7, 0] = True
    tape[3::11, 1:] = True
    source.advance(tape[:13], learn=True, adapt=True)
    golden = "b2001c0cd5f3d23ce8f80ef67be799bf70cf8227cbe7619e85f3c4b8309c5728"
    assert source.state_digest() == golden
    # Build the historical envelope independently of the new save path.
    np.savez_compressed(tmp_path / "arrays.npz", **{k: v for k, v in vars(source).items() if isinstance(v, np.ndarray)})
    envelope = dict(version=1, config=old_config, step_count=13, state_sha256=golden, metadata={"old": True},
                    arrays_sha256=hashlib.sha256((tmp_path / "arrays.npz").read_bytes()).hexdigest())
    (tmp_path / "metadata.json").write_text(json.dumps(envelope))
    loaded = ReferenceNetwork.load(tmp_path, {"old": True})
    assert loaded.config.dynamics_version == 1 and not hasattr(loaded, "release_e")
    loaded.advance(tape[13:], learn=True, adapt=True)
    assert loaded.state_digest() == "66d2585eace67386e212b548a5549ef0ebf6b001ae1ea6eae483a5ac28cc50d7"
    loaded.save(tmp_path / "resaved", {"old": True})
    saved = json.loads((tmp_path / "resaved" / "metadata.json").read_text())
    assert saved["version"] == 1 and saved["config"] == old_config
