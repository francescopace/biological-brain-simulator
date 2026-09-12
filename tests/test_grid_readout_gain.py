"""Readout gain changes synaptic transmission, not weights, RNG or action credit."""

import hashlib
import json

import numpy as np
import pytest
import torch

import examples.grid_nav_benchmark as grid
from examples.mnist_optimization_check import simulation_digest


@pytest.mark.parametrize("gain", [0., 1., 16., 128., 512.])
def test_builder_gain_changes_only_transmission_modulation(gain):
    original = grid.build_brain(transmission_gain=1.)
    model = grid.build_brain(transmission_gain=gain)
    before = simulation_digest(original)
    projection = model.get_projection("input", "motor")
    source = original.get_projection("input", "motor")
    ns = source.n_synapses
    torch.testing.assert_close(projection.syn_modulation[:ns], source.syn_modulation[:ns] * gain)
    projection.syn_modulation.copy_(source.syn_modulation)
    assert simulation_digest(model) == before
    assert simulation_digest(original) == before


@pytest.mark.parametrize("gain", [-1., float("nan"), float("inf")])
def test_invalid_gain_fails_before_allocating_a_brain(monkeypatch, gain):
    monkeypatch.setattr(grid, "Brain", lambda **kwargs: pytest.fail("Brain must not be constructed"))
    with pytest.raises(ValueError, match="gain"):
        grid.build_brain(transmission_gain=gain)


def test_calibrated_defaults_are_explicit_and_keep_the_legacy_gain_available():
    assert grid.READOUT_TRANSMISSION_GAIN == 512.
    assert grid.MOTOR_BASELINE_CURRENT == 0.
    assert grid.SELECTION_METRIC == "spike_only_success_rate"
    assert simulation_digest(grid.build_brain()) == simulation_digest(grid.build_brain(transmission_gain=512.))


def test_spike_only_evaluation_requires_no_fallback_in_the_whole_path(monkeypatch):
    rows = iter([
        {"success": True, "steps": 1, "silent_steps": 0, "final_distance": 0, "motor_spikes": 1},
        {"success": True, "steps": 2, "silent_steps": 1, "final_distance": 0, "motor_spikes": 1},
        {"success": False, "steps": 2, "silent_steps": 0, "final_distance": 1, "motor_spikes": 2},
    ])
    monkeypatch.setattr(grid, "run_episode", lambda *args, **kwargs: next(rows))
    result = grid.evaluate_policy(grid.build_brain(), grid.GridWorld(), np.random.default_rng(7),
                                  start_positions=[(0, 0), (1, 0), (2, 0)])
    assert result["success_rate"] == 2 / 3
    assert result["spike_only_success_rate"] == result["fallback_assisted_success_rate"] == 1 / 3
    assert result["silent_step_rate"] == 1 / 5


@pytest.mark.parametrize("metric,best", [("success_rate", 1), ("spike_only_success_rate", 2)])
def test_selection_metric_and_source_snapshot_are_recorded(monkeypatch, tmp_path, metric, best):
    for name, value in {"EPISODES": 2, "CHECKPOINT_INTERVAL": 1, "EVAL_EPISODES": 3,
                        "CHECKPOINT_EVAL_EPISODES": 2, "SELECTION_METRIC": metric}.items():
        monkeypatch.setattr(grid, name, value)
    def episode(model, *args, **kwargs):
        model.regions["motor"].theta[0] += 1.
        return {"success": True, "steps": 1, "silent_steps": 0, "final_distance": 0, "motor_spikes": 1,
                "eligibility_mean": 0., "eligibility_positive_fraction": 0.,
                "mean_abs_weight_change": 0., "saturated_fraction": 0.}
    def evaluate(model, *args, **kwargs):
        second = model.regions["motor"].theta[0] == 2.
        return {"success_rate": .5 if second else 1., "spike_only_success_rate": .5 if second else 0.,
                "fallback_assisted_success_rate": 0. if second else 1., "mean_steps": 2.,
                "mean_success_steps": 2., "mean_final_distance": 0., "silent_step_rate": .5,
                "mean_motor_spikes": 1.}
    monkeypatch.setattr(grid, "run_episode", episode)
    monkeypatch.setattr(grid, "evaluate_policy", evaluate)
    monkeypatch.setattr(grid, "policy_action_margin", lambda *args: {"mean": 0., "minimum": 0.})
    output = tmp_path / "study"
    result = grid.main(output=output)
    assert result["best_episode"] == best
    assert result["config"]["SELECTION_METRIC"] == metric
    assert result["complete"] and result["source_unchanged"]
    for name, digest in result["source_sha256"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    progress = json.loads((output / "checkpoints/selected/progress.json").read_text())["progress"]
    assert progress["completed_episodes"] == best
    assert progress["protocol"]["config"]["SELECTION_METRIC"] == metric


@pytest.mark.parametrize("name,value", [("READOUT_TRANSMISSION_GAIN", -1.), ("MOTOR_BASELINE_CURRENT", float("nan")),
    ("EPISODES", 0), ("CHECKPOINT_INTERVAL", 0), ("EVAL_EPISODES", 0), ("SELECTION_METRIC", "invalid")])
def test_invalid_protocol_fails_without_creating_output(monkeypatch, tmp_path, name, value):
    monkeypatch.setattr(grid, name, value)
    with pytest.raises(ValueError):
        grid.main(output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
