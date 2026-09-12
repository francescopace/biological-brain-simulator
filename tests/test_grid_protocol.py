"""Independent, paired Grid evaluation must keep within-episode neural dynamics."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import examples.grid_nav_benchmark as grid
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import directory_digest, load_training_checkpoint


def metrics(*, steps=2, silent=0, success=False):
    return {"success": success, "steps": steps, "final_distance": 0 if success else 2,
            "explore_steps": 0, "silent_steps": silent, "motor_spikes": 0,
            "eligibility_mean": 0., "eligibility_positive_fraction": 0.,
            "mean_abs_weight_change": 0., "saturated_fraction": 0.}


def test_independent_rollout_cache_matches_slow_and_preserves_source(monkeypatch):
    monkeypatch.setattr(grid, "MAX_STEPS_PER_EPISODE", 3)
    brain, env = grid.build_brain(), grid.GridWorld()
    proj = brain.projections[0]
    for state in range(25):
        action = 3 if state // 5 == 4 else 1
        selected = (proj.syn_pre[:100] == state) & (proj.syn_post[:100] == action)
        proj.syn_weight[:100][selected] = 10.
    before = simulation_digest(brain)
    positions = [(0, 0), (4, 3), (2, 2), (0, 0)]
    original = grid.run_episode
    calls = []
    def capture(model, env, rng, *args, **kwargs):
        assert all(torch.all(r.v[:r.n_neurons] == -65.) for r in model.regions.values())
        assert not model.reward_stdp.enabled and not model.homeostasis.theta_enabled
        assert all(not p.plasticity_enabled for p in model.projections)
        calls.append(kwargs["start_pos"])
        return original(model, env, rng, *args, **kwargs)
    monkeypatch.setattr(grid, "run_episode", capture)
    slow = grid.evaluate_policy(brain, env, np.random.default_rng(8), start_positions=positions, fast=False)
    assert calls == positions
    calls.clear()
    fast = grid.evaluate_policy(brain, env, np.random.default_rng(8), start_positions=positions)
    assert calls == list(dict.fromkeys(positions))
    reverse = grid.evaluate_policy(brain, env, np.random.default_rng(8), start_positions=positions[::-1])
    assert slow == fast == reverse
    assert simulation_digest(brain) == before


def test_start_schedule_and_caller_rng_do_not_depend_on_rollout_length(monkeypatch):
    brain, env = grid.build_brain(), grid.GridWorld()
    records, rng_states = [], []
    for duration in (1, 20):
        seen = []
        def episode(model, env, rng, *args, **kwargs):
            seen.append(kwargs["start_pos"])
            rng.random(duration)
            return metrics(steps=duration)
        monkeypatch.setattr(grid, "run_episode", episode)
        rng = np.random.default_rng(19)
        grid.evaluate_policy(brain, env, rng, 10, fast=False)
        records.append(seen)
        rng_states.append(rng.bit_generator.state)
    assert records[0] == records[1]
    assert rng_states[0] == rng_states[1]


def test_silence_denominator_counts_actual_steps(monkeypatch):
    monkeypatch.setattr(grid, "run_episode", lambda *args, **kwargs: metrics(steps=2, silent=2, success=True))
    result = grid.evaluate_policy(grid.build_brain(), grid.GridWorld(), np.random.default_rng(1), 3)
    assert result["silent_step_rate"] == 1.
    assert result["mean_steps"] == 2.


def test_exploration_does_not_hide_a_silent_motor_response(monkeypatch):
    monkeypatch.setattr(grid, "present_state", lambda *args: (np.zeros(4, dtype=np.int32), np.zeros(4)))
    monkeypatch.setattr(grid, "reset_between_steps", lambda *args: None)
    env = SimpleNamespace(reset=lambda rng: (0, 0), step=lambda *args: ((4, 4), True, True, 8, 0))
    result = grid.run_episode(object(), env, np.random.default_rng(8), 1, learn=False, epsilon_override=1.)
    assert result["explore_steps"] == result["silent_steps"] == result["steps"] == 1


@pytest.mark.parametrize("positions,count", [([], None), ([(4, 4)], None), ([(-1, 0)], None),
                                            ([(0, 5)], None), ([(.5, 1)], None), ([(0, 0)], 2)])
def test_invalid_evaluation_starts_are_rejected(positions, count):
    with pytest.raises(ValueError):
        grid.evaluation_starts(grid.GridWorld(), np.random.default_rng(1), count, positions)


def test_runtime_presentation_and_rest_lengths_are_used(monkeypatch):
    brain = grid.build_brain()
    monkeypatch.setattr(grid, "STATE_PRESENT_STEPS", 2)
    monkeypatch.setattr(grid, "INTER_STEP_REST", 3)
    grid.present_state(brain, grid.position_encode((0, 0)))
    grid.reset_between_steps(brain)
    assert brain.step_count == 5


def old_present(brain, x, n_steps=None):
    n_steps = grid.STATE_PRESENT_STEPS if n_steps is None else n_steps
    motor = brain.regions["motor"]
    before = motor.total_spikes[:4].clone()
    voltage = torch.zeros(4, dtype=torch.float32, device=grid.DEVICE)
    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.inject_current("motor", list(range(4)), grid.MOTOR_BASELINE_CURRENT)
        brain.step()
        voltage += motor.v[:4]
    return (motor.total_spikes[:4] - before).cpu().numpy(), (voltage / n_steps).cpu().numpy()


def old_mark(brain, x, action):
    for _ in range(grid.ACTION_MARK_STEPS):
        brain.stimulate("input", x)
        brain.inject_current("motor", np.arange(4), grid.MOTOR_BASELINE_CURRENT)
        brain.inject_current("motor", [action], grid.ACTION_MARK_CURRENT)
        brain.step()


@pytest.mark.parametrize("gain,tonic", [(1., 4.), (512., 0.)])
def test_training_conversion_shortcuts_preserve_weights_noise_and_actions(monkeypatch, gain, tonic):
    monkeypatch.setattr(grid, "MAX_STEPS_PER_EPISODE", 3)
    monkeypatch.setattr(grid, "READOUT_TRANSMISSION_GAIN", gain)
    monkeypatch.setattr(grid, "MOTOR_BASELINE_CURRENT", tonic)
    original = grid.build_brain()
    reference = copy.deepcopy(original)
    baselines = [np.zeros(25), np.zeros(25)]
    rngs = [np.random.default_rng(5), np.random.default_rng(5)]
    for episode in range(2):
        with patch.object(grid, "present_state", old_present), patch.object(grid, "mark_executed_action", old_mark):
            left = grid.run_episode(reference, grid.GridWorld(), rngs[0], episode, reward_baseline=baselines[0])
        right = grid.run_episode(original, grid.GridWorld(), rngs[1], episode, reward_baseline=baselines[1])
        assert left == right
        assert simulation_digest(reference) == simulation_digest(original)
        np.testing.assert_array_equal(baselines[0], baselines[1])
        assert rngs[0].bit_generator.state == rngs[1].bit_generator.state


def test_main_records_paired_starts_and_the_selected_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(grid, "EPISODES", 2)
    monkeypatch.setattr(grid, "CHECKPOINT_INTERVAL", 1)
    monkeypatch.setattr(grid, "EVAL_EPISODES", 3)
    monkeypatch.setattr(grid, "CHECKPOINT_EVAL_EPISODES", 2)
    def episode(model, *args, **kwargs):
        model.regions["motor"].theta[0] += 1.
        return metrics()
    starts = []
    def evaluate(model, env, rng, *args, **kwargs):
        starts.append(kwargs["start_positions"])
        return {"success_rate": 1. if model.regions["motor"].theta[0] == 1. else 0.,
                "spike_only_success_rate": 1. if model.regions["motor"].theta[0] == 1. else 0.,
                "fallback_assisted_success_rate": 0.,
                "mean_steps": 2., "mean_success_steps": float("inf"), "mean_final_distance": 2.,
                "silent_step_rate": 0., "mean_motor_spikes": 0.}
    baseline_starts = []
    monkeypatch.setattr(grid, "run_episode", episode)
    monkeypatch.setattr(grid, "evaluate_policy", evaluate)
    monkeypatch.setattr(grid, "policy_action_margin", lambda *args: {"mean": 0., "minimum": 0.})
    def random_baseline(env, rng, **kwargs):
        baseline_starts.append(kwargs["start_positions"])
        return {"success_rate": 0., "mean_steps": 20., "mean_success_steps": float("inf"), "mean_final_distance": 4.}
    monkeypatch.setattr(grid, "random_policy_baseline", random_baseline)
    destination = tmp_path / "recorded"
    result = grid.main(output=destination)
    assert result["complete"] and result["source_unchanged"] and result["best_episode"] == 1
    assert baseline_starts[0] == starts[0] == starts[-2]
    assert starts[1] == starts[2] and len(starts[-1]) == 24
    assert result["random_baseline"]["mean_success_steps"] is None
    checkpoint = destination / "checkpoints/selected"
    progress = json.loads((checkpoint / "progress.json").read_text())["progress"]
    selected, _ = load_training_checkpoint(checkpoint, progress["protocol"])
    assert float(selected.regions["motor"].theta[0]) == 1.
    assert (destination / "checkpoints/episode_000002/progress.json").exists()
    before = directory_digest(destination)
    with pytest.raises(FileExistsError):
        grid.main(output=destination)
    assert directory_digest(destination) == before
