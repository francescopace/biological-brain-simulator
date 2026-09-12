"""Trajectory instrumentation must leave policy, dynamics and sources intact."""

import json
import sys

import numpy as np
import pytest

import examples.grid_nav_benchmark as grid
import examples.grid_policy_check as check
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import save_training_checkpoint


def test_trace_reproduces_real_evaluation_without_mutating_source(monkeypatch):
    monkeypatch.setattr(grid, "MAX_STEPS_PER_EPISODE", 3)
    brain, env = grid.build_brain(), grid.GridWorld()
    before = simulation_digest(brain)
    starts = [(0, 0), (2, 2), (0, 0)]
    expected = grid.json_safe(grid.evaluate_policy(brain, env, np.random.default_rng(5),
                                                   start_positions=starts, fast=False))
    traced = check.trace_policy(brain, env, starts, 5)
    assert traced["aggregate"] == expected
    assert traced["episodes"][0] == traced["episodes"][2]
    assert traced["decisions"] == 9
    assert sum(traced["action_counts"].values()) == traced["decisions"]
    assert simulation_digest(brain) == before
    assert "step" not in env.__dict__
    for episode in traced["episodes"]:
        pos = tuple(episode["start"])
        for row in episode["decisions"]:
            assert row["position"] == list(pos)
            pos = env.step(pos, row["action"])[0]
            assert list(pos) == row["next_position"]
            counts = np.asarray(row["counts"])
            action, source = grid.choose_action(counts, np.asarray(row["mean_voltage"]), 0., np.random.default_rng(7))
            assert action == row["action"] and source == row["source"]


def test_trace_reports_wall_moves_and_spike_ties(monkeypatch):
    monkeypatch.setattr(grid, "MAX_STEPS_PER_EPISODE", 2)
    monkeypatch.setattr(grid, "present_state", lambda *args: (np.ones(4, dtype=np.int32), np.zeros(4)))
    monkeypatch.setattr(grid, "reset_between_steps", lambda *args: None)
    result = check.trace_policy(grid.build_brain(), grid.GridWorld(), [(0, 0)], 3)
    assert result["spike_ties"] == result["wall_actions"] == 2
    assert result["action_counts"] == {"up": 2, "down": 0, "left": 0, "right": 0}
    assert result["aggregate"]["mean_success_steps"] is None


@pytest.mark.parametrize("silent", [False, True])
def test_trace_distinguishes_spike_success_from_fallback_success(monkeypatch, silent):
    counts = np.zeros(4, dtype=np.int32) if silent else np.ones(4, dtype=np.int32)
    monkeypatch.setattr(grid, "present_state", lambda *args: (counts, np.zeros(4)))
    monkeypatch.setattr(grid, "reset_between_steps", lambda *args: None)
    result = check.trace_policy(grid.build_brain(), grid.GridWorld(goal=(0, 0)), [(1, 0)], 3)
    assert result["aggregate"]["success_rate"] == 1.
    assert result["successes_without_fallback"] == int(not silent)
    assert result["successes_with_fallback"] == int(silent)


def test_weight_diagnostics_records_the_actual_state_action_mapping():
    brain = grid.build_brain()
    result = check.weight_diagnostics(brain)
    np.testing.assert_allclose(result["state_action_weights"], .05)
    assert result["at_lower_bound"] == result["at_upper_bound"] == 0
    assert len(result["motor_theta"]) == 4


@pytest.mark.parametrize("complete", [False, True])
def test_check_rejects_incomplete_or_incompatible_study(monkeypatch, tmp_path, complete):
    (tmp_path / "summary.json").write_text(json.dumps({"complete": complete, "numpy": "mismatch"}))
    monkeypatch.setattr(sys, "argv", ["check", "--study", str(tmp_path), "--output", str(tmp_path / "out.json")])
    with pytest.raises(ValueError if complete else SystemExit):
        check.main()
    assert not (tmp_path / "out.json").exists()


def test_main_reproduces_selected_metrics_and_records_abba(monkeypatch, tmp_path):
    monkeypatch.setattr(grid, "MAX_STEPS_PER_EPISODE", 1)
    brain, env = grid.build_brain(), grid.GridWorld()
    starts = [(0, 0), (2, 2)]
    expected = grid.json_safe(grid.evaluate_policy(brain, env, np.random.default_rng(1), start_positions=starts))
    study = {"complete": True, "numpy": np.__version__, "torch": check.torch.__version__,
             "source_sha256": {}, "config": {"MAX_STEPS_PER_EPISODE": 1}, "best_episode": 1,
             "evaluation": {"size": 5, "goal": [4, 4], "action_deltas": grid.ACTION_DELTAS.tolist(),
                            "final_starts": starts}, "exhaustive_starts": starts,
             "final_policy": expected, "exhaustive_policy": expected}
    protocol = {key: study[key] for key in ("config", "source_sha256", "evaluation")}
    # Match the JSON-normalized protocol passed by the command.
    protocol = json.loads(json.dumps(protocol))
    save_training_checkpoint(brain, tmp_path / "checkpoints/selected", {
        "protocol": protocol, "selected_by": "checkpoint_starts", "completed_episodes": 1})
    (tmp_path / "summary.json").write_text(json.dumps(study))
    output = tmp_path / "check.json"
    monkeypatch.setattr(sys, "argv", ["check", "--study", str(tmp_path), "--output", str(output)])
    check.main()
    result = json.loads(output.read_text())
    assert result["complete"] and result["source_unchanged"] and result["all_metrics_equal"]
    assert [row["fast"] for row in result["timing"]] == [False, True, True, False]
    assert result["final_policy"] == result["trace"]["aggregate"] == expected
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        check.main()
    assert output.read_bytes() == before
