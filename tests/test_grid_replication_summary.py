"""Replication summaries reject incomplete, mixed or corrupted studies."""

import hashlib
import json

import pytest

import examples.grid_nav_benchmark as grid
from examples.grid_replication_summary import summarize, verify_current_defaults
from examples.training_checkpoint import save_training_checkpoint


def studies(tmp_path):
    directories = []
    for seed in (42, 43):
        directory = tmp_path / str(seed)
        (directory / "source_snapshot").mkdir(parents=True)
        (directory / "source_snapshot/source.py").write_text("# source\n")
        metrics = {"success_rate": 1., "spike_only_success_rate": 1.,
                   "fallback_assisted_success_rate": 0., "mean_steps": 2.}
        study = {"complete": True, "source_unchanged": True, "config": {"SEED": seed, "EPISODES": 2},
                 "numpy": grid.np.__version__, "torch": grid.torch.__version__,
                 "source_sha256": {"source.py": hashlib.sha256(b"# source\n").hexdigest()},
                 "evaluation": {"size": 2, "goal": [1, 1], "action_deltas": grid.ACTION_DELTAS.tolist(),
                                "final_starts": [[0, 0], [0, 1], [1, 0]]},
                 "exhaustive_starts": [[0, 0], [0, 1], [1, 0]], "episodes": [{"episode": 1}, {"episode": 2}],
                 "best_episode": 1, "initial_policy": metrics, "random_baseline": metrics,
                 "final_policy": metrics, "exhaustive_policy": metrics}
        save_training_checkpoint(grid.build_brain(seed), directory / "checkpoints/selected", {
            "protocol": {key: study[key] for key in ("config", "source_sha256", "evaluation")},
            "completed_episodes": 1, "selected_by": "checkpoint_starts"})
        save_training_checkpoint(grid.build_brain(seed), directory / "checkpoints/initial", {
            "protocol": {key: study[key] for key in ("config", "source_sha256", "evaluation")},
            "completed_episodes": 0})
        (directory / "summary.json").write_text(json.dumps(study))
        directories.append(directory)
    return directories


def test_summary_keeps_network_seeds_and_repeated_world_separate(tmp_path):
    result = summarize(studies(tmp_path)[::-1])
    assert result["complete"]
    assert [row["seed"] for row in result["runs"]] == [42, 43]
    assert [row["exhaustive_spike_only_successes"] for row in result["runs"]] == [3, 3]
    assert result["exhaustive_across_seeds"]["spike_only_success_rate"]["mean"] == 1.


@pytest.mark.parametrize("fault", ["incomplete", "duplicate_seed", "mixed_protocol", "snapshot", "cursor",
                                    "starts", "metrics", "checkpoint"])
def test_summary_rejects_invalid_source_study(tmp_path, fault):
    directories = studies(tmp_path)
    path = directories[1] / "summary.json"
    study = json.loads(path.read_text())
    if fault == "incomplete":
        study["complete"] = False
    elif fault == "duplicate_seed":
        study["config"]["SEED"] = 42
    elif fault == "mixed_protocol":
        study["config"]["EPISODES"] = 3
    elif fault == "snapshot":
        (directories[1] / "source_snapshot/source.py").write_text("changed")
    elif fault == "cursor":
        study["episodes"].pop()
    elif fault == "starts":
        study["exhaustive_starts"].pop()
    elif fault == "metrics":
        study["exhaustive_policy"]["spike_only_success_rate"] = .5
    elif fault == "checkpoint":
        study["best_episode"] = 2
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError):
        summarize(directories)


def test_current_defaults_reproduce_initial_state_and_both_evaluations(monkeypatch, tmp_path):
    directories = studies(tmp_path)
    monkeypatch.setattr(grid, "EPISODES", 2)
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs["start_positions"])
        return {"success_rate": 1., "spike_only_success_rate": 1.,
                "fallback_assisted_success_rate": 0., "mean_steps": 2.}
    monkeypatch.setattr(grid, "evaluate_policy", evaluate)
    result = verify_current_defaults(directories)
    assert [row["seed"] for row in result] == [42, 43]
    assert all(row["saved_metrics_reproduced"] for row in result)
    assert len(calls) == 4
    monkeypatch.setattr(grid, "EPISODES", 3)
    with pytest.raises(ValueError, match="default"):
        verify_current_defaults(directories)


def test_current_defaults_reject_changed_builder(monkeypatch, tmp_path):
    directories = studies(tmp_path)
    monkeypatch.setattr(grid, "EPISODES", 2)
    monkeypatch.setattr(grid, "READOUT_TRANSMISSION_GAIN", 1.)
    with pytest.raises(AssertionError, match="initial state"):
        verify_current_defaults(directories)
