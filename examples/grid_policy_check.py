"""Reproduce a saved Grid policy, inspect trajectories, and time exact reuse.

All timing arms use the recorded final starts and the same independent-episode
protocol. The trace runs all non-goal starts once, retaining within-path neural
history. No action is replaced by an environment-derived or weight-based oracle.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

import examples.grid_nav_benchmark as grid
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint
from src.persistence import save_brain


def trace_policy(brain, env, starts, seed):
    """Observe the real rollout without extra RNG draws or neural operations."""
    starts = grid.evaluation_starts(env, np.random.default_rng(seed), start_positions=starts)
    model = grid.inference_brain(brain)
    rng = np.random.default_rng(seed)
    original_choose, original_step = grid.choose_action, env.step
    episodes = []
    decisions = []

    def choose(counts, voltage, epsilon, rollout_rng):
        action, source = original_choose(counts, voltage, epsilon, rollout_rng)
        decisions.append({"counts": counts.tolist(), "mean_voltage": voltage.tolist(),
                          "action": action, "source": source,
                          "spike_tie": bool(counts.max() > 0 and np.sum(counts == counts.max()) > 1)})
        return action, source

    def step(pos, action):
        outcome = original_step(pos, action)
        nxt, moved, reached, before, after = outcome
        decisions[-1].update(position=list(pos), next_position=list(nxt),
                             moved=bool(moved), reached_goal=bool(reached),
                             distance_change=int(after - before))
        return outcome

    with patch.object(grid, "choose_action", choose), patch.object(env, "step", step):
        for start in starts:
            decisions = []
            grid.reset_independent_state(model)
            metrics = grid.run_episode(model, env, rng, grid.EPISODES, learn=False,
                                       epsilon_override=0., start_pos=start)
            if len(decisions) != metrics["steps"]:
                raise AssertionError("Trace did not record every action")
            episodes.append({"start": list(start), "metrics": metrics, "decisions": decisions})
    rows = [row for episode in episodes for row in episode["decisions"]]
    return {"episodes": episodes, "aggregate": aggregate_metrics(episodes),
            "successes_without_fallback": sum(bool(episode["metrics"]["success"])
                and episode["metrics"]["silent_steps"] == 0 for episode in episodes),
            "successes_with_fallback": sum(bool(episode["metrics"]["success"])
                and episode["metrics"]["silent_steps"] > 0 for episode in episodes),
            "decisions": len(rows), "spike_ties": sum(row["spike_tie"] for row in rows),
            "wall_actions": sum(not row["moved"] for row in rows),
            "distance_increasing_actions": sum(row["distance_change"] > 0 for row in rows),
            "action_counts": {name: sum(row["action"] == action for row in rows)
                              for action, name in enumerate(grid.ACTION_NAMES)}}


def aggregate_metrics(episodes):
    metrics = [episode["metrics"] for episode in episodes]
    successful = [item["steps"] for item in metrics if item["success"]]
    return grid.json_safe({
        "success_rate": sum(item["success"] for item in metrics) / len(metrics),
        "spike_only_success_rate": sum(item["success"] and item["silent_steps"] == 0 for item in metrics) / len(metrics),
        "fallback_assisted_success_rate": sum(item["success"] and item["silent_steps"] > 0 for item in metrics) / len(metrics),
        "mean_steps": float(np.mean([item["steps"] for item in metrics])),
        "mean_success_steps": float(np.mean(successful)) if successful else None,
        "mean_final_distance": float(np.mean([item["final_distance"] for item in metrics])),
        "silent_step_rate": sum(item["silent_steps"] for item in metrics) / sum(item["steps"] for item in metrics),
        "mean_motor_spikes": sum(item["motor_spikes"] for item in metrics) / len(metrics),
    })


def weight_diagnostics(brain):
    projection = next(p for p in brain.projections if p.source_name == "input" and p.target_name == "motor")
    ns = projection.n_synapses
    alive = projection.syn_alive[:ns]
    weights = projection.syn_weight[:ns][alive]
    matrix = np.zeros((grid.N_INPUT, grid.N_MOTOR), dtype=np.float64)
    pre = projection.syn_pre[:ns][alive].cpu().numpy()
    post = projection.syn_post[:ns][alive].cpu().numpy()
    np.add.at(matrix, (pre, post), weights.cpu().numpy())
    return {"state_action_weights": matrix.tolist(),
            "transmission_modulation": projection.syn_modulation[:ns][alive].cpu().tolist(),
            "at_lower_bound": int((weights <= projection.syn_min_weight[:ns][alive] + 1e-6).sum()),
            "at_upper_bound": int((weights >= projection.syn_max_weight[:ns][alive] - 1e-6).sum()),
            "motor_theta": brain.regions["motor"].theta[:grid.N_MOTOR].cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    root = Path(__file__).resolve().parent.parent
    study_bytes = (args.study / "summary.json").read_bytes()
    study = json.loads(study_bytes)
    if not study["complete"]:
        parser.error("The source benchmark must be complete")
    if study["numpy"] != np.__version__ or study["torch"] != torch.__version__:
        raise ValueError("Dependency versions differ from the recorded benchmark")
    sources = sorted(set(study["source_sha256"]) | {str(Path(__file__).resolve().relative_to(root))})
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    if any(hashes[name] != digest for name, digest in study["source_sha256"].items()):
        raise ValueError("The benchmark code differs from the source study")
    checkpoint = args.study / "checkpoints/selected"
    checkpoint_hash = directory_digest(checkpoint)
    protocol = {key: study[key] for key in ("config", "source_sha256", "evaluation")}
    brain, progress = load_training_checkpoint(checkpoint, protocol)
    if progress["selected_by"] != "checkpoint_starts" or progress["completed_episodes"] != study["best_episode"]:
        raise ValueError("Checkpoint is not the recorded selection")
    before = copy.deepcopy(brain)
    result = {"complete": False, "study": str(args.study.resolve()),
              "study_sha256": hashlib.sha256(study_bytes).hexdigest(),
              "source_sha256": hashes, "checkpoint_sha256": checkpoint_hash,
              "selected_episode": study["best_episode"], "timing": [],
              "interpretation": "Exact reuse versus full simulation within the corrected independent-episode "
              "protocol. Final starts overlap the same finite world as selection; not unseen-state generalization."}
    with patch.multiple(grid, **study["config"]):
        evaluation = study["evaluation"]
        env = grid.GridWorld(size=evaluation["size"], goal=tuple(evaluation["goal"]))
        if grid.ACTION_DELTAS.tolist() != evaluation["action_deltas"]:
            raise ValueError("Action encoding differs from the study")
        trace = trace_policy(brain, env, study["exhaustive_starts"], grid.SEED + 6000)
        if trace["aggregate"] != study["exhaustive_policy"]:
            raise AssertionError("Traced rollouts differ from exhaustive evaluation")
        result.update(trace=trace, weights=weight_diagnostics(brain))
        atomic_json(args.output, grid.json_safe(result))
        print("TRACE " + json.dumps({key: trace[key] for key in (
            "aggregate", "successes_without_fallback", "successes_with_fallback",
            "decisions", "spike_ties", "wall_actions", "action_counts")}), flush=True)
        for fast in (False, True, True, False):
            started = time.perf_counter()
            evaluated = grid.json_safe(grid.evaluate_policy(brain, env, np.random.default_rng(grid.SEED + 5000),
                start_positions=evaluation["final_starts"], fast=fast))
            elapsed = time.perf_counter() - started
            if evaluated != study["final_policy"]:
                raise AssertionError("Inference did not reproduce all recorded final metrics")
            row = {"fast": fast, "wall_s": elapsed}
            result["timing"].append(row)
            print("INFERENCE " + json.dumps(row), flush=True)
            atomic_json(args.output, result)
        result["final_policy"] = evaluated
    with tempfile.TemporaryDirectory(prefix="grid-inference-state-") as temporary:
        left, right = Path(temporary) / "before", Path(temporary) / "after"
        save_brain(before, left)
        save_brain(brain, right)
        if directory_digest(left) != directory_digest(right):
            raise AssertionError("Inference changed the source network or RNG state")
    if checkpoint_hash != directory_digest(checkpoint) or study_bytes != (args.study / "summary.json").read_bytes():
        raise AssertionError("Source checkpoint or study changed")
    if hashes != {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}:
        raise AssertionError("Sources changed during the check")
    timings = {fast: statistics.median(row["wall_s"] for row in result["timing"] if row["fast"] == fast)
               for fast in (False, True)}
    result.update(complete=True, source_unchanged=True, all_metrics_equal=True,
                  inference_speedup=timings[False] / timings[True])
    atomic_json(args.output, grid.json_safe(result))
    print("COMPLETE " + json.dumps({"speedup": result["inference_speedup"],
        "success": evaluated["success_rate"], "spike_ties": trace["spike_ties"],
        "wall_actions": trace["wall_actions"]}), flush=True)


if __name__ == "__main__":
    main()
