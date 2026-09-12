"""Summarize complete, matched Grid runs without treating starts as new worlds."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


def summarize(studies):
    if len(studies) < 2:
        raise ValueError("Replication requires at least two studies")
    rows, common, source_hashes, seen_seeds = [], None, None, set()
    for directory in map(Path, studies):
        raw = (directory / "summary.json").read_bytes()
        study = json.loads(raw)
        if not study["complete"] or not study["source_unchanged"]:
            raise ValueError("Every source study must be complete with unchanged sources")
        config = study["config"]
        seed = config["SEED"]
        if seed in seen_seeds:
            raise ValueError("Replication seeds must be distinct")
        seen_seeds.add(seed)
        evaluation = study["evaluation"]
        expected_starts = [[r, c] for r in range(evaluation["size"]) for c in range(evaluation["size"])
                           if [r, c] != evaluation["goal"]]
        if study["exhaustive_starts"] != expected_starts:
            raise ValueError("Exhaustive evaluation must cover each non-goal state once")
        signature = {"config": {key: value for key, value in config.items() if key != "SEED"},
                     "world": {key: evaluation[key] for key in ("size", "goal", "action_deltas")},
                     "numpy": study["numpy"], "torch": study["torch"]}
        if common is None:
            common, source_hashes = signature, study["source_sha256"]
        if signature != common or study["source_sha256"] != source_hashes:
            raise ValueError("Replication protocols, versions and source hashes must match")
        if [row["episode"] for row in study["episodes"]] != list(range(1, config["EPISODES"] + 1)):
            raise ValueError("Source study did not record all requested episodes")
        for name, digest in source_hashes.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source snapshot names must be repository-relative")
            if hashlib.sha256((directory / "source_snapshot" / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Source snapshot integrity check failed")
        protocol = {key: study[key] for key in ("config", "source_sha256", "evaluation")}
        checkpoint = directory / "checkpoints/selected"
        _, progress = load_training_checkpoint(checkpoint, protocol)
        if progress["completed_episodes"] != study["best_episode"] or progress["selected_by"] != "checkpoint_starts":
            raise ValueError("Selected checkpoint does not match the study")
        exhaustive = study["exhaustive_policy"]
        for metrics in (study["final_policy"], exhaustive):
            rates = [metrics[key] for key in ("success_rate", "spike_only_success_rate", "fallback_assisted_success_rate")]
            if any(not 0 <= rate <= 1 for rate in rates) or abs(rates[0] - rates[1] - rates[2]) > 1e-12:
                raise ValueError("Success metrics are inconsistent")
        count = exhaustive["spike_only_success_rate"] * len(expected_starts)
        if abs(count - round(count)) > 1e-8:
            raise ValueError("Exhaustive success rate does not describe an integer count")
        rows.append({"seed": seed, "study": str(directory.resolve()),
                     "study_sha256": hashlib.sha256(raw).hexdigest(),
                     "checkpoint_sha256": directory_digest(checkpoint),
                     "selected_episode": study["best_episode"], "initial": study["initial_policy"],
                     "random_baseline": study["random_baseline"], "final": study["final_policy"],
                     "exhaustive": exhaustive, "exhaustive_spike_only_successes": round(count),
                     "distinct_starts": len(expected_starts)})
        if raw != (directory / "summary.json").read_bytes():
            raise ValueError("Study changed while being summarized")
    rows.sort(key=lambda row: row["seed"])
    aggregate = {}
    for metric in ("success_rate", "spike_only_success_rate", "fallback_assisted_success_rate", "mean_steps"):
        values = [row["exhaustive"][metric] for row in rows]
        aggregate[metric] = {"mean": statistics.mean(values), "minimum": min(values), "maximum": max(values)}
    return {"complete": True, "signature": common, "training_source_sha256": source_hashes,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runs": rows, "exhaustive_across_seeds": aggregate,
            "interpretation": "Replicated network/training seeds on the same finite world, not independent "
            "held-out environments. Seed-specific final starts are reported separately; comparisons use all "
            "distinct non-goal starts. No training-speed comparison is made."}


def verify_current_defaults(studies):
    """Prove default promotion preserves initial states and recorded metrics."""
    import numpy as np
    import torch
    import examples.grid_nav_benchmark as grid
    from examples.mnist_optimization_check import simulation_digest

    checks = []
    for directory in map(Path, studies):
        raw = (directory / "summary.json").read_bytes()
        study = json.loads(raw)
        if not study["complete"] or not study["source_unchanged"]:
            raise ValueError("Default verification requires a complete source study")
        if study["numpy"] != np.__version__ or study["torch"] != torch.__version__:
            raise ValueError("Current dependency versions differ from the study")
        for key, value in study["config"].items():
            if key != "SEED" and getattr(grid, key) != value:
                raise ValueError(f"Current default differs from the studied configuration: {key}")
        evaluation = study["evaluation"]
        if grid.ACTION_DELTAS.tolist() != evaluation["action_deltas"]:
            raise ValueError("Current action encoding differs from the study")
        protocol = {key: study[key] for key in ("config", "source_sha256", "evaluation")}
        initial_hash = directory_digest(directory / "checkpoints/initial")
        selected_hash = directory_digest(directory / "checkpoints/selected")
        initial, _ = load_training_checkpoint(directory / "checkpoints/initial", protocol)
        seed = study["config"]["SEED"]
        expected_initial = simulation_digest(initial)
        if simulation_digest(grid.build_brain(seed)) != expected_initial:
            raise AssertionError("Default builder changed the full initial state or RNG")
        selected, _ = load_training_checkpoint(directory / "checkpoints/selected", protocol)
        before = simulation_digest(selected)
        env = grid.GridWorld(size=evaluation["size"], goal=tuple(evaluation["goal"]))
        for label, starts, offset in (("final_policy", evaluation["final_starts"], 5000),
                                     ("exhaustive_policy", study["exhaustive_starts"], 6000)):
            metrics = grid.json_safe(grid.evaluate_policy(selected, env, np.random.default_rng(seed + offset),
                                                          start_positions=starts))
            if metrics != study[label]:
                raise AssertionError(f"Current defaults changed {label} for seed {seed}")
        if simulation_digest(selected) != before:
            raise AssertionError("Default verification changed the saved model or RNG")
        if (raw != (directory / "summary.json").read_bytes()
                or initial_hash != directory_digest(directory / "checkpoints/initial")
                or selected_hash != directory_digest(directory / "checkpoints/selected")):
            raise AssertionError("Source study or checkpoints changed during verification")
        checks.append({"seed": seed, "initial_state_sha256": expected_initial,
                       "selected_state_sha256": before, "saved_metrics_reproduced": True})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-defaults", action="store_true",
                        help="Reproduce saved initialization and metrics using current benchmark defaults")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    result = summarize(args.studies)
    if args.verify_defaults:
        root = Path(__file__).resolve().parent.parent
        paths = sorted(set(result["training_source_sha256"]) | {
            "examples/grid_replication_summary.py", "examples/mnist_optimization_check.py"})
        hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
        result["current_defaults_check"] = verify_current_defaults(args.studies)
        if hashes != {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}:
            raise AssertionError("Sources changed during default verification")
        result["current_source_sha256"] = hashes
    atomic_json(args.output, result)
    print(json.dumps({"seeds": [row["seed"] for row in result["runs"]],
                      "exhaustive": result["exhaustive_across_seeds"]}), flush=True)


if __name__ == "__main__":
    main()
