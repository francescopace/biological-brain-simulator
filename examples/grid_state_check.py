"""Counterfactual Grid inference on a fixed checkpoint; no training or tuning claim.

Separate learned readout weights, motor-threshold asymmetry, tonic drive and
spike-tie handling. Every arm uses the same 24 starts and the actual neural
rollout. These interventions change the model/protocol and are not speedups.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

import examples.grid_nav_benchmark as grid
from examples.grid_policy_check import trace_policy, weight_diagnostics
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint
from src.persistence import save_brain


TRANSMISSION_CASES = {f"transmission_gain{gain}_no_motor_baseline": gain for gain in (32, 64, 128, 256, 512)}
CASES = ("baseline", "initial_readout", "equal_motor_theta", "readout_gain16",
         "readout_gain16_equal_theta", "no_motor_baseline", "voltage_spike_ties",
         "equal_theta_no_motor_baseline", "readout_gain16_no_motor_baseline",
         "readout_gain16_equal_theta_no_motor_baseline", "equal_theta_rest100", "equal_theta_rest200",
         *TRANSMISSION_CASES)


def variant(brain, case):
    if case not in CASES:
        raise ValueError("Unknown Grid intervention")
    model = copy.deepcopy(brain)
    projection = model.get_projection("input", "motor")
    ns = projection.n_synapses
    if case == "initial_readout":
        projection.syn_weight[:ns] = grid.READOUT_INIT_WEIGHT
    if "readout_gain16" in case:
        scaled = projection.syn_weight[:ns] * 16.
        if torch.any(scaled > projection.syn_max_weight[:ns]) or torch.any(scaled < projection.syn_min_weight[:ns]):
            raise ValueError("Diagnostic gain would violate readout bounds")
        projection.syn_weight[:ns].copy_(scaled)
    if case in TRANSMISSION_CASES:
        # Fixed transmission gain: weights and their plasticity bounds are not
        # edited. This changes effective synaptic current, not the action rule.
        projection.syn_modulation[:ns].mul_(TRANSMISSION_CASES[case])
    if case == "equal_motor_theta" or "equal_theta" in case:
        theta = model.regions["motor"].theta[:grid.N_MOTOR]
        theta.fill_(float(theta.mean()))
    return model


def voltage_tie_policy(original):
    def choose(counts, voltage, epsilon, rng):
        action, source = original(counts, voltage, epsilon, rng)
        if source == "policy":
            tied = np.flatnonzero(counts == counts.max())
            if len(tied) > 1:
                action = int(tied[np.argmax(voltage[tied])])
        return action, source
    return choose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES,
                        help="Selected interventions; baseline reproduction always runs first")
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
    sources = sorted(set(study["source_sha256"]) | {str(Path(__file__).resolve().relative_to(root)),
                                                   "examples/grid_policy_check.py"})
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    if any(hashes[name] != digest for name, digest in study["source_sha256"].items()):
        raise ValueError("The benchmark code differs from the source study")
    checkpoint = args.study / "checkpoints/selected"
    checkpoint_hash = directory_digest(checkpoint)
    brain, progress = load_training_checkpoint(checkpoint,
        {key: study[key] for key in ("config", "source_sha256", "evaluation")})
    if progress["selected_by"] != "checkpoint_starts" or progress["completed_episodes"] != study["best_episode"]:
        raise ValueError("Checkpoint is not the recorded selection")
    original = copy.deepcopy(brain)
    result = {"complete": False, "study": str(args.study.resolve()),
              "study_sha256": hashlib.sha256(study_bytes).hexdigest(), "source_sha256": hashes,
              "checkpoint_sha256": checkpoint_hash, "selected_episode": study["best_episode"],
              "interpretation": "Diagnostic interventions on saved state, not alternative training runs "
              "or a held-out comparison. Action choices use neural activity, never environment-optimal actions.",
              "cases": {}}
    with patch.multiple(grid, **study["config"]):
        evaluation = study["evaluation"]
        if grid.ACTION_DELTAS.tolist() != evaluation["action_deltas"]:
            raise ValueError("Action encoding differs from the study")
        for case in dict.fromkeys(("baseline", *args.cases)):
            model = variant(brain, case)
            env = grid.GridWorld(size=evaluation["size"], goal=tuple(evaluation["goal"]))
            overrides = {}
            if "no_motor_baseline" in case:
                overrides["MOTOR_BASELINE_CURRENT"] = 0.
            if case in ("equal_theta_rest100", "equal_theta_rest200"):
                overrides["INTER_STEP_REST"] = 100 if case.endswith("100") else 200
            if case == "voltage_spike_ties":
                overrides["choose_action"] = voltage_tie_policy(grid.choose_action)
            with patch.multiple(grid, **overrides) if overrides else nullcontext():
                trace = trace_policy(model, env, study["exhaustive_starts"], grid.SEED + 6000)
            if case == "baseline" and trace["aggregate"] != study["exhaustive_policy"]:
                raise AssertionError("Baseline does not reproduce the saved policy")
            result["cases"][case] = {"trace": trace, "weights": weight_diagnostics(model)}
            atomic_json(args.output, grid.json_safe(result))
            print("CASE " + json.dumps({"case": case, **trace["aggregate"],
                "successes_without_fallback": trace["successes_without_fallback"],
                "successes_with_fallback": trace["successes_with_fallback"],
                "spike_ties": trace["spike_ties"], "wall_actions": trace["wall_actions"]}), flush=True)
    with tempfile.TemporaryDirectory(prefix="grid-state-source-") as temporary:
        left, right = Path(temporary) / "before", Path(temporary) / "after"
        save_brain(original, left)
        save_brain(brain, right)
        if directory_digest(left) != directory_digest(right):
            raise AssertionError("Diagnostic changed the source model or RNG")
    if checkpoint_hash != directory_digest(checkpoint) or study_bytes != (args.study / "summary.json").read_bytes():
        raise AssertionError("Source checkpoint or study changed")
    if hashes != {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}:
        raise AssertionError("Sources changed during the check")
    result.update(complete=True, source_unchanged=True)
    atomic_json(args.output, grid.json_safe(result))
    print("COMPLETE: diagnostic source model and files unchanged", flush=True)


if __name__ == "__main__":
    main()
