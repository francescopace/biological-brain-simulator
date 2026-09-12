"""Compare optimized simulation paths with their independent-image reference.

Inference uses a saved network; training equivalence uses short-lived copies.
No checkpoint is replaced and no hyperparameter is selected on the test set.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import time
from unittest.mock import patch

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_diagnosis import ProtocolConfig, load_dataset, patched_benchmark_globals
from examples.mnist_state_diagnosis import checkpoint_digest, network_digest
from src.persistence import load_brain
from src.synaptic_events import SynapseEventIndex


def simulation_digest(brain):
    digest = hashlib.sha256()
    digest.update(repr((brain.time, brain.step_count)).encode())
    for target in [brain, brain.encoder, brain.memory, brain.growth,
                   brain.homeostasis, *brain.regions.values(), *brain.projections]:
        for key, value in sorted(vars(target).items()):
            if isinstance(value, torch.Tensor):
                digest.update(key.encode())
                digest.update(value.cpu().numpy().tobytes())
            elif isinstance(value, torch.Generator):
                digest.update(value.get_state().cpu().numpy().tobytes())
            elif isinstance(value, (bool, int, float, str)):
                digest.update(repr((key, value)).encode())
    return digest.hexdigest()


def inference_trial(brain, Xr, yr, Xt, yt, fast):
    snapshots = []
    original_snapshot = mn._inference_brain

    def snapshot(source):
        result = original_snapshot(source)
        snapshots.append((result, network_digest(result)))
        return result

    with patch.object(mn, "FAST_INDEPENDENT_INFERENCE", fast), \
            patch.object(mn, "_inference_brain", snapshot):
        start = time.perf_counter()
        readout = mn.build_readout(brain, Xr, yr)
        readout_s = time.perf_counter() - start
        start = time.perf_counter()
        inference = mn._inference_brain(brain)
        results = [mn.predict_sample(inference, x, *readout[2:]) for x in Xt]
        evaluation_s = time.perf_counter() - start
    for frozen, expected in snapshots:
        if network_digest(frozen) != expected:
            raise AssertionError("Inference changed weights or topology")
    predictions = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    return {
        "fast": fast, "readout_s": readout_s, "evaluation_s": evaluation_s,
        "total_s": readout_s + evaluation_s,
        "simulated_steps": sum(b.step_count - brain.step_count for b, _ in snapshots),
        "accuracy": float(np.mean(predictions == yt)),
        "predictions": predictions.tolist(),
    }, readout, scores


def reference_present(brain, x, n_steps):
    """Pre-optimization training loop, including its unused response collection."""
    cortex = brain.regions["cortex"]
    indices = mn.excitatory_cortex_indices(brain)
    before = cortex.total_spikes[indices].clone()
    voltage_sum = torch.zeros(len(indices), dtype=torch.float32, device=cortex.v.device)
    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.step()
        voltage_sum += cortex.v[indices]
        mn.apply_feedforward_stdp(brain)
    counts = (cortex.total_spikes[indices] - before).cpu().numpy()
    voltage = (voltage_sum / max(1, n_steps)).cpu().numpy()
    return counts, voltage


def training_trial(brain, X, steps, fast):
    model = copy.deepcopy(brain)
    target = mn.compute_norm_target(model)
    start = time.perf_counter()
    for x in X:
        if fast:
            mn.present_sample(model, x, steps, learn=True, collect_responses=False)
        else:
            reference_present(model, x, steps)
        mn.normalize_feedforward_weights(model, target)
        mn.reset_brain_state(model)
    elapsed = time.perf_counter() - start
    return {"fast": fast, "elapsed_s": elapsed, "samples": len(X),
            "simulation_sha256": simulation_digest(model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-reference", type=Path)
    parser.add_argument("--training-samples", type=int, default=8)
    parser.add_argument("--profile-steps", type=int, default=300)
    parser.add_argument("--background-note", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--thread-sweep-only", action="store_true",
                      help="Skip inference; compare CPU thread counts on training copies")
    mode.add_argument("--event-index-check", action="store_true",
                      help="Compare dense and indexed events with independent inference in both arms")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    if args.training_samples < 0 or args.profile_steps < 0:
        parser.error("Sample and step counts must be nonnegative")
    args.output.mkdir(parents=True)
    reference = json.loads(args.reference.read_text())
    config = ProtocolConfig(**reference["config"])
    source_hash = checkpoint_digest(args.checkpoint)
    brain = load_brain(args.checkpoint)
    state_hash = simulation_digest(brain)
    root = Path(__file__).resolve().parent.parent
    sources = ("examples/mnist_benchmark.py", "examples/mnist_optimization_check.py",
               "src/brain.py", "src/region.py", "src/plasticity.py", "src/stimulus.py",
               "src/synaptic_events.py")
    payload = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_hash,
        "config": reference["config"], "torch_version": torch.__version__,
        "device": str(mn.DEVICE), "threads": torch.get_num_threads(),
        "platform": platform.platform(), "background_note": args.background_note,
        "interpretation": "Equivalence check, not accuracy tuning. Timings share the current host load.",
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "inference": [], "training": [], "complete": False,
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    save()
    with patched_benchmark_globals(config):
        X, y, Xt, yt = load_dataset(config)
        if args.thread_sweep_only:
            original_threads = torch.get_num_threads()
            counts = sorted({1, 2, original_threads})
            payload["thread_trials"] = []
            try:
                for count in [*counts, *reversed(counts)]:
                    torch.set_num_threads(count)
                    # A throwaway presentation warms the thread pool without
                    # advancing any measured model or its random generator.
                    warm = copy.deepcopy(brain)
                    mn.present_sample(warm, X[0], 20, learn=True, collect_responses=False)
                    print(f"START CPU threads={count}", flush=True)
                    result = training_trial(brain, X[:args.training_samples],
                                            config.train_present_steps, fast=True)
                    result["threads"] = count
                    payload["thread_trials"].append(result)
                    print(json.dumps(result), flush=True)
                    save()
            finally:
                torch.set_num_threads(original_threads)
            assert len({r["simulation_sha256"] for r in payload["thread_trials"]}) == 1
            payload["exact_training_state_and_rng_across_threads"] = True
            payload["median_seconds_by_threads"] = {
                str(count): float(np.median([r["elapsed_s"] for r in payload["thread_trials"]
                                            if r["threads"] == count])) for count in counts
            }
            assert simulation_digest(brain) == state_hash
            assert checkpoint_digest(args.checkpoint) == source_hash
            payload["source_unchanged"] = True
            payload["complete"] = True
            save()
            print("PASS: exact training state across thread counts", flush=True)
            return
        Xr, yr = mn.build_readout_subset(X, y, config.readout_per_class, seed=config.seed)
        outputs = []
        for fast in (False, True):
            print(f"START inference fast={fast} indexed_check={args.event_index_check}: "
                  f"readout={len(Xr)} test={len(Xt)}", flush=True)
            with patch.object(SynapseEventIndex, "enabled", fast if args.event_index_check else True):
                result, readout, scores = inference_trial(
                    brain, Xr, yr, Xt, yt, True if args.event_index_check else fast,
                )
            if args.event_index_check:
                result["indexed_events"] = fast
            outputs.append((readout, scores))
            payload["inference"].append(result)
            save()
            print(json.dumps(result | {"predictions": "saved"}), flush=True)
        for a, b in zip(outputs[0][0], outputs[1][0]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        first, second = payload["inference"]
        assert first["predictions"] == second["predictions"]
        payload["exact_readout_scores_and_predictions"] = True
        payload["inference_speedup"] = first["total_s"] / second["total_s"]
        if args.learning_reference:
            previous = json.loads(args.learning_reference.read_text())
            assert previous["config"] == reference["config"]
            assert previous["checkpoint_sha256"] == source_hash
            assert previous["test_labels"] == yt.tolist()
            trained = next(r for r in previous["results"] if r["mode"] == "trained")
            assert trained["predictions"] == second["predictions"]
            payload["matches_preoptimization_predictions"] = True
        save()
        # ABBA order limits warm-up/order bias; do not overinterpret short timings.
        for fast in (False, True, True, False):
            print(f"START training-copy fast={fast}", flush=True)
            with patch.object(SynapseEventIndex, "enabled", fast if args.event_index_check else True):
                result = training_trial(brain, X[:args.training_samples],
                                        config.train_present_steps,
                                        True if args.event_index_check else fast)
            if args.event_index_check:
                result["fast"] = fast
                result["indexed_events"] = fast
            payload["training"].append(result)
            save()
        assert len({r["simulation_sha256"] for r in payload["training"]}) == 1
        payload["exact_training_state_and_rng"] = True
        baseline = np.median([r["elapsed_s"] for r in payload["training"] if not r["fast"]])
        fast_time = np.median([r["elapsed_s"] for r in payload["training"] if r["fast"]])
        payload["training_speedup"] = float(baseline / fast_time)
        if args.profile_steps:
            model = copy.deepcopy(brain)
            profile = cProfile.Profile()
            profile.runcall(mn.present_sample, model, X[0], args.profile_steps,
                            learn=True, collect_responses=False)
            profile.dump_stats(str(args.output / "training.prof"))
            report = io.StringIO()
            pstats.Stats(profile, stream=report).sort_stats("cumulative").print_stats(45)
            (args.output / "profile.txt").write_text(report.getvalue())
    assert simulation_digest(brain) == state_hash
    assert checkpoint_digest(args.checkpoint) == source_hash
    payload["source_unchanged"] = True
    payload["comparison"] = "indexed_vs_dense_events" if args.event_index_check else "inference_schedule"
    payload["complete"] = True
    save()
    print(f"PASS: inference {payload['inference_speedup']:.2f}x, "
          f"training {payload['training_speedup']:.2f}x; exact checks passed", flush=True)


if __name__ == "__main__":
    main()
