"""Evaluation-only interventions on a saved MNIST checkpoint; never trains.

All variants use the same fitted network, readout samples and test samples.
Test accuracies are diagnostic comparisons, not a new held-out model selection.
The existing MNIST benchmark is patched only inside this diagnostic process.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_diagnosis import ProtocolConfig, load_dataset, patched_benchmark_globals
from src.persistence import load_brain


MODES = (
    "baseline", "zero_theta", "independent_samples",
    "independent_zero_theta", "rest_frozen_theta", "rest_adaptive_theta",
)


def reset_transients(brain):
    """Restore repeatable electrical/STP state, preserving theta and weights.

    This is a diagnostic intervention, not a change to the training protocol.
    Time and cumulative counters remain monotonic. Oscillations must be disabled
    so that absolute simulation time cannot change the next sample's drive.
    """
    mn.reset_inference_state(brain)


def network_digest(brain):
    """Fingerprint long-term weights and topology, not electrical state."""
    digest = hashlib.sha256()
    for region in brain.regions.values():
        digest.update(region.name.encode())
        digest.update(str(region.n_neurons).encode())
        for attr in ("neuron_alive", "neuron_type"):
            digest.update(getattr(region, attr)[:region.n_neurons].cpu().numpy().tobytes())
    for target in [*brain.regions.values(), *brain.projections]:
        ns = target.n_synapses
        digest.update(str(ns).encode())
        for attr in ("syn_pre", "syn_post", "syn_alive", "syn_weight", "syn_delay"):
            digest.update(getattr(target, attr)[:ns].cpu().numpy().tobytes())
    return digest.hexdigest()


def state_summary(brain):
    result = {}
    for name, region in brain.regions.items():
        result[name] = {}
        for attr in ("v", "u", "theta", "activity"):
            values = getattr(region, attr)[:region.n_neurons]
            result[name][attr] = {
                "min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max()),
            }
    return result


@contextmanager
def intervention(mode, rest_steps=5000):
    if mode not in MODES and mode != "standard":
        raise ValueError(f"Unknown intervention: {mode}")
    original_snapshot = mn._inference_brain
    original_present = mn.present_sample
    snapshots = []
    presentations = []

    def snapshot(source):
        brain = original_snapshot(source)
        expected = network_digest(brain)
        if mode in ("zero_theta", "independent_zero_theta"):
            for region in brain.regions.values():
                region.theta[:region.n_neurons] = 0.0
        if mode.startswith("rest_"):
            if mode == "rest_adaptive_theta":
                brain.enable_adaptive_thresholds()
            for _ in range(rest_steps):
                brain.step()
            brain.freeze_adaptive_thresholds()
        snapshots.append((brain, expected, state_summary(brain)))
        return brain

    def present(brain, x, n_steps, learn=False):
        if learn:
            raise ValueError("State diagnosis must not train")
        if mode in ("independent_samples", "independent_zero_theta"):
            reset_transients(brain)
        counts, voltage = original_present(brain, x, n_steps, learn=False)
        presentations.append((counts.copy(), voltage.copy()))
        return counts, voltage

    # Keep the historical baseline reproducible after independent inference
    # becomes the standard; individual interventions apply their own resets.
    with patch.object(mn, "INDEPENDENT_INFERENCE", mode == "standard"), \
            patch.object(mn, "_inference_brain", snapshot), \
            patch.object(mn, "present_sample", present):
        try:
            yield snapshots, presentations
        finally:
            for brain, expected, _ in snapshots:
                if network_digest(brain) != expected:
                    raise AssertionError("Evaluation changed weights or topology")


def response_metrics(presentations, spike_templates, voltage_templates, y, repeats,
                     *, requested_repeats=None):
    counts = np.asarray([p[0] for p in presentations]).reshape(len(y), repeats, -1).sum(axis=1)
    voltage = np.asarray([p[1] for p in presentations]).reshape(len(y), repeats, -1).sum(axis=1)
    if requested_repeats is not None:
        counts = counts * (requested_repeats / repeats)
        voltage = voltage * (requested_repeats / repeats)
    spike_queries = counts / (np.linalg.norm(counts, axis=1, keepdims=True) + 1e-9)
    voltage_queries = voltage / (np.linalg.norm(voltage, axis=1, keepdims=True) + 1e-9)
    spike_scores = spike_queries @ spike_templates.T
    voltage_scores = voltage_queries @ voltage_templates.T
    spike_preds = np.asarray(mn.CLASSES)[spike_scores.argmax(axis=1)]
    silent = counts.sum(axis=1) == 0
    spike_preds[silent] = -1
    voltage_preds = np.asarray(mn.CLASSES)[voltage_scores.argmax(axis=1)]
    return {
        "spike_only_accuracy": float(np.mean(spike_preds == y)),
        "voltage_only_accuracy": float(np.mean(voltage_preds == y)),
        "silent_samples": int(silent.sum()),
        "mean_spikes_per_sample": float(counts.sum(axis=1).mean()),
        "mean_spike_score_range": float(np.ptp(spike_scores, axis=1).mean()),
        "mean_voltage_score_range": float(np.ptp(voltage_scores, axis=1).mean()),
    }


def run_variant(brain, mode, X_readout, y_readout, X_test, y_test, *, reverse=False, rest_steps=5000):
    start = time.monotonic()
    expected = network_digest(brain)
    with intervention(mode, rest_steps) as (snapshots, responses):
        print("  Building readout...", flush=True)
        _, _, spikes, voltage = mn.build_readout(brain, X_readout, y_readout)
        responses.clear()
        print("  Evaluating...", flush=True)
        accuracy, predictions = mn.evaluate(brain, X_test, y_test, spikes, voltage)
        simulated_repeats = len(responses) // len(y_test)
        result = {
            "mode": mode, "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "requested_repeats": mn.TEST_REPEATS,
            "simulated_repeats": simulated_repeats,
            **response_metrics(responses, spikes, voltage, y_test, simulated_repeats,
                               requested_repeats=mn.TEST_REPEATS),
        }
        if reverse:
            responses.clear()
            print("  Checking reversed order...", flush=True)
            reverse_accuracy, reverse_preds = mn.evaluate(
                brain, X_test[::-1], y_test[::-1], spikes, voltage,
            )
            result.update(
                reverse_accuracy=reverse_accuracy,
                order_disagreements=int(np.sum(predictions != reverse_preds[::-1])),
                reverse_predictions_aligned=reverse_preds[::-1].tolist(),
            )
    if network_digest(brain) != expected:
        raise AssertionError("Source checkpoint changed")
    result.update(
        elapsed_s=time.monotonic() - start, weights_and_topology_unchanged=True,
        prepared_state=snapshots[0][2],
    )
    return result


def checkpoint_digest(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def audit_sleep_weights(checkpoint, rest_steps=5000):
    """Separate homeostatic weight drift from quiet electrical relaxation.

    This reproduces the no-replay sleep branch on copies. The saved sleep
    benchmark disables R-STDP; reject other checkpoints rather than silently
    changing their learning protocol. No accuracy is inferred from this audit.
    """
    source = load_brain(checkpoint)
    if source.reward_stdp.enabled:
        raise ValueError("Sleep audit requires the existing R-STDP-disabled checkpoint")
    source_hash = checkpoint_digest(checkpoint)
    before = [*source.regions.values(), *source.projections]
    cases = []
    for freeze_scaling in (False, True):
        brain = copy.deepcopy(source)
        # Match sleep_phase(enable_replay=False): capture remains enabled,
        # consolidation alone is suppressed, and no manual STDP is called.
        brain.memory.consolidate = lambda *args, **kwargs: 0
        if freeze_scaling:
            brain.freeze_homeostatic_scaling()
        for _ in range(rest_steps):
            brain.step()
        differences = []
        for old, new in zip(before, [*brain.regions.values(), *brain.projections]):
            ns = old.n_synapses
            if not ns:
                continue
            w, v = old.syn_weight[:ns], new.syn_weight[:ns]
            label = (f"region:{old.name}" if hasattr(old, "name") else
                     f"projection:{old.source_name}->{old.target_name}")
            changes = {
                "target": label, "changed": int(torch.count_nonzero(v - w)),
                "total": ns, "max_abs_delta": float((v - w).abs().max()),
            }
            for sign, mask in (("excitatory", w > 0), ("inhibitory", w < 0)):
                if torch.any(mask):
                    changes[f"{sign}_before_mean"] = float(w[mask].mean())
                    changes[f"{sign}_after_mean"] = float(v[mask].mean())
            differences.append(changes)
        cases.append({
            "frozen_scaling": freeze_scaling, "weight_changes": differences,
            "final_state": state_summary(brain),
        })
    if checkpoint_digest(checkpoint) != source_hash:
        raise AssertionError("Sleep checkpoint files changed")
    return {
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": source_hash,
        "checkpoint_unchanged": True, "rest_steps": rest_steps,
        "manual_stdp": False, "reward_stdp": False,
        "original_state": state_summary(source), "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="JSON containing the saved protocol config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rest-steps", type=int, default=5000)
    parser.add_argument(
        "--compare-initial", action="store_true",
        help="Compare the initial network and checkpoint with standard independent inference, instead of ablations",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve earlier runs")
    if args.rest_steps < 0:
        parser.error("rest-steps must be nonnegative")
    reference = json.loads(args.reference.read_text())
    config = ProtocolConfig(**reference["config"])
    brain = load_brain(args.checkpoint)
    source_hash = checkpoint_digest(args.checkpoint)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_hash,
        "config": reference["config"], "reference_accuracy": reference["accuracy"],
        "training": False, "rest_steps": args.rest_steps,
        "interpretation": "Diagnostic reuse of test data; not held-out model selection.",
        "original_state": state_summary(brain), "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with patched_benchmark_globals(config):
        X, y, Xt, yt = load_dataset(config)
        Xr, yr = mn.build_readout_subset(
            X, y, readout_per_class=config.readout_per_class, seed=config.seed,
        )
        payload["test_labels"] = yt.tolist()
        print(f"Readout={len(Xr)} test={len(Xt)}; no training", flush=True)
        if args.compare_initial:
            projection = brain.get_projection("input", "cortex")
            posts = projection.syn_post[:projection.n_synapses].to(torch.int64)
            has_inhibitory_targets = bool((brain.regions["cortex"].neuron_type[posts] == 1).any())
            initial = mn.build_brain(
                seed=config.seed, integration_method=brain.integration_method,
                integration_max_step=brain.integration_max_step,
                feedforward_exc_only=not has_inhibitory_targets,
            )
            for name, region in brain.regions.items():
                initial.regions[name].integration_method = region.integration_method
                initial.regions[name].integration_max_step = region.integration_max_step
            assert list(initial.regions) == list(brain.regions), "Region mismatch"
            assert len(initial.projections) == len(brain.projections), "Projection mismatch"
            for left, right in zip(initial.regions.values(), brain.regions.values()):
                assert left.n_neurons == right.n_neurons, "Neuron count mismatch"
                for attr in ("neuron_alive", "neuron_type", "a", "b", "c", "d"):
                    assert torch.equal(getattr(left, attr), getattr(right, attr)), f"Neuron mismatch: {attr}"
            for left, right in zip(
                [*initial.regions.values(), *initial.projections],
                [*brain.regions.values(), *brain.projections],
            ):
                assert left.n_synapses == right.n_synapses, "Synapse count mismatch"
                for attr in ("syn_pre", "syn_post", "syn_alive", "syn_delay"):
                    n = left.n_synapses
                    assert torch.equal(getattr(left, attr)[:n], getattr(right, attr)[:n]), f"Topology mismatch: {attr}"
            payload["comparison"] = "initial_vs_checkpoint_standard_independent"
            payload["interpretation"] += (
                " Each model gets a decoder fitted on the same labelled samples."
                " The comparison measures the training pipeline, including weight"
                " normalization and retained theta, not STDP alone."
            )
            variants = [("initial", initial), ("trained", brain)]
        else:
            variants = [(mode, brain) for mode in args.modes]
        for mode, model in variants:
            print(f"START {mode}", flush=True)
            result = run_variant(
                model, "standard" if args.compare_initial else mode, Xr, yr, Xt, yt,
                reverse=mode in ("baseline", "independent_samples", "trained"),
                rest_steps=args.rest_steps,
            )
            result["mode"] = mode
            if args.compare_initial and mode == "trained":
                assert result["order_disagreements"] == 0, "Standard inference depends on order"
                payload["training_pipeline_gain_pp"] = 100 * (
                    result["accuracy"] - payload["results"][0]["accuracy"]
                )
            payload["results"].append(result)
            if checkpoint_digest(args.checkpoint) != source_hash:
                raise AssertionError("Checkpoint files changed")
            payload["checkpoint_unchanged"] = True
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"END {mode}: accuracy={result['accuracy']:.1%} "
                f"spike={result['spike_only_accuracy']:.1%} "
                f"silent={result['silent_samples']} "
                f"order_disagreements={result.get('order_disagreements', 'n/a')} "
                f"time={result['elapsed_s']:.1f}s", flush=True,
            )


if __name__ == "__main__":
    main()
