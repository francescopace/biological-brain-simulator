"""Paired learning controls and cached readout probes on a train-only validation split.

The canonical MNIST test split is excluded. Results guide subsequent experiments;
they are not held-out test estimates or proof of a general learning advantage.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

import examples.mnist_benchmark as mn
from examples.mnist_trace_stdp import validate_learning_rule
from examples.mnist_readout_features import readout_features
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import (
    array_digest, atomic_json, load_training_checkpoint, save_training_checkpoint,
)


CONDITIONS = ("initial", "normalization_only", "stdp_normalized")
DECODERS = ("template", "ridge_spikes", "ridge_spikes_voltage", "ridge_spikes_centered_voltage")


@dataclass(frozen=True)
class LearningConfig:
    train_per_class: int = 20
    readout_per_class: int = 10
    validation_per_class: int = 20
    split_seed: int = 20260912
    epochs: int = 1
    train_steps: int = 100
    inference_steps: int = 50
    rest_steps: int = 25
    checkpoint_every: int = 100
    ridge_alpha: float = 1.0
    normalization_policy: str = "population_mean"
    exc_to_inh_weight: float = 8.0
    stdp_scale: float = 0.2
    input_density: float = 0.15
    input_weight_boost: float = 4.0
    learning_rule: str = "pair"
    trace_tau: float = 20.0
    trace_target: float = 0.2

    def validate(self):
        positive = (self.train_per_class, self.readout_per_class, self.validation_per_class,
                    self.epochs, self.train_steps, self.inference_steps, self.checkpoint_every)
        if (min(positive) < 1 or self.rest_steps < 0 or self.ridge_alpha <= 0
                or not math.isfinite(self.ridge_alpha)):
            raise ValueError("Invalid learning-check configuration")
        if self.readout_per_class > self.train_per_class:
            raise ValueError("Readout samples must be a subset of training samples")
        if self.normalization_policy not in ("population_mean", "initial_per_neuron"):
            raise ValueError("Unknown normalization policy")
        if not math.isfinite(self.exc_to_inh_weight) or self.exc_to_inh_weight < 0:
            raise ValueError("Excitatory-to-inhibitory coupling must be finite and nonnegative")
        if not math.isfinite(self.stdp_scale) or self.stdp_scale < 0:
            raise ValueError("STDP scale must be finite and nonnegative")
        validate_learning_rule(self.learning_rule, self.trace_tau, self.trace_target)
        mn.validate_input_projection(self.input_density, self.input_weight_boost)


@dataclass(frozen=True)
class Dataset:
    train_X: np.ndarray
    train_y: np.ndarray
    readout_indices: np.ndarray
    validation_X: np.ndarray
    validation_y: np.ndarray
    manifest: dict


def prepare_dataset(raw_X, labels, config, *, classes=tuple(range(10)), train_boundary=60000):
    """Split raw canonical training rows before fitting intensity equalization."""
    config.validate()
    if len(raw_X) != len(labels) or len(labels) < train_boundary:
        raise ValueError("Dataset does not cover the canonical training split")
    rng = np.random.default_rng(config.split_seed)
    train_ids, validation_ids = [], []
    canonical_y = np.asarray(labels[:train_boundary], dtype=np.int64)
    for cls in classes:
        available = np.flatnonzero(canonical_y == cls)
        needed = config.train_per_class + config.validation_per_class
        if len(available) < needed:
            raise ValueError(f"Not enough canonical training samples for class {cls}")
        rng.shuffle(available)
        train_ids.extend(available[:config.train_per_class])
        validation_ids.extend(available[config.train_per_class:needed])
    train_ids, validation_ids = np.array(train_ids), np.array(validation_ids)
    rng.shuffle(train_ids)
    rng.shuffle(validation_ids)
    train_raw = np.asarray(raw_X[train_ids], dtype=np.float64) / 255.0
    validation_raw = np.asarray(raw_X[validation_ids], dtype=np.float64) / 255.0
    target_l1 = mn.l1_equalization_target(train_raw, factor=mn.DOWNSAMPLE)
    train_X = mn.downsample_images(train_raw, factor=mn.DOWNSAMPLE, target_l1=target_l1)
    validation_X = mn.downsample_images(validation_raw, factor=mn.DOWNSAMPLE, target_l1=target_l1)
    train_y, validation_y = canonical_y[train_ids], canonical_y[validation_ids]
    readout = mn._balanced_subset(train_y, classes, config.readout_per_class, config.split_seed)
    manifest = {
        "source": "mnist_784 version 1, canonical training rows only",
        "canonical_train_boundary": train_boundary,
        "train_ids": train_ids.tolist(), "validation_ids": validation_ids.tolist(),
        "readout_ids": train_ids[readout].tolist(), "intensity_target_l1": target_l1,
        "train_sha256": array_digest(train_X, train_y),
        "validation_sha256": array_digest(validation_X, validation_y),
        "readout_sha256": array_digest(train_X[readout], train_y[readout]),
    }
    return Dataset(train_X, train_y, readout, validation_X, validation_y, manifest)


def weight_delta_metrics(before, after_stdp, after_normalization):
    raw = (after_stdp - before).to(torch.float64)
    norm = (after_normalization - after_stdp).to(torch.float64)
    total = (after_normalization - before).to(torch.float64)
    raw_l1, norm_l1, total_l1 = (float(d.abs().sum()) for d in (raw, norm, total))
    denominator = float(torch.linalg.vector_norm(raw) * torch.linalg.vector_norm(norm))
    return {
        "raw_stdp_l1": raw_l1, "normalization_l1": norm_l1, "total_l1": total_l1,
        "raw_stdp_changed": int(torch.count_nonzero(raw)),
        "normalization_changed": int(torch.count_nonzero(norm)),
        "raw_normalization_cosine": float(torch.dot(raw, norm)) / denominator if denominator else None,
        "total_to_raw_l1": total_l1 / raw_l1 if raw_l1 else None,
    }


def weight_summary(projection, reference=None):
    """Describe live weights and exact bound occupancy without changing state."""
    ns = projection.n_synapses
    live = projection.syn_alive[:ns]
    weights = projection.syn_weight[:ns][live].to(torch.float64)
    count = weights.numel()
    lower, upper = projection.syn_min_weight[:ns][live], projection.syn_max_weight[:ns][live]
    result = {
        "live_synapses": count, "at_min": int((weights == lower).sum()),
        "at_max": int((weights == upper).sum()),
        "mean": float(weights.mean()) if count else None,
        "std": float(weights.std(unbiased=False)) if count else None,
        "min": float(weights.min()) if count else None,
        "max": float(weights.max()) if count else None,
    }
    if reference is not None:
        if ns != reference.n_synapses or any(not torch.equal(getattr(projection, key)[:ns],
                                                              getattr(reference, key)[:ns])
                                            for key in ("syn_pre", "syn_post", "syn_alive")):
            raise ValueError("Weight comparison requires identical live topology")
        old = reference.syn_weight[:ns][live].to(torch.float64)
        delta = weights - old
        denominator = float(old.abs().sum())
        result.update(l1_change_from_reference=float(delta.abs().sum()),
                      relative_l1_change_from_reference=float(delta.abs().sum()) / denominator
                      if denominator else None,
                      changed_weights=int(torch.count_nonzero(delta)))
    return result


def training_protocol(dataset, config, seed, condition, source_hashes):
    return {
        "config": asdict(config), "seed": seed, "condition": condition,
        "train_sha256": dataset.manifest["train_sha256"],
        "source_sha256": source_hashes,
    }


def train_condition(brain, dataset, config, seed, condition, root, source_hashes,
                    *, resume_from=None, stop_after=None):
    """Train or resume at a completed image boundary, preserving the initial norm target."""
    config.validate()
    if condition not in CONDITIONS[1:]:
        raise ValueError("Training requires a normalized control or STDP condition")
    protocol = training_protocol(dataset, config, seed, condition, source_hashes)
    initial_hash = simulation_digest(brain)
    if resume_from is None:
        norm_target = mn.compute_norm_target(brain, per_neuron=config.normalization_policy == "initial_per_neuron")
        if isinstance(norm_target, torch.Tensor):
            norm_target = norm_target.cpu().tolist()
        progress = {"protocol": protocol, "completed_samples": 0,
                    "norm_target": norm_target, "weight_deltas": [],
                    "initial_sha256": initial_hash, "initial_step_count": brain.step_count,
                    "initial_time": brain.time}
    else:
        brain, progress = load_training_checkpoint(resume_from, protocol)
        if progress["initial_sha256"] != initial_hash:
            raise ValueError("Training checkpoint initial state mismatch")
    if (brain.reward_stdp.enabled or brain.homeostasis.scaling_enabled
            or brain.memory.enabled or brain.oscillators.enabled
            or brain.metaplasticity_enabled
            or brain.growth.growth_interval != mn.DISABLED_GROWTH_INTERVAL):
        raise ValueError("Learning controls require fixed topology and reward/scaling/memory/oscillations off")
    total_samples = len(dataset.train_X) * config.epochs
    completed = progress["completed_samples"]
    if type(completed) is not int or not 0 <= completed <= total_samples:
        raise ValueError("Invalid training checkpoint sample position")
    if len(progress["weight_deltas"]) != completed:
        raise ValueError("Training checkpoint diagnostics/cursor mismatch")
    elapsed_steps = completed * (config.train_steps + config.rest_steps)
    if (brain.step_count != progress["initial_step_count"] + elapsed_steps
            or not math.isclose(brain.time, progress["initial_time"] + elapsed_steps * brain.dt,
                                rel_tol=1e-10, abs_tol=1e-9)):
        raise ValueError("Training checkpoint sample position/state mismatch")
    final = total_samples if stop_after is None else min(total_samples, stop_after)
    if final < completed:
        raise ValueError("Stop position precedes the resumed checkpoint")
    projection = brain.get_projection("input", "cortex")
    weights = projection.syn_weight[:projection.n_synapses]
    timer = time.perf_counter()
    previous_elapsed = progress.get("training_wall_s", 0.0)
    order = None
    for position in range(completed, final):
        epoch, offset = divmod(position, len(dataset.train_X))
        if order is None or offset == 0:
            order = np.random.default_rng(seed + epoch).permutation(len(dataset.train_X))
        before = weights.clone()
        mn.present_sample(brain, dataset.train_X[order[offset]], config.train_steps,
                          learn=condition == "stdp_normalized", collect_responses=False,
                          learning_rule=config.learning_rule, trace_tau=config.trace_tau,
                          trace_target=config.trace_target)
        after_stdp = weights.clone()
        mn.normalize_feedforward_weights(brain, progress["norm_target"])
        progress["weight_deltas"].append(weight_delta_metrics(before, after_stdp, weights))
        mn.reset_brain_state(brain, rest_steps=config.rest_steps)
        progress["completed_samples"] = position + 1
        if (position + 1) % config.checkpoint_every == 0 or position + 1 == final:
            progress["training_wall_s"] = previous_elapsed + time.perf_counter() - timer
            destination = Path(root) / f"sample_{position + 1:08d}"
            save_training_checkpoint(brain, destination, progress)
            print(f"  {condition} seed={seed} sample={position + 1}/{total_samples}", flush=True)
    return brain, progress


def ridge_predictions(train_features, labels, validation_features, alpha):
    # Both standardization and the classifier see labelled readout samples only.
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="cholesky"))
    model.fit(train_features, labels)
    return model.predict(validation_features), model


def probe_decoders(readout, y_readout, validation, y_validation, config, classes):
    templates = mn._templates_from_responses(readout, y_readout, classes)
    predictions = []
    for spike, voltage in zip(validation.spikes, validation.voltages):
        counts, voltage_sum = np.zeros_like(spike, dtype=np.float32), np.zeros_like(voltage)
        for _ in range(mn.TEST_REPEATS):
            counts += spike
            voltage_sum += voltage
        predictions.append(mn.classify_response(counts, voltage_sum, *templates, classes)[0])
    all_predictions = {"template": np.array(predictions)}
    for decoder, left, right in (
        ("ridge_spikes", readout.spikes, validation.spikes),
        ("ridge_spikes_voltage", np.concatenate((readout.spikes, readout.voltages), axis=1),
         np.concatenate((validation.spikes, validation.voltages), axis=1)),
        ("ridge_spikes_centered_voltage", readout_features(readout, "voltage_centered"),
         readout_features(validation, "voltage_centered")),
    ):
        all_predictions[decoder], _ = ridge_predictions(left, y_readout, right, config.ridge_alpha)
    return {name: {"accuracy": float(np.mean(preds == y_validation)), "predictions": preds.tolist()}
            for name, preds in all_predictions.items()}


def cached_responses(brain, dataset, config, path):
    """Extract each image once on frozen copies; cached arrays cannot train the SNN."""
    source = simulation_digest(brain)
    identity = {
        "brain_sha256": source, "inference_steps": config.inference_steps,
        "readout_sha256": dataset.manifest["readout_sha256"],
        "validation_sha256": dataset.manifest["validation_sha256"],
    }
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored["identity"])) != identity:
                raise ValueError("Response cache identity mismatch")
            left = mn.ReadoutResponses(stored["exc_indices"], stored["readout_spikes"], stored["readout_voltages"])
            right = mn.ReadoutResponses(stored["exc_indices"], stored["validation_spikes"], stored["validation_voltages"])
        return left, right
    with patch.object(mn, "ASSIGN_PRESENT_STEPS", config.inference_steps), \
            patch.object(mn, "INDEPENDENT_INFERENCE", True), \
            patch.object(mn, "FAST_INDEPENDENT_INFERENCE", True):
        left = mn.collect_readout_responses(mn._inference_brain(brain),
                                           dataset.train_X[dataset.readout_indices])
        right = mn.collect_readout_responses(mn._inference_brain(brain), dataset.validation_X)
    assert simulation_digest(brain) == source
    np.testing.assert_array_equal(left.exc_indices, right.exc_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".responses-", dir=path.parent) as folder:
        staged = Path(folder) / "responses.npz"
        np.savez_compressed(staged, identity=json.dumps(identity), exc_indices=left.exc_indices,
                            readout_spikes=left.spikes, readout_voltages=left.voltages,
                            validation_spikes=right.spikes, validation_voltages=right.voltages,
                            readout_y=dataset.train_y[dataset.readout_indices], validation_y=dataset.validation_y)
        os.replace(staged, path)
    return left, right


def comparison_summary(results, seeds):
    rows = {(r["seed"], r["condition"]): r for r in results}
    summary = {}
    for decoder in DECODERS:
        paired = []
        for seed in seeds:
            # Older studies have only the original three decoders.
            if all((seed, c) in rows and decoder in rows[seed, c]["decoders"] for c in CONDITIONS[1:]):
                trained = rows[seed, "stdp_normalized"]["decoders"][decoder]["accuracy"]
                control = rows[seed, "normalization_only"]["decoders"][decoder]["accuracy"]
                paired.append({"seed": seed, "stdp_gain_pp": 100.0 * (trained - control)})
        summary[decoder] = {"paired_seeds": paired, "mean_stdp_gain_pp":
                            float(np.mean([r["stdp_gain_pp"] for r in paired])) if paired else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    for name, field in LearningConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args()
    config = LearningConfig(**{name: getattr(args, name) for name in LearningConfig.__dataclass_fields__})
    config.validate()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    if args.output.exists() != args.resume:
        parser.error("Use a new output directory, or --resume with the original arguments")
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/training_checkpoint.py", root / "examples/mnist_benchmark.py",
               root / "examples/mnist_optimization_check.py", root / "examples/_utils.py",
               root / "examples/mnist_diagnosis.py", root / "examples/mnist_state_diagnosis.py",
               root / "examples/mnist_trace_stdp.py",
               root / "examples/mnist_readout_features.py",
               *sorted((root / "src").glob("*.py"))]
    source_bytes = {str(p.relative_to(root)): p.read_bytes() for p in sources}
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    signature = {"config": asdict(config), "seeds": args.seeds, "source_sha256": source_hashes,
                 "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}
    if args.resume:
        payload = json.loads((args.output / "summary.json").read_text())
        if payload["signature"] != signature:
            raise ValueError("Cannot resume with changed code, dependencies, seeds or configuration")
        payload["pid"] = os.getpid()
        payload["resumed_at"] = time.time()
    else:
        args.output.mkdir(parents=True)
        for name, content in source_bytes.items():
            destination = args.output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        payload = {"signature": signature, "pid": os.getpid(), "started_at": time.time(),
                   "complete": False, "results": [], "interpretation":
                   "Screening on a fixed train-only validation split. No canonical test evaluation. "
                   "Initialization comparisons are descriptive, not a significance claim. "
                   "Ridge probes train a labelled decoder, not the SNN."}
        atomic_json(args.output / "summary.json", payload)
    print(f"LEARNING CHECK pid={os.getpid()} seeds={args.seeds}", flush=True)
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    if "dataset" in payload and payload["dataset"] != data.manifest:
        raise ValueError("Cannot resume with changed dataset rows or values")
    payload["dataset"] = data.manifest
    pixel_preds, _ = ridge_predictions(data.train_X[data.readout_indices], data.train_y[data.readout_indices],
                                      data.validation_X, config.ridge_alpha)
    payload["pixel_ridge_baseline"] = {"accuracy": float(np.mean(pixel_preds == data.validation_y)),
                                       "predictions": pixel_preds.tolist()}
    atomic_json(args.output / "summary.json", payload)
    for seed in args.seeds:
        initial = mn.build_brain(seed=seed, exc_to_inh_weight=config.exc_to_inh_weight,
                                 stdp_scale=config.stdp_scale, input_density=config.input_density,
                                 input_weight_boost=config.input_weight_boost)
        initial_hash = simulation_digest(initial)
        # Alternate the two trained conditions to avoid a fixed timing order.
        conditions = CONDITIONS if seed % 2 else (CONDITIONS[0], CONDITIONS[2], CONDITIONS[1])
        for condition in conditions:
            if any(r["seed"] == seed and r["condition"] == condition for r in payload["results"]):
                continue
            start = time.perf_counter()
            print(f"START seed={seed} condition={condition}", flush=True)
            folder = args.output / f"seed_{seed}" / condition
            checkpoints = folder / "checkpoints"
            model = copy.deepcopy(initial)
            progress = None
            if condition == "initial":
                checkpoint = checkpoints / "sample_00000000"
                if not checkpoint.exists():
                    save_training_checkpoint(model, checkpoint, {"protocol":
                        training_protocol(data, config, seed, condition, source_hashes), "completed_samples": 0})
            else:
                existing = sorted(checkpoints.glob("sample_*/progress.json"))
                model, progress = train_condition(model, data, config, seed, condition, checkpoints,
                                                   source_hashes,
                                                   resume_from=existing[-1].parent if existing else None)
            readout, validation = cached_responses(model, data, config, folder / "responses.npz")
            result = {
                "seed": seed, "condition": condition, "initial_sha256": initial_hash,
                "final_sha256": simulation_digest(model),
                "training_wall_s": 0.0 if progress is None else progress["training_wall_s"],
                "condition_this_invocation_s": time.perf_counter() - start,
                "decoders": probe_decoders(readout, data.train_y[data.readout_indices], validation,
                                           data.validation_y, config, mn.CLASSES),
                "validation_silent_samples": int(np.sum(validation.spikes.sum(axis=1) == 0)),
                "completed_samples": 0 if progress is None else progress["completed_samples"],
                "feedforward_weights": weight_summary(model.get_projection("input", "cortex"),
                                                         initial.get_projection("input", "cortex")),
            }
            if progress:
                atomic_json(folder / "weight_deltas.json", progress["weight_deltas"])
            atomic_json(folder / "result.json", result)
            payload["results"].append(result)
            payload["comparison"] = comparison_summary(payload["results"], args.seeds)
            atomic_json(args.output / "summary.json", payload)
            print(f"END seed={seed} condition={condition}: " + json.dumps(
                {d: v["accuracy"] for d, v in result["decoders"].items()}), flush=True)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
           or hashlib.sha256((args.output / "source_snapshot" / name).read_bytes()).hexdigest() != digest
           for name, digest in source_hashes.items()):
        raise RuntimeError("Learning-check source or snapshot changed during execution")
    payload["source_unchanged"] = True
    payload["complete"] = True
    payload["finished_at"] = time.time()
    atomic_json(args.output / "summary.json", payload)
    print("COMPLETE " + json.dumps(payload["comparison"]), flush=True)


if __name__ == "__main__":
    main()
