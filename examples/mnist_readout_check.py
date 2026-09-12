"""Verify voltage centering on additional, disjoint MNIST validation images.

The two readouts and ridge alpha are fixed before extracting new responses.
SNNs are loaded from completed learning studies and never retrained. Fresh here
means disjoint from those studies' training/readout/validation rows, not from
every historical MNIST experiment. The canonical test split is excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import sklearn
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import Dataset, LearningConfig, cached_responses, prepare_dataset, ridge_predictions
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_readout_features import FEATURE_MODES as MODES, readout_features
from examples.training_checkpoint import array_digest, atomic_json, directory_digest, load_training_checkpoint


def additional_validation(raw_X, labels, original, *, per_class=80, seed=20260913, classes=tuple(range(10))):
    """Select unused canonical-training rows and reuse the fitted intensity target."""
    if type(per_class) is not int or per_class < 1:
        raise ValueError("Fresh validation samples per class must be positive integers")
    manifest = original.manifest
    boundary = manifest["canonical_train_boundary"]
    if len(raw_X) != len(labels) or len(labels) < boundary:
        raise ValueError("Dataset does not cover the canonical training split")
    train_ids, prior_ids = set(manifest["train_ids"]), set(manifest["validation_ids"])
    readout_ids = set(manifest["readout_ids"])
    excluded = train_ids | prior_ids | readout_ids
    if (train_ids & prior_ids or not readout_ids <= train_ids or not excluded
            or min(excluded) < 0 or max(excluded) >= boundary):
        raise ValueError("Invalid source study partitions")
    canonical_y = np.asarray(labels[:boundary], dtype=np.int64)
    eligible = np.ones(boundary, dtype=bool)
    eligible[list(excluded)] = False
    rng = np.random.default_rng(seed)
    ids = []
    for cls in classes:
        candidates = np.flatnonzero(eligible & (canonical_y == cls))
        if len(candidates) < per_class:
            raise ValueError(f"Not enough unused canonical training samples for class {cls}")
        rng.shuffle(candidates)
        ids.extend(candidates[:per_class])
    ids = np.asarray(ids, dtype=np.int64)
    rng.shuffle(ids)
    X = mn.downsample_images(np.asarray(raw_X[ids], dtype=np.float64) / 255.,
                              factor=mn.DOWNSAMPLE, target_l1=manifest["intensity_target_l1"])
    y = canonical_y[ids]
    fresh_manifest = dict(manifest, validation_ids=ids.tolist(), validation_sha256=array_digest(X, y),
                          prior_validation_ids=manifest["validation_ids"],
                          additional_validation_seed=seed, additional_validation_per_class=per_class,
                          excluded_ids=sorted(excluded))
    return Dataset(original.train_X, original.train_y, original.readout_indices, X, y, fresh_manifest)


def selected_studies(paths):
    """Require completed, matched studies with distinct network seeds."""
    selected, seen, expected = [], set(), None
    for path in paths:
        path = Path(path).resolve()
        source = (path / "summary.json").read_bytes()
        study = json.loads(source)
        if not study["complete"]:
            raise ValueError("Readout verification requires completed learning studies")
        config = LearningConfig(**study["signature"]["config"])
        config.validate()
        signature = (asdict(config), study["dataset"])
        if expected is not None and signature != expected:
            raise ValueError("Source studies must share configuration and dataset")
        expected = signature
        for seed in study["signature"]["seeds"]:
            if seed in seen:
                raise ValueError("Source studies must have distinct network seeds")
            seen.add(seed)
            rows = [row for row in study["results"]
                    if row["seed"] == seed and row["condition"] == "stdp_normalized"]
            if len(rows) != 1 or rows[0]["completed_samples"] != len(study["dataset"]["train_ids"]) * config.epochs:
                raise ValueError("Missing or incomplete trained condition")
            selected.append((path, study, seed, rows[0], hashlib.sha256(source).hexdigest()))
    if not selected:
        raise ValueError("At least one completed study is required")
    return selected


def verify_response_cache(path, brain, original, config, readout, validation):
    identity = {"brain_sha256": simulation_digest(brain), "inference_steps": config.inference_steps,
                "readout_sha256": original.manifest["readout_sha256"],
                "validation_sha256": original.manifest["validation_sha256"]}
    with np.load(path, allow_pickle=False) as cache:
        if json.loads(str(cache["identity"])) != identity:
            raise ValueError("Source response cache identity mismatch")
        pairs = (("exc_indices", readout.exc_indices), ("readout_spikes", readout.spikes),
                 ("readout_voltages", readout.voltages), ("validation_spikes", validation.spikes),
                 ("validation_voltages", validation.voltages),
                 ("readout_y", original.train_y[original.readout_indices]),
                 ("validation_y", original.validation_y))
        for name, value in pairs:
            cached = cache[name]
            if cached.shape != value.shape or cached.dtype != value.dtype or cached.tobytes() != value.tobytes():
                raise AssertionError(f"Current inference does not reproduce source {name}")
    np.testing.assert_array_equal(readout.exc_indices, validation.exc_indices)


def compare_readouts(readout, labels, prior, prior_y, fresh, fresh_y, alpha, recorded_predictions):
    scores, predictions = {}, {}
    for mode in MODES:
        prior_predictions, classifier = ridge_predictions(readout_features(readout, mode), labels,
                                                          readout_features(prior, mode), alpha)
        if mode == "standard" and prior_predictions.tolist() != recorded_predictions:
            raise AssertionError("Standard ridge predictions differ from the source study")
        fresh_predictions = classifier.predict(readout_features(fresh, mode))
        predictions[mode] = fresh_predictions
        scores[mode] = {"prior_accuracy": float(np.mean(prior_predictions == prior_y)),
                        "fresh_accuracy": float(np.mean(fresh_predictions == fresh_y)),
                        "fresh_predictions": fresh_predictions.tolist()}
    original_correct = predictions["standard"] == fresh_y
    centered_correct = predictions["voltage_centered"] == fresh_y
    paired = {"centered_wins": int((centered_correct & ~original_correct).sum()),
              "centered_losses": int((original_correct & ~centered_correct).sum()),
              "both_correct": int((original_correct & centered_correct).sum()),
              "both_wrong": int((~original_correct & ~centered_correct).sum()),
              "gain_pp": 100. * float(centered_correct.mean() - original_correct.mean())}
    return scores, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-per-class", type=int, default=80)
    parser.add_argument("--fresh-seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output.exists() or args.fresh_per_class < 1:
        parser.error("Use a new output directory and a positive fresh sample count")
    selected = selected_studies(args.studies)
    config = LearningConfig(**selected[0][1]["signature"]["config"])
    root = Path(__file__).resolve().parent.parent
    sources = [*sorted((root / "examples").glob("mnist*.py")), root / "examples/_utils.py",
               root / "examples/training_checkpoint.py", *sorted((root / "src").glob("*.py"))]
    source_bytes = {str(path.relative_to(root)): path.read_bytes() for path in sources}
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    original = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    if original.manifest != selected[0][1]["dataset"]:
        raise ValueError("Source data rows, values or fitted preprocessing changed")
    fresh = additional_validation(raw.data, raw.target, original, per_class=args.fresh_per_class,
                                  seed=args.fresh_seed)
    del raw
    args.output.mkdir(parents=True)
    for name, content in source_bytes.items():
        destination = args.output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    payload = {"complete": False, "pid": os.getpid(), "started_at": time.time(),
               "config": asdict(config), "modes": MODES, "source_sha256": source_hashes,
               "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__,
               "original_dataset": original.manifest, "fresh_dataset": fresh.manifest,
               "results": [], "interpretation": "Fixed decoder-only comparison on additional canonical-training "
               "rows disjoint from these studies' training, readout and prior validation. The same fresh "
               "images are reused across network seeds, not independent observations per seed. "
               "No SNN training, hyperparameter fitting on fresh labels, or canonical test evaluation."}
    atomic_json(args.output / "summary.json", payload)
    for path, study, seed, row, study_hash in selected:
        print(f"START seed={seed}: reproduce prior responses", flush=True)
        folder = path / f"seed_{seed}/stdp_normalized"
        checkpoint = folder / f"checkpoints/sample_{row['completed_samples']:08d}"
        checkpoint_hash = directory_digest(checkpoint)
        protocol = {"config": study["signature"]["config"], "seed": seed, "condition": "stdp_normalized",
                    "train_sha256": original.manifest["train_sha256"],
                    "source_sha256": study["signature"]["source_sha256"]}
        brain, progress = load_training_checkpoint(checkpoint, protocol)
        state = simulation_digest(brain)
        if state != row["final_sha256"] or progress["completed_samples"] != row["completed_samples"]:
            raise ValueError("Checkpoint does not match the completed trained condition")
        response_path = folder / "responses.npz"
        response_hash = hashlib.sha256(response_path.read_bytes()).hexdigest()
        destination = args.output / f"seed_{seed}"
        readout, prior = cached_responses(brain, original, config, destination / "prior_responses.npz")
        verify_response_cache(response_path, brain, original, config, readout, prior)
        print(f"  seed={seed}: prior bytes match; extract {len(fresh.validation_X)} fresh images", flush=True)
        again, validation = cached_responses(brain, fresh, config, destination / "fresh_responses.npz")
        np.testing.assert_array_equal(readout.spikes, again.spikes)
        np.testing.assert_array_equal(readout.voltages, again.voltages)
        scores, paired = compare_readouts(readout, original.train_y[original.readout_indices],
            prior, original.validation_y, validation, fresh.validation_y, config.ridge_alpha,
            row["decoders"]["ridge_spikes_voltage"]["predictions"])
        if (simulation_digest(brain) != state or directory_digest(checkpoint) != checkpoint_hash
                or hashlib.sha256(response_path.read_bytes()).hexdigest() != response_hash
                or hashlib.sha256((path / "summary.json").read_bytes()).hexdigest() != study_hash):
            raise AssertionError("Readout check changed or lost a source artifact/state")
        result = {"seed": seed, "study": str(path), "study_sha256": study_hash,
                  "checkpoint_sha256": checkpoint_hash, "response_cache_sha256": response_hash,
                  "state_rng_sha256": state, "source_unchanged": True, "prior_responses_bitwise_equal": True,
                  "scores": scores, "paired": paired,
                  "fresh_silent_samples": int((validation.spikes.sum(axis=1) == 0).sum())}
        atomic_json(destination / "result.json", result)
        payload["results"].append(result)
        atomic_json(args.output / "summary.json", payload)
        print(f"END seed={seed}: " + json.dumps(paired), flush=True)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
           or hashlib.sha256((args.output / "source_snapshot" / name).read_bytes()).hexdigest() != digest
           for name, digest in source_hashes.items()):
        raise RuntimeError("Readout-check source or snapshot changed during execution")
    payload["means"] = {mode: float(np.mean([row["scores"][mode]["fresh_accuracy"]
                                             for row in payload["results"]])) for mode in MODES}
    payload["mean_gain_pp"] = float(np.mean([row["paired"]["gain_pp"] for row in payload["results"]]))
    payload.update(complete=True, finished_at=time.time(), source_unchanged=True)
    atomic_json(args.output / "summary.json", payload)
    print("COMPLETE " + json.dumps(payload["means"]), flush=True)


if __name__ == "__main__":
    main()
