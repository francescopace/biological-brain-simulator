"""Measure fixed-budget readout sensitivity on frozen, completed MNIST studies.

Plan subsets before scoring; extract the training pool once per frozen network.
The validation cache is reused. Subset percentiles are sensitivity ranges, not
confidence intervals, and overlapping subsets are not independent replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from examples import mnist_benchmark as mn
from examples.mnist_learning_check import CONDITIONS, LearningConfig, prepare_dataset, ridge_predictions
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_readout_features import readout_features
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


DECODERS = ("ridge_spikes", "ridge_spikes_centered_voltage")
REPO = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_subsets(labels, *, repeats=100, per_class=10, seed=20260917):
    """Sample without replacement within a draw; draws may overlap."""
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu"
            or type(repeats) is not int or repeats < 1
            or type(per_class) is not int or per_class < 1):
        raise ValueError("Expected integer labels and positive integer subset settings")
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    if any(len(group) < per_class for group in groups):
        raise ValueError("Insufficient training rows for the fixed per-class budget")
    rng = np.random.default_rng(seed)
    return np.array([np.concatenate([rng.choice(g, per_class, replace=False) for g in groups])
                     for _ in range(repeats)], dtype=np.int64)


def feature_matrix(response, decoder):
    if decoder == "ridge_spikes":
        return response.spikes
    if decoder == "ridge_spikes_centered_voltage":
        return readout_features(response, "voltage_centered")
    raise ValueError(f"Unknown decoder: {decoder}")


def verify_original_readout(pool, original, indices):
    np.testing.assert_array_equal(pool.exc_indices, original.exc_indices)
    for attr in ("spikes", "voltages"):
        actual, expected = getattr(pool, attr)[indices], getattr(original, attr)
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.tobytes() != expected.tobytes()):
            raise ValueError(f"Original readout {attr} bytes did not reproduce")


def summarize_counts(counts, validation_size):
    """Keep seeds paired within each subset; never pool them as independent runs."""
    arrays = {key: np.asarray(value) for key, value in counts.items()}
    if (set(arrays) != set(CONDITIONS) or type(validation_size) is not int
            or validation_size < 1):
        raise ValueError("Expected all conditions and a positive validation size")
    shape = arrays["initial"].shape
    if (len(shape) != 2 or not all(shape) or any(
            a.shape != shape or a.dtype.kind not in "iu" or np.any(a < 0)
            or np.any(a > validation_size) for a in arrays.values())):
        raise ValueError("Expected aligned seed-by-subset integer correct counts")
    result = {"mean_accuracy": {c: float(a.mean() / validation_size) for c, a in arrays.items()},
              "contrasts": {}}
    for control in ("normalization_only", "initial"):
        # Integer subtraction preserves exact ties when counting positive draws.
        delta = arrays["stdp_normalized"].astype(np.int64) - arrays[control].astype(np.int64)
        gain = delta.mean(axis=0) * (100. / validation_size)
        result["contrasts"][control] = {
            "mean_gain_pp": float(gain.mean()),
            "subset_mean_gains_pp": gain.tolist(),
            "subset_mean_gain_5_95_percentiles_pp": np.quantile(gain, [.05, .95]).tolist(),
            "positive_subsets": int((delta.sum(axis=0) > 0).sum()),
            "zero_subsets": int((delta.sum(axis=0) == 0).sum()),
            "negative_subsets": int((delta.sum(axis=0) < 0).sum()),
            "all_seeds_positive_subsets": int((delta > 0).all(axis=0).sum()),
            "per_seed_mean_gain_pp": (delta.mean(axis=1) * (100. / validation_size)).tolist(),
            "per_seed_positive_subsets": (delta > 0).sum(axis=1).tolist(),
        }
    return result


def load_dataset(study):
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(REPO / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64),
        LearningConfig(**study["signature"]["config"]),
        excluded_ids=study["signature"].get("excluded_ids", []))
    assert data.manifest == study["dataset"]
    return data


def make_plan(study_root, output, seeds):
    if output.exists():
        raise ValueError("Use a new output directory")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected distinct seeds")
    paths = [study_root / f"seed_{seed}" / "summary.json" for seed in seeds]
    studies = [json.loads(p.read_text()) for p in paths]
    for seed, study in zip(seeds, studies):
        assert study["complete"] and study["source_unchanged"]
        assert study["signature"]["seeds"] == [seed]
        assert study["dataset"] == studies[0]["dataset"]
        assert study["signature"]["config"] == studies[0]["signature"]["config"]
    data = load_dataset(studies[0])
    assert len(data.train_y) == 1000 and len(data.validation_y) == 800
    assert len(data.readout_indices) == 100
    np.testing.assert_array_equal(np.bincount(data.train_y), np.full(10, 100))
    np.testing.assert_array_equal(np.bincount(data.validation_y), np.full(10, 80))
    assert not set(data.manifest["train_ids"]) & set(data.manifest["validation_ids"])
    subsets = balanced_subsets(data.train_y)
    names = set(studies[0]["signature"]["source_sha256"]) | {
        "examples/mnist_readout_subset_check.py", "examples/mnist_readout_features.py"}
    hashes = {name: sha256(REPO / name) for name in sorted(names)}
    for path, study in zip(paths, studies):
        for name, digest in study["signature"]["source_sha256"].items():
            assert hashes[name] == digest == sha256(path.parent / "source_snapshot" / name)
    output.mkdir(parents=True)
    for name in names:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / name).read_bytes())
    plan = {"created_at": time.time(), "seeds": seeds, "conditions": list(CONDITIONS),
        "decoders": list(DECODERS), "source_sha256": hashes,
        "studies": {str(s): {"path": str(p.resolve()), "sha256": sha256(p)} for s, p in zip(seeds, paths)},
        "config": studies[0]["signature"]["config"], "dataset": data.manifest,
        "train_y": data.train_y.tolist(), "validation_y": data.validation_y.tolist(),
        "original_readout_indices": data.readout_indices.tolist(),
        "subset_seed": 20260917, "repeats": 100, "labels_per_fit": 100, "labels_per_digit": 10,
        "subset_indices": subsets.tolist(),
        "subset_row_ids": np.asarray(data.manifest["train_ids"])[subsets].tolist(),
        "backend": "cpu", "primary": "ridge_spikes: stdp_normalized minus normalization_only",
        "interpretation": "Post-hoc fixed-budget sensitivity, not fresh confirmation. Each decoder uses 100 labels from the existing 1000 training rows; the collection of fits can use all 1000 labels. The same 800 validation rows, subsets and settings are shared across fixed seeds and conditions. Subsets can overlap; percentiles are sensitivity ranges, not confidence intervals or probabilities of generalization. No subset or hyperparameter selection, no SNN training, no canonical test evaluation, no default change."}
    atomic_json(output / "plan.json", plan)
    print(f"PLANNED {len(subsets)} matched 100-label subsets for seeds {seeds}", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    for name, digest in plan["source_sha256"].items():
        assert sha256(REPO / name) == digest == sha256(output / "source_snapshot" / name)
    for study in plan["studies"].values():
        assert sha256(study["path"]) == study["sha256"]
    subsets = balanced_subsets(plan["train_y"], repeats=plan["repeats"],
        per_class=plan["labels_per_digit"], seed=plan["subset_seed"])
    assert subsets.tolist() == plan["subset_indices"]
    assert np.asarray(plan["dataset"]["train_ids"])[subsets].tolist() == plan["subset_row_ids"]
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    if seed not in plan["seeds"]:
        raise ValueError("Seed not in the saved plan")
    destination = output / f"seed_{seed}"
    destination.mkdir()  # A partial or completed run is never overwritten.
    plan_hash = sha256(output / "plan.json")
    study_path = Path(plan["studies"][str(seed)]["path"])
    study = json.loads(study_path.read_text())
    data = load_dataset(study)
    assert data.train_y.tolist() == plan["train_y"]
    assert data.validation_y.tolist() == plan["validation_y"]
    config = LearningConfig(**plan["config"])
    assert config.ridge_alpha == 1.
    assert torch.get_num_threads() == 1
    rows = {r["condition"]: r for r in study["results"] if r["seed"] == seed}
    assert set(rows) == set(CONDITIONS)
    subsets = np.asarray(plan["subset_indices"], dtype=np.int64)
    payload = {"complete": False, "seed": seed, "started_at": time.time(),
               "plan_sha256": plan_hash, "results": {}}
    atomic_json(destination / "summary.json", payload)
    for condition in CONDITIONS:
        started = time.monotonic()
        row = rows[condition]
        folder = study_path.parent / f"seed_{seed}" / condition
        checkpoint = folder / "checkpoints" / f"sample_{row['completed_samples']:08d}"
        metadata = json.loads((checkpoint / "progress.json").read_text())
        checkpoint_hash = directory_digest(checkpoint)
        brain, _ = load_training_checkpoint(checkpoint, metadata["progress"]["protocol"])
        state = simulation_digest(brain)
        assert state == row["final_sha256"]
        assert all(r.v.device.type == "cpu" for r in brain.regions.values())
        source_cache = folder / "responses.npz"
        cache_hash = sha256(source_cache)
        with np.load(source_cache, allow_pickle=False) as cache:
            assert json.loads(str(cache["identity"])) == {
                "brain_sha256": state, "inference_steps": config.inference_steps,
                "readout_sha256": data.manifest["readout_sha256"],
                "validation_sha256": data.manifest["validation_sha256"]}
            original = mn.ReadoutResponses(cache["exc_indices"], cache["readout_spikes"], cache["readout_voltages"])
            validation = mn.ReadoutResponses(cache["exc_indices"], cache["validation_spikes"], cache["validation_voltages"])
            np.testing.assert_array_equal(cache["readout_y"], data.train_y[data.readout_indices])
            np.testing.assert_array_equal(cache["validation_y"], data.validation_y)
        print(f"EXTRACT seed={seed} {condition}: 1000 frozen training-pool responses", flush=True)
        with patch.object(mn, "ASSIGN_PRESENT_STEPS", config.inference_steps), \
                patch.object(mn, "INDEPENDENT_INFERENCE", True), \
                patch.object(mn, "FAST_INDEPENDENT_INFERENCE", True):
            pool = mn.collect_readout_responses(mn._inference_brain(brain), data.train_X)
        assert simulation_digest(brain) == state
        verify_original_readout(pool, original, data.readout_indices)
        pool_path = destination / f"{condition}-pool.npz"
        np.savez_compressed(pool_path, exc_indices=pool.exc_indices, spikes=pool.spikes,
            voltages=pool.voltages, labels=data.train_y, row_ids=data.manifest["train_ids"],
            identity=json.dumps({"plan_sha256": plan_hash, "brain_sha256": state,
                                 "train_sha256": data.manifest["train_sha256"]}))
        print(f"VERIFIED seed={seed} {condition}: original readout bytes; fit 100 paired subsets", flush=True)
        scores, predictions = {}, {}
        for decoder in DECODERS:
            train, valid = feature_matrix(pool, decoder), feature_matrix(validation, decoder)
            baseline, _ = ridge_predictions(train[data.readout_indices],
                data.train_y[data.readout_indices], valid, config.ridge_alpha)
            assert baseline.tolist() == row["decoders"][decoder]["predictions"]
            fitted = np.asarray([ridge_predictions(train[indices], data.train_y[indices],
                valid, config.ridge_alpha)[0] for indices in subsets])
            counts = (fitted == data.validation_y).sum(axis=1)
            scores[decoder] = {"correct_counts": counts.tolist(),
                "mean_accuracy": float(counts.mean() / len(data.validation_y)),
                "original_readout_correct": int((baseline == data.validation_y).sum())}
            predictions[decoder] = fitted
            predictions[decoder + "_original"] = baseline
        prediction_path = destination / f"{condition}-predictions.npz"
        np.savez_compressed(prediction_path, **predictions, validation_y=data.validation_y)
        assert simulation_digest(brain) == state
        assert directory_digest(checkpoint) == checkpoint_hash
        assert sha256(source_cache) == cache_hash
        payload["results"][condition] = {"decoders": scores, "source_state_sha256": state,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "cache_path": str(source_cache), "cache_sha256": cache_hash,
            "pool_sha256": sha256(pool_path), "predictions_sha256": sha256(prediction_path),
            "original_response_bytes_equal": True, "original_predictions_equal": True,
            "source_unchanged": True, "elapsed_seconds": time.monotonic() - started}
        atomic_json(destination / "summary.json", payload)
        print(f"END seed={seed} {condition}: " + json.dumps(
            {d: s["mean_accuracy"] for d, s in scores.items()}), flush=True)
        del brain
    assert sha256(output / "plan.json") == plan_hash
    checked_plan(output)
    payload.update(complete=True, source_unchanged=True, completed_at=time.time())
    atomic_json(destination / "summary.json", payload)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output / "summary.json").exists():
        raise ValueError("Summary already exists")
    plan = checked_plan(output)
    summaries = []
    for seed in plan["seeds"]:
        folder = output / f"seed_{seed}"
        result = json.loads((folder / "summary.json").read_text())
        assert result["seed"] == seed and result["complete"] and result["source_unchanged"]
        assert result["plan_sha256"] == sha256(output / "plan.json")
        assert result["started_at"] >= plan["created_at"]
        for condition in CONDITIONS:
            row = result["results"][condition]
            assert row["source_unchanged"] and row["original_response_bytes_equal"] and row["original_predictions_equal"]
            assert directory_digest(Path(row["checkpoint_path"])) == row["checkpoint_sha256"]
            assert sha256(row["cache_path"]) == row["cache_sha256"]
            assert sha256(folder / f"{condition}-pool.npz") == row["pool_sha256"]
            predictions_path = folder / f"{condition}-predictions.npz"
            assert sha256(predictions_path) == row["predictions_sha256"]
            with np.load(predictions_path, allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions["validation_y"], plan["validation_y"])
                for decoder in DECODERS:
                    values = predictions[decoder]
                    assert values.shape == (plan["repeats"], len(plan["validation_y"]))
                    assert (values == predictions["validation_y"]).sum(axis=1).tolist() == row["decoders"][decoder]["correct_counts"]
                    assert int((predictions[decoder + "_original"] == predictions["validation_y"]).sum()) == row["decoders"][decoder]["original_readout_correct"]
        summaries.append(result)
    result = {"complete": True, "plan_sha256": sha256(output / "plan.json"),
        "seeds": plan["seeds"], "subsets": plan["repeats"],
        "interpretation": plan["interpretation"], "source_unchanged": True, "decoders": {}}
    for decoder in DECODERS:
        counts = {c: [s["results"][c]["decoders"][decoder]["correct_counts"] for s in summaries] for c in CONDITIONS}
        baseline = {c: [[s["results"][c]["decoders"][decoder]["original_readout_correct"]] for s in summaries] for c in CONDITIONS}
        result["decoders"][decoder] = summarize_counts(counts, len(plan["validation_y"]))
        result["decoders"][decoder]["original_readout"] = summarize_counts(baseline, len(plan["validation_y"]))
    atomic_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        if args.study_root is None:
            parser.error("plan requires --study-root")
        make_plan(args.study_root, args.output, args.seeds)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
