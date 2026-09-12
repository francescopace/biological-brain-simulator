"""Evaluate fixed old networks/readouts on a newer study's validation row IDs.

No SNN training, decoder refit on new rows, or parameter selection. The old
training-fitted intensity target stays fixed; old response bytes are replayed
before scoring the new rows. This is post-hoc diagnosis, not fresh confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import Dataset, LearningConfig, cached_responses, prepare_dataset, ridge_predictions
from examples.mnist_readout_features import readout_features
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import array_digest, atomic_json, directory_digest, load_training_checkpoint


def transferred_dataset(raw, labels, original, target):
    """Use target row IDs but retain all source training and preprocessing."""
    ids = target["validation_ids"]
    boundary = original.manifest["canonical_train_boundary"]
    excluded = set(original.manifest["train_ids"]) | set(original.manifest["validation_ids"])
    if (not ids or any(type(i) is not int or not 0 <= i < boundary for i in ids)
            or len(set(ids)) != len(ids) or set(ids) & excluded):
        raise ValueError("Target validation rows must be unique, train-only and unused by the source study")
    X = mn.downsample_images(np.asarray(raw[ids], dtype=np.float64) / 255.,
                            factor=mn.DOWNSAMPLE, target_l1=original.manifest["intensity_target_l1"])
    y = np.asarray(labels, dtype=np.int64)[ids]
    manifest = dict(original.manifest, validation_ids=ids, validation_sha256=array_digest(X, y),
                    prior_validation_ids=original.manifest["validation_ids"])
    return Dataset(original.train_X, original.train_y, original.readout_indices, X, y, manifest)


def features(response, decoder):
    return response.spikes if decoder == "ridge_spikes" else readout_features(response, "voltage_centered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--target-study", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    report_path, target_path = args.study / "summary.json", args.target_study / "summary.json"
    study, target_study = (json.loads(p.read_text()) for p in (report_path, target_path))
    assert study["complete"] and target_study["complete"]
    rows = {r["condition"]: r for r in study["results"] if r["seed"] == args.seed}
    assert set(rows) == {"initial", "normalization_only", "stdp_normalized"}
    config = LearningConfig(**study["signature"]["config"])
    repo = Path(__file__).resolve().parent.parent
    names = set(study["signature"]["source_sha256"]) | {
        "examples/mnist_readout_transfer_check.py", "examples/mnist_readout_features.py"}
    paths = [repo / name for name in names] + [report_path, target_path]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True)
    for name in names:
        destination = args.output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())
    payload = {"complete": False, "seed": args.seed, "source_study": str(args.study),
               "target_study": str(args.target_study), "source_sha256": hashes, "results": {},
               "interpretation": "Fixed source networks, 100 source labels, source-fitted preprocessing and decoder settings; target validation IDs only. Current source code must reproduce old response bytes. Post-hoc reuse of the new validation data, not an independent confirmation. No canonical test evaluation."}
    atomic_json(args.output / "summary.json", payload)
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(repo / ".sklearn_data"))
    original = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config,
                               excluded_ids=study["signature"].get("excluded_ids", []))
    assert original.manifest == study["dataset"]
    transfer = transferred_dataset(raw.data, raw.target, original, target_study["dataset"])
    del raw
    payload["transfer_dataset"] = transfer.manifest
    checkpoints = {}
    for condition in ("normalization_only", "stdp_normalized"):
        print(f"START seed={args.seed} {condition}: verify old responses", flush=True)
        row = rows[condition]
        folder = args.study / f"seed_{args.seed}" / condition
        checkpoint = folder / "checkpoints" / f"sample_{row['completed_samples']:08d}"
        metadata = json.loads((checkpoint / "progress.json").read_text())
        before = directory_digest(checkpoint)
        brain, _ = load_training_checkpoint(checkpoint, metadata["progress"]["protocol"])
        assert all(r.v.device.type == "cpu" for r in brain.regions.values())
        state = simulation_digest(brain)
        assert state == row["final_sha256"]
        checkpoints[str(checkpoint)] = before
        cache_path = folder / "responses.npz"
        cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        readout, old = cached_responses(brain, original, config, args.output / f"{condition}-prior.npz")
        with np.load(cache_path, allow_pickle=False) as cached:
            for prefix, response in (("readout", readout), ("validation", old)):
                for attr in ("spikes", "voltages"):
                    assert cached[f"{prefix}_{attr}"].tobytes() == getattr(response, attr).tobytes()
            assert np.array_equal(cached["exc_indices"], readout.exc_indices)
            assert np.array_equal(cached["readout_y"], original.train_y[original.readout_indices])
            assert np.array_equal(cached["validation_y"], original.validation_y)
        print(f"VERIFIED seed={args.seed} {condition}: evaluate {len(transfer.validation_y)} new rows", flush=True)
        again, new = cached_responses(brain, transfer, config, args.output / f"{condition}-target.npz")
        assert readout.spikes.tobytes() == again.spikes.tobytes()
        assert readout.voltages.tobytes() == again.voltages.tobytes()
        scores = {}
        for decoder in ("ridge_spikes", "ridge_spikes_centered_voltage"):
            old_predictions, classifier = ridge_predictions(features(readout, decoder),
                original.train_y[original.readout_indices], features(old, decoder), config.ridge_alpha)
            if decoder in row["decoders"]:
                assert old_predictions.tolist() == row["decoders"][decoder]["predictions"]
            predictions = classifier.predict(features(new, decoder))
            scores[decoder] = {"prior_accuracy": float((old_predictions == original.validation_y).mean()),
                               "target_accuracy": float((predictions == transfer.validation_y).mean()),
                               "target_predictions": predictions.tolist()}
        assert simulation_digest(brain) == state
        assert directory_digest(checkpoint) == before
        assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == cache_hash
        payload["results"][condition] = {"decoders": scores, "source_state_sha256": state,
            "silent_target_samples": int((new.spikes.sum(axis=1) == 0).sum()), "old_response_bytes_equal": True}
        atomic_json(args.output / "summary.json", payload)
        print("END " + json.dumps({k: v["target_accuracy"] for k, v in scores.items()}), flush=True)
    assert hashes == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in names:
        assert hashlib.sha256((args.output / "source_snapshot" / name).read_bytes()).hexdigest() == hashes[str((repo / name).resolve())]
    payload.update(complete=True, source_unchanged=True, checkpoint_sha256=checkpoints)
    atomic_json(args.output / "summary.json", payload)
    print(f"COMPLETE seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
