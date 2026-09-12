"""Verify and time repeated Iris inference on a selected, saved checkpoint.

Both timing arms use independent deterministic samples. The slow arm simulates
every requested repetition; the fast arm reuses the identical response. An
optional final test evaluation uses the validation-selected checkpoint as-is.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np

import examples.iris_benchmark as iris
from examples.training_checkpoint import array_digest, atomic_json, directory_digest, load_training_checkpoint


def encoded_partition(study, partition):
    raw = iris.load_iris()
    dataset = study["dataset"]
    ids = dataset[partition + "_ids"]
    X = iris.place_field_encode(iris.normalize_features(raw.data[ids],
        np.asarray(dataset["normalization_lo"]), np.asarray(dataset["normalization_hi"])))
    y = raw.target[ids]
    if array_digest(X, y) != dataset[partition + "_sha256"]:
        raise ValueError("Recorded Iris dataset or preprocessing differs")
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    root = Path(__file__).resolve().parent.parent
    study_bytes = (args.study / "summary.json").read_bytes()
    study = json.loads(study_bytes)
    if not study["complete"]:
        parser.error("The source benchmark must be complete")
    if args.evaluate_test and not study["validation_only"]:
        parser.error("This source study already scored its test partition")
    sources = sorted(set(study["source_sha256"]) | {
        str(Path(__file__).resolve().relative_to(root)), "examples/training_checkpoint.py"})
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    if any(hashes[name] != digest for name, digest in study["source_sha256"].items()):
        raise ValueError("The benchmark code differs from the source study")
    checkpoint = args.study / "checkpoints/selected"
    checkpoint_hash = directory_digest(checkpoint)
    protocol = {"config": study["config"], "source_sha256": study["source_sha256"],
                "dataset": study["dataset"]}
    brain, progress = load_training_checkpoint(checkpoint, protocol)
    if progress["selected_by"] != "validation" or progress["completed_epochs"] != study["best_epoch"]:
        raise ValueError("The checkpoint is not the validation-selected epoch")
    # A saved copy also checks all tensors, counters and RNGs after evaluation.
    selected_state = copy.deepcopy(brain)
    result = {"complete": False, "study": str(args.study.resolve()),
              "study_sha256": hashlib.sha256(study_bytes).hexdigest(),
              "source_sha256": hashes, "checkpoint_sha256": checkpoint_hash,
              "selected_epoch": study["best_epoch"], "timing": [],
              "interpretation": "ABBA within the independent frozen protocol, not the former "
              "sequential-image protocol. Test scoring, if requested, does not tune the model."}
    with patch.multiple(iris, **study["config"]):
        X, y = encoded_partition(study, "validation")
        expected = None
        for fast in (False, True, True, False):
            start = time.perf_counter()
            evaluated = iris.evaluation_payload(iris.evaluate(brain, X, y, fast=fast))
            elapsed = time.perf_counter() - start
            if expected is None:
                expected = evaluated
            if evaluated != expected:
                raise AssertionError("Repeated and reused inference results differ")
            if evaluated != study["epochs"][study["best_epoch"] - 1]["validation"]:
                raise AssertionError("Validation does not reproduce the selected epoch")
            row = {"fast": fast, "wall_s": elapsed}
            result["timing"].append(row)
            print("INFERENCE " + json.dumps(row), flush=True)
            atomic_json(args.output, result)
        result["validation"] = expected
        if args.evaluate_test:
            Xtest, ytest = encoded_partition(study, "test")
            result["test"] = iris.evaluation_payload(iris.evaluate(brain, Xtest, ytest))
            Xtrain, ytrain = encoded_partition(study, "train")
            baseline = iris.LogisticRegression(max_iter=1000, random_state=iris.SEED).fit(Xtrain, ytrain)
            result["baseline_test_accuracy"] = float(baseline.score(Xtest, ytest))
            result["test_ids"] = study["dataset"]["test_ids"]
    # Use the serializer's full-state fingerprints without depending on MNIST.
    import tempfile
    from src.persistence import save_brain
    with tempfile.TemporaryDirectory(prefix="iris-inference-state-") as temporary:
        left, right = Path(temporary) / "before", Path(temporary) / "after"
        save_brain(selected_state, left)
        save_brain(brain, right)
        if directory_digest(left) != directory_digest(right):
            raise AssertionError("Inference changed the source network or RNG state")
    assert checkpoint_hash == directory_digest(checkpoint)
    assert study_bytes == (args.study / "summary.json").read_bytes()
    assert hashes == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    timing = {flag: statistics.median(row["wall_s"] for row in result["timing"] if row["fast"] == flag)
              for flag in (False, True)}
    result.update(complete=True, source_unchanged=True, counts_and_predictions_equal=True,
                  inference_speedup=timing[False] / timing[True])
    atomic_json(args.output, result)
    print("COMPLETE " + json.dumps({"speedup": result["inference_speedup"],
                                   "test": result.get("test", {}).get("spike_accuracy")}), flush=True)


if __name__ == "__main__":
    main()
