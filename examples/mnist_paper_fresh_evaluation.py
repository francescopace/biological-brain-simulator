"""Evaluate frozen reference networks and exported decoders on new MNIST rows.

No network training, decoder fitting, hyperparameter selection or canonical test
evaluation. Plan the data and contrasts, run each seed, then summarize.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from sklearn.datasets import fetch_openml

from examples.mnist_paper_readout_tuning import (
    CONDITIONS, METHODS, checked_plan as check_tuning, feature_matrix, predict_state, sha256,
)
from examples.mnist_paper_reference import ReferenceNetwork, frozen_responses
from examples.training_checkpoint import array_digest, atomic_json, directory_digest


REPO = Path(__file__).resolve().parent.parent
DECODERS = ("fixed_baseline", *METHODS, "class_average")
STATE_KEYS = ("mean", "scale", "weights", "coef", "intercept", "classes")
LEGACY = "e26adf97f729a0fb856c797f89c9ca3f38032bb3"
INTERPRETATION = (
    "Fresh-row confirmation with frozen triplet-reference networks and already "
    "exported decoder coefficients. No refitting, selection, training or default "
    "changes. 1000 class-balanced canonical-training rows, excluding the source "
    "lineage's 3800 used/reserved rows and the reconstructed historic 50000-row "
    "seed-42 training split. Exact pixel duplicates of exclusions and within the "
    "new cohort are removed. This does not certify exclusion of undocumented "
    "historical experiments. The canonical test split is not evaluated. All seeds "
    "share the same new images and previous per-row Poisson seed rule. Primary: "
    "STDP joint-CV accuracy and its paired gain over the fixed decoder. Planned "
    "secondary contrasts compare STDP to no-STDP and initialization, and joint "
    "to voltage-only. Stratified image bootstrap keeps seeds paired; intervals "
    "are conditional on these fixed networks, decoders and 100 readout labels, "
    "not uncertainty across new training runs. Secondary intervals are unadjusted. "
    "The reference's existing numerical/grid limitations remain unchanged."
)


def legacy_training_ids(labels, *, seed=42, per_class=5000, boundary=60000):
    """Replay both train and test shuffles: test draws affect later classes."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    result = []
    for digit in range(10):
        train = np.flatnonzero(labels[:boundary] == digit)
        test = np.flatnonzero(labels[boundary:] == digit) + boundary
        rng.shuffle(train)
        rng.shuffle(test)
        if len(train) < per_class:
            raise ValueError("Not enough rows for the historical split")
        result.extend(train[:per_class].tolist())
    return sorted(result)


def pixel_hashes(images):
    images = np.asarray(images)
    if (images.ndim != 2 or not np.isfinite(images).all() or np.any(images < 0)
            or np.any(images > 255) or np.any(images != np.floor(images))):
        raise ValueError("Expected finite integer-valued MNIST pixels")
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in images.astype(np.uint8)]


def fresh_ids(images, labels, excluded, *, per_class=100, seed=20260913, boundary=60000):
    labels = np.asarray(labels)
    if (type(per_class) is not int or per_class < 1 or len(images) != len(labels)
            or len(labels) < boundary or labels.dtype.kind not in "iu"
            or any(type(i) is not int or not 0 <= i < boundary for i in excluded)):
        raise ValueError("Expected canonical row IDs and a positive class budget")
    hashes = pixel_hashes(np.asarray(images)[:boundary])
    blocked = set(excluded)
    seen = {hashes[i] for i in blocked}
    rng = np.random.default_rng(seed)
    selected, eligible_counts = [], []
    for digit in range(10):
        candidates = [int(i) for i in np.flatnonzero(labels[:boundary] == digit)
                      if int(i) not in blocked and hashes[i] not in seen]
        eligible_counts.append(len(candidates))
        chosen = []
        for i in rng.permutation(candidates):
            i = int(i)
            if hashes[i] not in seen:
                chosen.append(i)
                seen.add(hashes[i])
            if len(chosen) == per_class:
                break
        if len(chosen) != per_class:
            raise ValueError(f"Not enough fresh unique images for digit {digit}")
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected, eligible_counts


def class_predictions(spikes, assignments):
    assignments = np.asarray(assignments)
    if assignments.shape != (spikes.shape[1],):
        raise ValueError("Neuron assignments do not match the frozen network")
    scores = np.zeros((len(spikes), 10))
    for digit in range(10):
        mask = assignments == digit
        if mask.any():
            scores[:, digit] = spikes[:, mask].mean(axis=1)
    pred = scores.argmax(axis=1)
    pred[scores.max(axis=1) == 0] = -1
    return pred


def load_decoders(row, plan):
    if sha256(row["decoder_path"]) != row["decoder_sha256"]:
        raise ValueError("Decoder artifact changed")
    with np.load(row["decoder_path"], allow_pickle=False) as data:
        expected = {"identity"} | {f"{m}__{k}" for m in DECODERS[:-1] for k in STATE_KEYS}
        if set(data.files) != expected or json.loads(str(data["identity"])) != {
                "plan_sha256": plan["tuning_plan_sha256"], "selection_sha256": plan["selection_sha256"],
                "network_state_sha256": row["state_sha256"]}:
            raise ValueError("Decoder identity/state keys differ")
        states = {m: {k: data[f"{m}__{k}"].copy() for k in STATE_KEYS} for m in DECODERS[:-1]}
    for state in states.values():
        for value in state.values():
            value.setflags(write=False)
    return states


def predict_all(states, assignments, spikes, volts):
    features = feature_matrix(spikes, volts)
    result = {m: predict_state(s, features) for m, s in states.items()}
    result["class_average"] = class_predictions(spikes, assignments)
    return result


def verify_legacy():
    path = REPO / "results/20260911-190006"
    status = json.loads((path / "status.json").read_text())
    assert status["git_head"] == LEGACY and status["status"] == "completed"
    assert (path / "working_tree.patch").read_bytes() == b""
    source = subprocess.check_output(["git", "show", f"{LEGACY}:examples/mnist_benchmark.py"], cwd=REPO, text=True)
    def loader(text):
        return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "load_reduced_mnist")
    assert ast.dump(loader(source)) == ast.dump(loader((REPO / "examples/mnist_benchmark.py").read_text()))
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in {"SEED", "TRAIN_PER_CLASS", "TEST_PER_CLASS"}}
    assert constants == {"SEED": 42, "TRAIN_PER_CLASS": 5000, "TEST_PER_CLASS": 50}
    assert "Train: 50000 samples | Test: 500 samples" in (path / "09_mnist_full.log").read_text()
    return {str(path / name): sha256(path / name) for name in ("status.json", "working_tree.patch", "09_mnist_full.log")}, source


def make_plan(tuning_root, output):
    if output.exists():
        raise FileExistsError(output)
    tuning_root = tuning_root.resolve()
    tuning = check_tuning(tuning_root)
    result = json.loads((tuning_root / "summary.json").read_text())
    assert result["complete"] and result["source_unchanged"]
    assert result["plan_sha256"] == sha256(tuning_root / "plan.json")
    assert result["selection_sha256"] == sha256(tuning_root / "selection.json")
    pilot = Path(tuning["source_root"])
    pilot_plan = json.loads((pilot / "plan.json").read_text())
    lineage = sorted(set().union(*(set(pilot_plan["dataset"][k]) for k in
                                  ("train_ids", "validation_ids", "readout_ids", "excluded_ids"))))
    assert len(lineage) == 3800
    legacy_files, legacy_source = verify_legacy()
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=str(REPO / ".sklearn_data"))
    labels = np.asarray(raw.target, dtype=np.int64)
    historical = legacy_training_ids(labels)
    excluded = sorted(set(lineage) | set(historical))
    ids, eligible = fresh_ids(raw.data, labels, excluded)
    images, y = np.asarray(raw.data[ids], dtype=np.float64), labels[ids]
    replay_ids = tuning["readout_ids"][:5]
    replay_images = np.asarray(raw.data[replay_ids], dtype=np.float64)
    rows, files = [], dict(tuning["input_files"])
    files.update(legacy_files)
    for name in ("plan.json", "selection.json", "summary.json"):
        files[str(tuning_root / name)] = sha256(tuning_root / name)
    for item in result["results"]:
        seed, condition = item["seed"], item["condition"]
        study = json.loads((pilot / f"seed_{seed}" / "summary.json").read_text())
        source = study["reference"][condition]
        assert sha256(item["decoder_path"]) == item["decoder_sha256"]
        files[str(Path(item["decoder_path"]).resolve())] = item["decoder_sha256"]
        rows.append({"seed": seed, "condition": condition, "checkpoint": str(Path(source["checkpoint_path"]).resolve()),
            "checkpoint_sha256": source["checkpoint_sha256"], "state_sha256": source["state_sha256"],
            "checkpoint_metadata": {"plan_sha256": sha256(pilot / "plan.json"), "seed": seed,
                "condition": condition, "completed_images": 0 if condition == "initial" else study["training"]["images"]},
            "source_cache": str(pilot / f"seed_{seed}" / f"{condition}-responses.npz"),
            "decoder_path": str(Path(item["decoder_path"]).resolve()), "decoder_sha256": item["decoder_sha256"],
            "assignments": source["decoders"]["class_average"]["assignments"],
            "old_predictions": {m: item["methods"][m]["predictions"] for m in DECODERS[:-1]} |
                               {"class_average": source["decoders"]["class_average"]["predictions"]}})
    output.mkdir(parents=True)
    snapshot = output / "images.npz"
    np.savez_compressed(snapshot, images=images, labels=y, row_ids=ids,
                        replay_images=replay_images, replay_ids=replay_ids)
    sources = dict(tuning["source_sha256"])
    sources["examples/mnist_paper_fresh_evaluation.py"] = sha256(__file__)
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    (output / "legacy_mnist_source.py").write_text(legacy_source)
    plan = {"created_at": time.time(), "seeds": tuning["seeds"], "rows": rows, "methods": list(DECODERS),
        "input_files": files, "source_sha256": sources, "tuning_root": str(tuning_root),
        "tuning_plan_sha256": result["plan_sha256"], "selection_sha256": result["selection_sha256"],
        "legacy_source_sha256": sha256(output / "legacy_mnist_source.py"),
        "images_sha256": sha256(snapshot), "raw_data_sha256": array_digest(images, y),
        "row_ids": ids, "labels": y.tolist(), "replay_ids": replay_ids, "selection_seed": 20260913,
        "lineage_excluded_ids": lineage, "historic_training_ids": historical, "excluded_ids": excluded,
        "eligible_rows_per_class_before_within_cohort_dedup": eligible,
        "per_class": 100, "chunk_size": 100, "poisson_seed_offset": 100000,
        "bootstrap_seed": 20260913, "bootstrap_repeats": 5000, "interpretation": INTERPRETATION}
    atomic_json(output / "plan.json", plan)
    print(f"PLANNED: 1000 new rows, {len(excluded)} excluded IDs; eligible by digit={eligible}", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    check_tuning(Path(plan["tuning_root"]))
    for name, digest in plan["source_sha256"].items():
        assert sha256(REPO / name) == digest == sha256(output / "source_snapshot" / name)
    for path, digest in plan["input_files"].items():
        assert sha256(path) == digest
    for row in plan["rows"]:
        assert directory_digest(row["checkpoint"]) == row["checkpoint_sha256"]
    assert sha256(output / "legacy_mnist_source.py") == plan["legacy_source_sha256"]
    assert sha256(output / "images.npz") == plan["images_sha256"]
    assert len(plan["row_ids"]) == len(set(plan["row_ids"])) == 1000
    assert not set(plan["row_ids"]) & set(plan["excluded_ids"])
    with np.load(output / "images.npz", allow_pickle=False) as data:
        assert array_digest(data["images"], data["labels"]) == plan["raw_data_sha256"]
        np.testing.assert_array_equal(data["row_ids"], plan["row_ids"])
        np.testing.assert_array_equal(data["labels"], plan["labels"])
        np.testing.assert_array_equal(data["replay_ids"], plan["replay_ids"])
        np.testing.assert_array_equal(np.bincount(data["labels"]), np.full(10, plan["per_class"]))
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    if seed not in plan["seeds"]:
        raise ValueError("Seed not in plan")
    folder = output / f"seed_{seed}"
    folder.mkdir()
    plan_hash = sha256(output / "plan.json")
    started = time.monotonic()
    result = {"complete": False, "seed": seed, "plan_sha256": plan_hash, "results": {}}
    atomic_json(folder / "summary.json", result)
    with np.load(output / "images.npz", allow_pickle=False) as data:
        images, labels, ids = data["images"], data["labels"], data["row_ids"]
        replay_images, replay_ids = data["replay_images"], data["replay_ids"]
    for row in [r for r in plan["rows"] if r["seed"] == seed]:
        condition = row["condition"]
        network = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
        assert network.state_digest() == row["state_sha256"]
        states = load_decoders(row, plan)
        with np.load(row["source_cache"], allow_pickle=False) as old:
            previous = predict_all(states, row["assignments"], old["validation_spikes"], old["validation_voltages"])
            for method, pred in previous.items():
                np.testing.assert_array_equal(pred, row["old_predictions"][method])
            replay = frozen_responses(network, replay_images, replay_ids, seed=seed + plan["poisson_seed_offset"])
            for got, name in zip(replay, ("readout_spikes", "readout_voltages", "readout_attempts")):
                np.testing.assert_array_equal(got, old[name][:len(replay_ids)])
        chunks = []
        timer = time.monotonic()
        for start in range(0, len(ids), plan["chunk_size"]):
            stop = min(start + plan["chunk_size"], len(ids))
            chunks.append(frozen_responses(network, images[start:stop], ids[start:stop],
                          seed=seed + plan["poisson_seed_offset"]))
            print(f"EVAL seed={seed} {condition}: {stop}/{len(ids)} new images", flush=True)
        spikes, volts, attempts = (np.concatenate([chunk[j] for chunk in chunks]) for j in range(3))
        predictions = predict_all(states, row["assignments"], spikes, volts)
        artifact = folder / f"{condition}-responses.npz"
        np.savez_compressed(artifact, spikes=spikes, voltages=volts, attempts=attempts, labels=labels,
                            row_ids=ids, identity=json.dumps({"plan_sha256": plan_hash, "state_sha256": row["state_sha256"]}))
        result["results"][condition] = {"cache_sha256": sha256(artifact),
            "methods": {m: {"predictions": pred.tolist(), "correct": int((pred == labels).sum()),
                            "accuracy": float((pred == labels).mean())} for m, pred in predictions.items()},
            "inference_seconds": time.monotonic() - timer, "extra_attempts": int((attempts - 1).sum()),
            "old_predictions_reproduced": True, "old_response_replay_rows": len(replay_ids)}
        assert network.state_digest() == row["state_sha256"]
        atomic_json(folder / "summary.json", result)
        print(f"SCORES seed={seed} {condition}: " + json.dumps(
            {m: v["accuracy"] for m, v in result["results"][condition]["methods"].items()}), flush=True)
    checked_plan(output)
    assert sha256(output / "plan.json") == plan_hash
    result.update(complete=True, source_unchanged=True, wall_seconds=time.monotonic() - started)
    atomic_json(folder / "summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def paired_interval(values, labels, *, repeats=5000, seed=20260913):
    """Bootstrap image IDs within digit, retaining paired predictions/seeds."""
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    if (values.ndim != 2 or values.shape[1] != len(labels) or not all(values.shape)
            or not np.isfinite(values).all() or repeats < 1):
        raise ValueError("Expected finite seed-by-image observations")
    means = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repeats)
    for digit in np.unique(labels):
        group = np.flatnonzero(labels == digit)
        sampled = rng.choice(group, size=(repeats, len(group)), replace=True)
        estimates += means[sampled].sum(axis=1) / len(labels)
    return {"mean_percent_or_pp": float(100 * means.mean()),
            "per_seed_percent_or_pp": (100 * values.mean(axis=1)).tolist(),
            "conditional_image_bootstrap_95_interval": (100 * np.quantile(estimates, [.025, .975])).tolist()}


def summarize(output):
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    plan = checked_plan(output)
    plan_hash = sha256(output / "plan.json")
    studies = [json.loads((output / f"seed_{s}" / "summary.json").read_text()) for s in plan["seeds"]]
    for seed, study in zip(plan["seeds"], studies):
        assert study["complete"] and study["source_unchanged"] and study["seed"] == seed
        assert study["plan_sha256"] == plan_hash and set(study["results"]) == set(CONDITIONS)
        for row in [r for r in plan["rows"] if r["seed"] == seed]:
            condition = row["condition"]
            data_path = output / f"seed_{seed}" / f"{condition}-responses.npz"
            assert sha256(data_path) == study["results"][condition]["cache_sha256"]
            with np.load(data_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["row_ids"], plan["row_ids"])
                np.testing.assert_array_equal(data["labels"], plan["labels"])
                assert json.loads(str(data["identity"])) == {"plan_sha256": plan_hash, "state_sha256": row["state_sha256"]}
                predictions = predict_all(load_decoders(row, plan), row["assignments"], data["spikes"], data["voltages"])
                for method, pred in predictions.items():
                    saved = study["results"][condition]["methods"][method]
                    np.testing.assert_array_equal(pred, saved["predictions"])
                    assert saved["correct"] == int((pred == data["labels"]).sum())
                    assert saved["accuracy"] == float((pred == data["labels"]).mean())
    labels = np.asarray(plan["labels"])
    correct = {c: {m: np.asarray([s["results"][c]["methods"][m]["predictions"] for s in studies]) == labels
                   for m in DECODERS} for c in CONDITIONS}
    observations = {"stdp_joint_accuracy": correct["stdp_normalized"]["joint_cv"].astype(float)}
    for name, condition, method in (("joint_minus_fixed", "stdp_normalized", "fixed_baseline"),
            ("stdp_minus_no_stdp", "normalization_only", "joint_cv"),
            ("stdp_minus_initial", "initial", "joint_cv"),
            ("joint_minus_voltage", "stdp_normalized", "voltage_only_cv")):
        observations[name] = observations["stdp_joint_accuracy"] - correct[condition][method]
    result = {"complete": True, "source_unchanged": True, "plan_sha256": plan_hash,
        "seeds": plan["seeds"], "images": len(labels), "interpretation": INTERPRETATION,
        "mean_accuracy": {c: {m: float(a.mean()) for m, a in methods.items()} for c, methods in correct.items()},
        "planned_estimates": {k: paired_interval(v, labels, repeats=plan["bootstrap_repeats"],
            seed=plan["bootstrap_seed"]) for k, v in observations.items()}}
    checked_plan(output)
    assert sha256(output / "plan.json") == plan_hash
    atomic_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-root", type=Path, default=Path("results/20260912-paper-readout-tuning"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.tuning_root, args.output)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run needs --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
