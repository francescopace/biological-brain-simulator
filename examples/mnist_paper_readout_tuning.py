"""Train-only decoder selection on frozen conductance-LIF response caches.

Run plan, select, then evaluate. Selection never loads validation features or
labels. The reused validation set is exploratory, not a new held-out test.
No SNN simulation, extra labels, production change or network training occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from examples.training_checkpoint import atomic_json, directory_digest


REPO = Path(__file__).resolve().parent.parent
CONDITIONS = ("initial", "normalization_only", "stdp_normalized")
METHODS = ("joint_cv", "equal_weight_cv", "spikes_only_cv", "voltage_only_cv")
BASELINE = {"spike_weight": 1., "voltage_weight": 1., "alpha": 1.}
INTERPRETATION = (
    "Decoder-only exploratory analysis of saved triplet-reference responses. "
    "Each seed/condition uses the same 100 labelled readout rows: 5-fold stratified "
    "CV repeated three times, with scalers fitted inside each fold. Weights are "
    "applied after standardization. All selections are saved before validation "
    "arrays are loaded. The selected models are refitted on all 100 readout rows. "
    "Primary: jointly selected STDP decoder minus fixed equal-weight alpha=1 "
    "STDP decoder. Controls receive the same selection budget. Reused 800-row "
    "validation is not an independent confirmation or canonical test; CV scores "
    "are selection scores, not unbiased performance estimates. No SNN training, "
    "new responses, extra labelled rows or production default changes."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_grid():
    weights = [(1., r) for r in (1., .5, 2., .25, 4., 0.)] + [(0., 1.)]
    return [{"spike_weight": s, "voltage_weight": v, "alpha": a}
            for s, v in weights for a in (.01, .1, 1., 10., 100., 1000., 10000.)]


def feature_matrix(spikes, voltages):
    spikes, voltages = np.asarray(spikes, dtype=np.float64), np.asarray(voltages, dtype=np.float64)
    if (spikes.ndim != 2 or not all(spikes.shape) or voltages.shape != spikes.shape
            or not np.isfinite(spikes).all() or not np.isfinite(voltages).all()
            or np.any(spikes < 0)):
        raise ValueError("Expected aligned finite spike and voltage matrices")
    return np.concatenate((spikes, voltages - voltages.mean(axis=1, keepdims=True)), axis=1)


def block_weights(n_features, parameter):
    s, v, a = (parameter[k] for k in ("spike_weight", "voltage_weight", "alpha"))
    if (n_features < 2 or n_features % 2 or not all(math.isfinite(x) for x in (s, v, a))
            or min(s, v) < 0 or max(s, v) == 0 or a <= 0):
        raise ValueError("Expected two equal feature blocks and valid weights/alpha")
    return np.repeat([s, v], n_features // 2)


def fit_state(features, labels, parameter):
    scaler = StandardScaler().fit(features)
    weights = block_weights(features.shape[1], parameter)
    model = RidgeClassifier(alpha=parameter["alpha"], solver="cholesky")
    model.fit(scaler.transform(features) * weights, labels)
    return {"mean": scaler.mean_, "scale": scaler.scale_, "weights": weights,
            "coef": np.atleast_2d(model.coef_), "intercept": model.intercept_, "classes": model.classes_}


def predict_state(state, features):
    scores = ((features - state["mean"]) / state["scale"] * state["weights"]) @ state["coef"].T
    scores += state["intercept"]
    index = (scores[:, 0] > 0).astype(int) if len(state["classes"]) == 2 else scores.argmax(axis=1)
    return state["classes"][index]


def fold_plan(labels, *, splits=5, repeats=3, seed=20260912):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or not labels.size
            or splits < 2 or repeats < 1 or len(np.unique(labels)) < 2
            or np.unique(labels, return_counts=True)[1].min() < splits):
        raise ValueError("Expected integer labels with enough rows per class")
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    return [{"train": a.tolist(), "heldout": b.tolist()} for a, b in cv.split(np.zeros(len(labels)), labels)]


def cross_validate(features, labels, folds, candidates):
    """Only readout data enter here; no validation argument or global cache."""
    labels = np.asarray(labels)
    predictions = np.full((len(candidates), len(folds), len(labels)), -1, dtype=np.int64)
    counts = np.zeros((len(candidates), len(folds)), dtype=np.int64)
    for j, fold in enumerate(folds):
        train, heldout = np.asarray(fold["train"]), np.asarray(fold["heldout"])
        if (set(train) | set(heldout) != set(range(len(labels))) or len(set(train) & set(heldout))
                or len(train) + len(heldout) != len(labels)):
            raise ValueError("Folds must be disjoint partitions of the readout rows")
        scaler = StandardScaler().fit(features[train])
        left, right = scaler.transform(features[train]), scaler.transform(features[heldout])
        for i, parameter in enumerate(candidates):
            weights = block_weights(features.shape[1], parameter)
            model = RidgeClassifier(alpha=parameter["alpha"], solver="cholesky")
            model.fit(left * weights, labels[train])
            pred = model.predict(right * weights)
            predictions[i, j, heldout] = pred
            counts[i, j] = np.count_nonzero(pred == labels[heldout])
    return counts, predictions


def select_candidates(counts, candidates):
    """Exact-count ties prefer stronger regularization, then balanced weights."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or len(counts) != len(candidates) or np.any(counts < 0):
        raise ValueError("Expected candidate-by-fold correct counts")
    totals = counts.sum(axis=1)
    def key(i):
        p = candidates[i]
        distance = abs(math.log2(p["voltage_weight"] / p["spike_weight"])) if min(
            p["spike_weight"], p["voltage_weight"]) > 0 else math.inf
        return -int(totals[i]), -p["alpha"], distance, i
    groups = {
        "joint_cv": range(len(candidates)),
        "equal_weight_cv": [i for i, p in enumerate(candidates) if p["spike_weight"] == p["voltage_weight"]],
        "spikes_only_cv": [i for i, p in enumerate(candidates) if p["voltage_weight"] == 0],
        "voltage_only_cv": [i for i, p in enumerate(candidates) if p["spike_weight"] == 0],
    }
    return {name: min(indices, key=key) for name, indices in groups.items()}


def make_plan(source, output):
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve()
    old_plan = json.loads((source / "plan.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    assert summary["complete"] and summary["source_unchanged"]
    assert summary["plan_sha256"] == sha256(source / "plan.json")
    labels = np.asarray(old_plan["train_y"])[old_plan["readout_indices"]]
    assert len(labels) == 100 and np.array_equal(np.bincount(labels), np.full(10, 10))
    readout_ids, validation_ids = old_plan["dataset"]["readout_ids"], old_plan["dataset"]["validation_ids"]
    assert len(set(readout_ids)) == 100 and len(set(validation_ids)) == 800
    assert not set(readout_ids) & set(validation_ids)
    files = {str(source / n): sha256(source / n) for n in ("plan.json", "summary.json")}
    checkpoints, rows = {}, []
    for seed in old_plan["seeds"]:
        path = source / f"seed_{seed}" / "summary.json"
        study = json.loads(path.read_text())
        assert study["complete"] and study["source_unchanged"]
        assert study["plan_sha256"] == summary["plan_sha256"]
        files[str(path)] = sha256(path)
        for condition in CONDITIONS:
            row = study["reference"][condition]
            cache = source / f"seed_{seed}" / f"{condition}-responses.npz"
            assert sha256(cache) == row["cache_sha256"]
            files[str(cache)] = row["cache_sha256"]
            checkpoint = str(Path(row["checkpoint_path"]).resolve())
            assert directory_digest(checkpoint) == row["checkpoint_sha256"]
            checkpoints[checkpoint] = row["checkpoint_sha256"]
            with np.load(cache, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["readout_y"], labels)
                np.testing.assert_array_equal(data["readout_ids"], readout_ids)
                assert json.loads(str(data["identity"])) == {
                    "plan_sha256": summary["plan_sha256"], "state_sha256": row["state_sha256"]}
            rows.append({"seed": seed, "condition": condition, "cache": str(cache),
                         "state_sha256": row["state_sha256"],
                         "baseline_predictions": row["decoders"]["ridge_spikes_centered_voltage"]["predictions"]})
    hashes = dict(old_plan["source_sha256"])
    for name, digest in hashes.items():
        assert sha256(REPO / name) == digest == sha256(source / "source_snapshot" / name)
    hashes["examples/mnist_paper_readout_tuning.py"] = sha256(__file__)
    plan = {"created_at": time.time(), "source_root": str(source), "seeds": old_plan["seeds"],
            "rows": rows, "input_files": files, "checkpoints": checkpoints, "source_sha256": hashes,
            "readout_ids": readout_ids, "validation_ids": validation_ids, "readout_y": labels.tolist(),
            "folds": fold_plan(labels), "cv_splits": 5, "cv_repeats": 3, "cv_seed": 20260912,
            "candidates": candidate_grid(), "baseline": BASELINE, "methods": list(METHODS),
            "tie_break": "correct count descending; alpha descending; abs(log2(voltage/spike)) ascending, endpoints last; candidate index ascending",
            "numpy": np.__version__, "sklearn": sklearn.__version__, "interpretation": INTERPRETATION}
    output.mkdir(parents=True)
    for name in hashes:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / name).read_bytes())
    atomic_json(output / "plan.json", plan)
    print(f"PLANNED: {len(rows)} frozen networks, 49 candidates, 15 training-only folds", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    for name, digest in plan["source_sha256"].items():
        assert sha256(REPO / name) == digest == sha256(output / "source_snapshot" / name)
    for path, digest in plan["input_files"].items():
        assert sha256(path) == digest
    for path, digest in plan["checkpoints"].items():
        assert directory_digest(path) == digest
    assert plan["numpy"] == np.__version__ and plan["sklearn"] == sklearn.__version__
    assert plan["folds"] == fold_plan(plan["readout_y"], splits=plan["cv_splits"],
        repeats=plan["cv_repeats"], seed=plan["cv_seed"])
    return plan


def select(output):
    plan = checked_plan(output)
    destination = output / "selection"
    destination.mkdir()
    before = sha256(output / "plan.json")
    started = time.monotonic()
    labels = np.asarray(plan["readout_y"])
    result = {"complete": False, "plan_sha256": before, "results": []}
    for row in plan["rows"]:
        with np.load(row["cache"], allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["readout_y"], labels)
            features = feature_matrix(cache["readout_spikes"], cache["readout_voltages"])
        counts, predictions = cross_validate(features, labels, plan["folds"], plan["candidates"])
        choices = select_candidates(counts, plan["candidates"])
        artifact = destination / f"seed_{row['seed']}-{row['condition']}-cv.npz"
        np.savez_compressed(artifact, predictions=predictions, correct_counts=counts)
        entry = {"seed": row["seed"], "condition": row["condition"], "cv_path": str(artifact),
                 "cv_sha256": sha256(artifact), "choices": choices,
                 "candidate_correct_counts": counts.tolist(), "heldout_predictions_per_candidate": 300}
        result["results"].append(entry)
        p = plan["candidates"][choices["joint_cv"]]
        print(f"SELECT seed={row['seed']} {row['condition']}: {p}, CV={counts[choices['joint_cv']].sum()}/300", flush=True)
    checked_plan(output)
    assert sha256(output / "plan.json") == before
    result.update(complete=True, wall_seconds=time.monotonic() - started)
    atomic_json(output / "selection.json", result)


def evaluate(output):
    plan = checked_plan(output)
    selection = json.loads((output / "selection.json").read_text())
    assert selection["complete"] and selection["plan_sha256"] == sha256(output / "plan.json")
    selection_hash = sha256(output / "selection.json")
    destination = output / "evaluation"
    destination.mkdir()
    labels = np.asarray(plan["readout_y"])
    old_plan = json.loads((Path(plan["source_root"]) / "plan.json").read_text())
    result = {"complete": False, "plan_sha256": selection["plan_sha256"],
              "selection_sha256": selection_hash, "interpretation": INTERPRETATION, "results": []}
    started = time.monotonic()
    assert len(selection["results"]) == len(plan["rows"])
    for row, selected in zip(plan["rows"], selection["results"]):
        assert (row["seed"], row["condition"]) == (selected["seed"], selected["condition"])
        assert sha256(selected["cv_path"]) == selected["cv_sha256"]
        counts = np.asarray(selected["candidate_correct_counts"])
        with np.load(selected["cv_path"], allow_pickle=False) as cv:
            np.testing.assert_array_equal(cv["correct_counts"], counts)
            for j, fold in enumerate(plan["folds"]):
                ids = fold["heldout"]
                np.testing.assert_array_equal((cv["predictions"][:, j, ids] == labels[ids]).sum(axis=1), counts[:, j])
        assert selected["choices"] == select_candidates(counts, plan["candidates"])
        with np.load(row["cache"], allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["readout_y"], labels)
            np.testing.assert_array_equal(cache["readout_ids"], plan["readout_ids"])
            np.testing.assert_array_equal(cache["validation_ids"], plan["validation_ids"])
            np.testing.assert_array_equal(cache["validation_y"], old_plan["validation_y"])
            train = feature_matrix(cache["readout_spikes"], cache["readout_voltages"])
            valid = feature_matrix(cache["validation_spikes"], cache["validation_voltages"])
            valid_y = cache["validation_y"]
        methods, states = {}, {}
        for name in ("fixed_baseline", *METHODS):
            p = plan["baseline"] if name == "fixed_baseline" else plan["candidates"][selected["choices"][name]]
            state = fit_state(train, labels, p)
            predictions = predict_state(state, valid)
            if name == "fixed_baseline":
                np.testing.assert_array_equal(predictions, row["baseline_predictions"])
            methods[name] = {"parameters": p, "predictions": predictions.tolist(),
                             "correct": int((predictions == valid_y).sum()),
                             "accuracy": float((predictions == valid_y).mean())}
            states.update({f"{name}__{k}": v for k, v in state.items()})
        artifact = destination / f"seed_{row['seed']}-{row['condition']}-decoders.npz"
        np.savez_compressed(artifact, **states,
            identity=json.dumps({"plan_sha256": result["plan_sha256"], "selection_sha256": selection_hash,
                                 "network_state_sha256": row["state_sha256"]}))
        with np.load(artifact, allow_pickle=False) as saved:
            for name in methods:
                state = {k: saved[f"{name}__{k}"] for k in ("mean", "scale", "weights", "coef", "intercept", "classes")}
                np.testing.assert_array_equal(predict_state(state, valid), methods[name]["predictions"])
        result["results"].append({"seed": row["seed"], "condition": row["condition"], "methods": methods,
                                  "decoder_path": str(artifact), "decoder_sha256": sha256(artifact)})
        print(f"SCORED seed={row['seed']} {row['condition']}: " + json.dumps(
            {k: v["accuracy"] for k, v in methods.items()}), flush=True)
    result["mean_accuracy"] = {condition: {name: float(np.mean([r["methods"][name]["accuracy"]
        for r in result["results"] if r["condition"] == condition])) for name in ("fixed_baseline", *METHODS)}
        for condition in CONDITIONS}
    result["primary_per_seed_gain_pp"] = [100 * (r["methods"]["joint_cv"]["accuracy"] -
        r["methods"]["fixed_baseline"]["accuracy"]) for r in result["results"] if r["condition"] == "stdp_normalized"]
    checked_plan(output)
    assert sha256(output / "plan.json") == result["plan_sha256"]
    assert sha256(output / "selection.json") == selection_hash
    result.update(complete=True, source_unchanged=True, wall_seconds=time.monotonic() - started)
    atomic_json(output / "summary.json", result)
    print("COMPLETE " + json.dumps({"means": result["mean_accuracy"], "primary_gains_pp": result["primary_per_seed_gain_pp"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "select", "evaluate"))
    parser.add_argument("--source", type=Path, default=Path("results/20260912-paper-triplet-pilot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.source, args.output)
    elif args.mode == "select":
        select(args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
