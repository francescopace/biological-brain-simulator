"""Bounded, paired CPU pilot of the isolated Diehl--Cook triplet reference.

Use plan, then run each seed, then summarize. Existing Izhikevich caches are
scored with the same classifiers and image IDs; their dynamics, preprocessing
and presentation durations still differ. This is not an 87% paper replication.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.datasets import fetch_openml

from examples.mnist_learning_check import LearningConfig, prepare_dataset, ridge_predictions
from examples.mnist_paper_reference import (
    ReferenceConfig, ReferenceNetwork, class_average_predictions, frozen_responses, poisson_tape,
)
from examples.training_checkpoint import array_digest, atomic_json, directory_digest


REPO = Path(__file__).resolve().parent.parent
CONDITIONS = ("initial", "normalization_only", "stdp_normalized")
DECODERS = ("class_average", "ridge_spikes", "ridge_spikes_centered_voltage")
INTERPRETATION = (
    "Exploratory triplet-reference pilot, not the paper's 87% power-law result. "
    "1000 training rows once, 100 labelled readout rows, 800 reused validation rows, "
    "400 excitatory cells, no canonical test rows or parameter selection. "
    "Class-average spike decoding is primary; fixed-alpha ridge probes are secondary. "
    "Reference initialization includes the divisive normalization required before "
    "the first training image. The no-STDP control replays every training attempt chosen by the STDP arm, "
    "with normalization and theta adaptation active in both. Inference resets state "
    "per image and keys Poisson draws by seed/row/attempt, with weights and theta frozen. "
    "Reference: raw pixel rates, dense conductance LIF, 350ms presentation/150ms rest. "
    "Cached Izhikevich comparator: training-fitted L1 preprocessing, sparse current "
    "synapses, 100ms training/25ms rest/50ms independent inference. Same data/decoders "
    "do not isolate one mechanism or equalize simulated time. A new NumPy integrator "
    "and scheduler are not asserted to match Brian-1 trajectories. The grid uses "
    "Bernoulli Poisson draws, floored input delays, clamped refractory voltages, "
    "bounded divisive normalization and a fail-fast retry cap of ten. Silent readout "
    "neurons are unassigned instead of assigned to digit zero. New plans pin "
    "reference dynamics v2: eight complete internal steps per .5ms input interval "
    "and integer refractory release ticks. Production Brain defaults are unchanged."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(study):
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                       data_home=str(REPO / ".sklearn_data"))
    labels = np.asarray(raw.target, dtype=np.int64)
    original = prepare_dataset(raw.data, labels, LearningConfig(**study["signature"]["config"]),
        excluded_ids=study["signature"].get("excluded_ids", []))
    assert original.manifest == study["dataset"]
    train = np.asarray(raw.data[original.manifest["train_ids"]], dtype=np.float64)
    valid = np.asarray(raw.data[original.manifest["validation_ids"]], dtype=np.float64)
    return train, valid, original


def decoders(train_spikes, train_volts, labels, valid_spikes, valid_volts, valid_labels):
    predictions, assignments, _ = class_average_predictions(train_spikes, labels, valid_spikes)
    scores = {"class_average": {"predictions": predictions.tolist(),
        "accuracy": float((predictions == valid_labels).mean()), "assignments": assignments.tolist()}}
    train_volts, valid_volts = train_volts.astype(np.float64), valid_volts.astype(np.float64)
    for name, left, right in (
        ("ridge_spikes", train_spikes, valid_spikes),
        ("ridge_spikes_centered_voltage",
         np.concatenate((train_spikes.astype(np.float64), train_volts - train_volts.mean(axis=1, keepdims=True)), axis=1),
         np.concatenate((valid_spikes.astype(np.float64), valid_volts - valid_volts.mean(axis=1, keepdims=True)), axis=1)),
    ):
        prediction, _ = ridge_predictions(left, labels, right, 1.)
        scores[name] = {"predictions": prediction.tolist(), "accuracy": float((prediction == valid_labels).mean())}
    return scores


def train_pair(trained, control, images, *, seed, callback=None):
    """Same order, input events and retry multiplicity in both learning arms."""
    if trained.state_digest() != control.state_digest():
        raise ValueError("Paired training requires identical initial states")
    c = trained.config
    order = np.random.default_rng(seed).permutation(len(images))
    records = []
    for position, row in enumerate(order):
        for attempt in range(c.max_attempts):
            tape = poisson_tape(images[row], c,
                seed=np.random.SeedSequence([seed, 0, position, attempt]), attempt=attempt)
            trained.normalize()
            control.normalize()
            spikes, inh, _ = trained.advance(tape, learn=True, adapt=True)
            off, _, _ = control.advance(tape, learn=False, adapt=True)
            # Trace, voltage, threshold and learning state persist during rest.
            trained.rest(learn=True, adapt=True)
            control.rest(learn=False, adapt=True)
            if spikes.sum() >= c.min_spikes:
                records.append({"position": position, "training_index": int(row), "attempts": attempt + 1,
                    "accepted_exc_spikes": int(spikes.sum()), "accepted_inh_spikes": int(inh.sum()),
                    "control_accepted_exc_spikes": int(off.sum())})
                break
        else:
            raise RuntimeError(f"Training image {int(row)} exhausted the retry budget")
        assert trained.step_count == control.step_count
        if callback is not None:
            callback(position + 1, records)
    return records


def weight_diagnostics(network, initial):
    w = network.weights
    unit = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-15)
    cosine = unit.T @ unit
    probabilities = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-15)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-15))).sum(axis=0) / np.log(w.shape[0])
    return {"relative_l1_from_initial": float(np.abs(w - initial.weights).sum() / np.abs(initial.weights).sum()),
        "mean_off_diagonal_weight_cosine": float((cosine.sum() - np.trace(cosine)) / (w.shape[1]*(w.shape[1]-1))),
        "mean_normalized_weight_entropy": float(entropy.mean()),
        "weights_at_zero": int((w == 0).sum()), "weights_at_max": int((w == network.config.weight_max).sum()),
        "mean_theta_mv": float(network.theta.mean())}


def make_plan(source_root, output, seeds):
    if output.exists():
        raise ValueError("Use a new output directory")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected distinct seeds")
    paths = [source_root / f"seed_{s}" / "summary.json" for s in seeds]
    studies = [json.loads(p.read_text()) for p in paths]
    for seed, study in zip(seeds, studies):
        assert study["complete"] and study["signature"]["seeds"] == [seed]
        assert study["dataset"] == studies[0]["dataset"]
        assert study["signature"]["config"] == studies[0]["signature"]["config"]
    train, valid, data = dataset(studies[0])
    assert len(train) == 1000 and len(valid) == 800 and len(data.readout_indices) == 100
    assert not set(data.manifest["train_ids"]) & set(data.manifest["validation_ids"])
    names = set(studies[0]["signature"]["source_sha256"]) | {
        "examples/mnist_paper_reference.py", "examples/mnist_paper_check.py", "examples/iris_benchmark.py"}
    hashes = {n: sha256(REPO / n) for n in sorted(names)}
    for path, study in zip(paths, studies):
        for name, digest in study["signature"]["source_sha256"].items():
            assert hashes[name] == digest == sha256(path.parent / "source_snapshot" / name)
    plan = {"created_at": time.time(), "seeds": seeds, "conditions": list(CONDITIONS),
        "decoders": list(DECODERS), "reference_config": asdict(ReferenceConfig()),
        "dataset": data.manifest, "readout_indices": data.readout_indices.tolist(),
        "train_y": data.train_y.tolist(), "validation_y": data.validation_y.tolist(),
        "raw_train_sha256": array_digest(train, data.train_y),
        "raw_validation_sha256": array_digest(valid, data.validation_y),
        "studies": {str(s): {"path": str(p.resolve()), "sha256": sha256(p)} for s, p in zip(seeds, paths)},
        "source_sha256": hashes, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "primary": "class_average: reference STDP minus reference matched no-STDP control",
        "interpretation": INTERPRETATION,
        "preflight": {"path": "results/20260912-paper-triplet-normalized-preflight.json",
            "sha256": sha256(REPO / "results/20260912-paper-triplet-normalized-preflight.json"),
            "finding": "Historical v1 preflight: eight normalized initial-network training images gave 117 spikes at dt=.5ms versus 109 at .25ms with matched input events. This does not validate v2. New v2 plans retain the .5ms input grid but integrate the full event and learning dynamics at .0625ms; MNIST convergence is still unmeasured."},
        "references": ["https://doi.org/10.3389/fncom.2015.00099",
            "https://github.com/peter-u-diehl/stdp-mnist/blob/master/Diehl%26Cook_spiking_MNIST.py",
            "https://github.com/peter-u-diehl/stdp-mnist/blob/master/Diehl%26Cook_MNIST_random_conn_generator.py"]}
    output.mkdir(parents=True)
    for name in names:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / name).read_bytes())
    atomic_json(output / "plan.json", plan)
    print("PLANNED " + json.dumps({"seeds": seeds, "train": len(train), "readout": 100, "validation": len(valid)}), flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    for name, digest in plan["source_sha256"].items():
        assert sha256(REPO / name) == digest == sha256(output / "source_snapshot" / name)
    for source in plan["studies"].values():
        assert sha256(source["path"]) == source["sha256"]
    assert sha256(REPO / plan["preflight"]["path"]) == plan["preflight"]["sha256"]
    assert plan["numpy"] == np.__version__ and plan["sklearn"] == sklearn.__version__
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    if seed not in plan["seeds"]:
        raise ValueError("Seed not in plan")
    folder = output / f"seed_{seed}"
    folder.mkdir()
    plan_hash = sha256(output / "plan.json")
    study_path = Path(plan["studies"][str(seed)]["path"])
    study = json.loads(study_path.read_text())
    train, valid, data = dataset(study)
    assert array_digest(train, data.train_y) == plan["raw_train_sha256"]
    assert array_digest(valid, data.validation_y) == plan["raw_validation_sha256"]
    models = {"initial": ReferenceNetwork(ReferenceConfig.from_dict(plan["reference_config"]), seed=seed)}
    models["initial"].normalize()
    models["normalization_only"] = copy.deepcopy(models["initial"])
    models["stdp_normalized"] = copy.deepcopy(models["initial"])
    result = {"complete": False, "started_at": time.time(), "seed": seed, "plan_sha256": plan_hash,
              "interpretation": INTERPRETATION, "reference": {}, "cached_izhikevich": {}}
    atomic_json(folder / "summary.json", result)
    def checkpoint(completed, records):
        if completed % 100 == 0:
            atomic_json(folder / "training-progress.json", {"completed_images": completed,
                "attempts": sum(r["attempts"] for r in records), "records": records})
            print(f"TRAIN seed={seed}: {completed}/{len(train)} images, {sum(r['attempts'] for r in records)} attempts", flush=True)
        if completed % 250 == 0:
            for condition in CONDITIONS[1:]:
                models[condition].save(folder / "checkpoints" / f"{condition}-{completed:04d}",
                    {"plan_sha256": plan_hash, "seed": seed, "condition": condition, "completed_images": completed})
    print(f"TRAIN seed={seed}: paired triplet/no-STDP reference", flush=True)
    timer = time.monotonic()
    records = train_pair(models["stdp_normalized"], models["normalization_only"], train, seed=seed, callback=checkpoint)
    result["training"] = {"wall_seconds": time.monotonic() - timer, "records": records,
        "images": len(train), "attempts": sum(r["attempts"] for r in records),
        "step_count_per_arm": models["stdp_normalized"].step_count,
        "paired_relative_l1_weight_difference": float(np.abs(models["stdp_normalized"].weights -
            models["normalization_only"].weights).sum() / np.abs(models["normalization_only"].weights).sum())}
    for condition in CONDITIONS:
        network = models[condition]
        metadata = {"plan_sha256": plan_hash, "seed": seed, "condition": condition,
                    "completed_images": 0 if condition == "initial" else len(train)}
        path = folder / "checkpoints" / ("initial-0000" if condition == "initial" else f"{condition}-{len(train):04d}")
        if condition == "initial":
            network.save(path, metadata)
        assert ReferenceNetwork.load(path, metadata).state_digest() == network.state_digest()
        before = directory_digest(path)
        print(f"INFERENCE seed={seed} {condition}: 100 readout + 800 validation images", flush=True)
        timer = time.monotonic()
        readout = frozen_responses(network, train[data.readout_indices], data.manifest["readout_ids"], seed=seed + 100000)
        validation = frozen_responses(network, valid, data.manifest["validation_ids"], seed=seed + 100000)
        cache = folder / f"{condition}-responses.npz"
        np.savez_compressed(cache, readout_spikes=readout[0], readout_voltages=readout[1], readout_attempts=readout[2],
            validation_spikes=validation[0], validation_voltages=validation[1], validation_attempts=validation[2],
            readout_y=data.train_y[data.readout_indices], validation_y=data.validation_y,
            readout_ids=data.manifest["readout_ids"], validation_ids=data.manifest["validation_ids"],
            identity=json.dumps({"plan_sha256": plan_hash, "state_sha256": network.state_digest()}))
        scores = decoders(readout[0], readout[1], data.train_y[data.readout_indices],
                           validation[0], validation[1], data.validation_y)
        assignments = np.asarray(scores["class_average"]["assignments"])
        result["reference"][condition] = {"decoders": scores, "state_sha256": network.state_digest(),
            "checkpoint_path": str(path), "checkpoint_sha256": before, "cache_sha256": sha256(cache),
            "weight_diagnostics": weight_diagnostics(network, models["initial"]),
            "mean_validation_spikes": float(validation[0].sum(axis=1).mean()),
            "unassigned_neurons": int((assignments < 0).sum()),
            "neuron_class_counts": [int((assignments == c).sum()) for c in range(10)],
            "readout_extra_attempts": int((readout[2]-1).sum()),
            "validation_extra_attempts": int((validation[2]-1).sum()),
            "inference_wall_seconds": time.monotonic() - timer}
        assert directory_digest(path) == before
        print(f"SCORES seed={seed} {condition}: " + json.dumps({k: v["accuracy"] for k, v in scores.items()}), flush=True)
        atomic_json(folder / "summary.json", result)
    old_rows = {r["condition"]: r for r in study["results"]}
    for condition in CONDITIONS:
        cache_path = study_path.parent / f"seed_{seed}" / condition / "responses.npz"
        before = sha256(cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            assert json.loads(str(cache["identity"])) == {
                "brain_sha256": old_rows[condition]["final_sha256"],
                "inference_steps": study["signature"]["config"]["inference_steps"],
                "readout_sha256": data.manifest["readout_sha256"],
                "validation_sha256": data.manifest["validation_sha256"]}
            np.testing.assert_array_equal(cache["readout_y"], data.train_y[data.readout_indices])
            np.testing.assert_array_equal(cache["validation_y"], data.validation_y)
            scores = decoders(cache["readout_spikes"], cache["readout_voltages"], cache["readout_y"],
                cache["validation_spikes"], cache["validation_voltages"], cache["validation_y"])
            for name in DECODERS[1:]:
                assert scores[name]["predictions"] == old_rows[condition]["decoders"][name]["predictions"]
        assert sha256(cache_path) == before
        result["cached_izhikevich"][condition] = {"decoders": scores, "cache_path": str(cache_path), "cache_sha256": before}
    checked_plan(output)
    assert sha256(output / "plan.json") == plan_hash
    result.update(complete=True, source_unchanged=True, completed_at=time.time())
    atomic_json(folder / "summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output / "summary.json").exists():
        raise ValueError("Summary exists")
    plan = checked_plan(output)
    results = [json.loads((output / f"seed_{s}" / "summary.json").read_text()) for s in plan["seeds"]]
    for seed, result in zip(plan["seeds"], results):
        assert result["seed"] == seed and result["complete"] and result["source_unchanged"]
        assert result["plan_sha256"] == sha256(output / "plan.json")
        for condition in CONDITIONS:
            row = result["reference"][condition]
            assert directory_digest(Path(row["checkpoint_path"])) == row["checkpoint_sha256"]
            cache_path = output / f"seed_{seed}" / f"{condition}-responses.npz"
            assert sha256(cache_path) == row["cache_sha256"]
            with np.load(cache_path, allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["readout_ids"], plan["dataset"]["readout_ids"])
                np.testing.assert_array_equal(cache["validation_ids"], plan["dataset"]["validation_ids"])
                actual = decoders(cache["readout_spikes"], cache["readout_voltages"], cache["readout_y"],
                    cache["validation_spikes"], cache["validation_voltages"], cache["validation_y"])
                assert actual == row["decoders"]
            old = result["cached_izhikevich"][condition]
            assert sha256(old["cache_path"]) == old["cache_sha256"]
    summary = {"complete": True, "source_unchanged": True, "seeds": plan["seeds"],
        "plan_sha256": sha256(output / "plan.json"), "interpretation": INTERPRETATION, "results": {}}
    for architecture in ("reference", "cached_izhikevich"):
        means = {condition: {decoder: float(np.mean([r[architecture][condition]["decoders"][decoder]["accuracy"]
            for r in results])) for decoder in DECODERS} for condition in CONDITIONS}
        effects = {decoder: {control: [100*(r[architecture]["stdp_normalized"]["decoders"][decoder]["accuracy"] -
            r[architecture][control]["decoders"][decoder]["accuracy"]) for r in results]
            for control in ("initial", "normalization_only")} for decoder in DECODERS}
        summary["results"][architecture] = {"mean_accuracy": means, "per_seed_stdp_gains_pp": effects}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def preflight(source_root, output):
    if output.exists():
        raise ValueError("Use a new preflight output file")
    study = json.loads((source_root / "seed_201" / "summary.json").read_text())
    train, _, data = dataset(study)
    initial = ReferenceNetwork()
    initial.normalize()
    trained, control = copy.deepcopy(initial), copy.deepcopy(initial)
    timer = time.monotonic()
    records = train_pair(trained, control, train[:8], seed=201)
    refinement = []
    for row in range(8):
        low = copy.deepcopy(initial)
        low.reset_transients()
        high = ReferenceNetwork(replace(low.config, dt=.25))
        high.weights = low.weights.copy()
        high.delays = low.delays * 2
        high.reset_transients()
        tape = poisson_tape(train[row], low.config, seed=2000 + row)
        finer = np.zeros((len(tape)*2, low.config.n_input), dtype=bool)
        finer[::2] = tape
        left, _, _ = low.advance(tape)
        right, _, _ = high.advance(finer)
        refinement.append({"row_id": data.manifest["train_ids"][row], "spikes_dt05": int(left.sum()),
            "spikes_dt025": int(right.sum()), "different_neuron_counts": int((left != right).sum())})
    payload = {"complete": True, "config": asdict(initial.config), "training_records": records,
        "wall_seconds": time.monotonic() - timer, "refinement_fixed_input_events": refinement,
        "paired_weight_l1": float(np.abs(trained.weights-control.weights).sum()),
        "interpretation": "Training-only smoke/refinement checks, no decoder or accuracy-based parameter selection. Finer grid shares weights and physical input arrival times; recurrent grid latency also changes."}
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "plan", "run", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("results/20260912-wta64-fresh-replication"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[201, 202, 203])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.source_root, args.output, args.seeds)
    elif args.mode == "preflight":
        preflight(args.source_root, args.output)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
