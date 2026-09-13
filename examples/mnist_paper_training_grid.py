"""Bounded, matched-exposure timestep diagnostic during triplet learning.

Replay the first 50 images/attempts of each original seed on three grids, with
matched no-STDP controls. Probe learned states at one common inference grid.
No classifier fitting, accuracy selection or production-default changes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from examples import mnist_paper_check as pilot
from examples import mnist_paper_grid_check as inference
from examples.mnist_paper_reference import ReferenceNetwork, poisson_tape
from examples.training_checkpoint import array_digest, atomic_json, directory_digest


FACTORS = (1, 2, 4)
ARMS = ("normalization_only", "stdp_normalized")
require = inference.require
INTERPRETATION = (
    "Early-training grid sensitivity, not an accuracy benchmark or convergence "
    "certificate. Three original seeds, first 50 training images in each seed's "
    "original permutation, 0.5/0.25/0.125 ms grids. All grids and no-STDP controls "
    "replay the original coarse STDP arm's exact attempts, events and physical "
    "delays. Normalize before each attempt; theta adapts in both arms; STDP is "
    "active in the trained arm during presentation and rest. Electrical and trace "
    "state persist between images, including the original below-rest initial "
    "voltages. Fine grids do not choose their own retry budgets. Weight/theta "
    "diagnostics at 10/25/50 images; 20 reused balanced probe images at a common "
    "0.125 ms inference grid, same coarse input tapes, one presentation per image, "
    "no retries, frozen weights/theta, reset transients. No readout fitting or "
    "accuracy calculation. This isolates sensitivity of early learned state and "
    "its response at a common finite grid, not long-training convergence, "
    "continuous-time Poisson input or Brian/paper equivalence. No tolerance, "
    "hyperparameter selection or default changes. Seeds share probe images and "
    "the source training pool, but have different original training prefixes."
)


def training_copy(initial, factor):
    """Refine an untouched initial checkpoint without resetting its voltages."""
    require(type(factor) is int and factor in FACTORS, "Unsupported grid factor")
    require(initial.step_count == 0 and not initial.pending_times.size and
            not initial.pending_synapses.size and not initial.pending_e.any() and
            not initial.pending_i.any(), "Training refinement requires an untouched initial state")
    result = copy.deepcopy(initial)
    result.config = replace(initial.config, dt=initial.config.dt / factor)
    result.config.validate()
    result.delays *= factor
    np.testing.assert_array_equal(result.delays * result.config.dt, initial.delays * initial.config.dt)
    return result


def state_metrics(initial, models):
    result = {"by_factor": {}, "grid_changes": {}}
    for f in FACTORS:
        control, trained = (models[f][a] for a in ARMS)
        result["by_factor"][str(f)] = {
            "stdp_minus_control_weight_relative_l1": float(np.abs(trained.weights - control.weights).sum() /
                                                          np.abs(control.weights).sum()),
            "arms": {a: {"weight_relative_l1_from_initial": float(np.abs(n.weights - initial.weights).sum() /
                                                                 np.abs(initial.weights).sum()),
                         "mean_theta_mv": float(n.theta.mean()),
                         "theta_rmse_from_initial_mv": float(np.sqrt(np.mean((n.theta - initial.theta) ** 2))),
                         "weights_at_zero": int((n.weights == 0).sum()),
                         "weights_at_max": int((n.weights == n.config.weight_max).sum())}
                     for a, n in models[f].items()}}
    for start, end in ((1, 2), (2, 4), (1, 4)):
        left, right = models[start], models[end]
        # All no-STDP weights must experience the same normalization operations.
        np.testing.assert_array_equal(left[ARMS[0]].weights, right[ARMS[0]].weights)
        changes = {a: {"weight_relative_l1": float(np.abs(right[a].weights - left[a].weights).sum() /
                                                  np.abs(left[a].weights).sum()),
                       "theta_rmse_mv": float(np.sqrt(np.mean((right[a].theta - left[a].theta) ** 2)))}
                   for a in ARMS}
        update_left = left[ARMS[1]].weights - left[ARMS[0]].weights
        update_right = right[ARMS[1]].weights - right[ARMS[0]].weights
        scale = np.abs(update_left).sum()
        changes["learning_component_relative_l1"] = (
            float(np.abs(update_right - update_left).sum() / scale) if scale > 0 else None)
        result["grid_changes"][f"{start}_to_{end}"] = changes
    return result


def train_coupled(initial, images, original_records, *, seed, callback=None):
    require(len(original_records) > 0, "Empty training prefix")
    before = initial.state_digest()
    models = {f: {a: training_copy(initial, f) for a in ARMS} for f in FACTORS}
    c = initial.config
    history = []
    for position, old in enumerate(original_records):
        require(old["position"] == position and type(old["training_index"]) is int
                and 0 <= old["training_index"] < len(images) and type(old["attempts"]) is int
                and 1 <= old["attempts"] <= c.max_attempts, "Invalid original training record")
        counts = {str(f): {a: {"presentation_exc_spikes": 0} for a in ARMS} for f in FACTORS}
        for attempt in range(old["attempts"]):
            tape = poisson_tape(images[old["training_index"]], c,
                seed=np.random.SeedSequence([seed, 0, position, attempt]), attempt=attempt)
            for factor, arms in models.items():
                events = inference.lift_tape(tape, factor)
                for arm, model in arms.items():
                    learn = arm == "stdp_normalized"
                    model.normalize()
                    spikes, inh, _ = model.advance(events, learn=learn, adapt=True)
                    model.rest(learn=learn, adapt=True)
                    record = counts[str(factor)][arm]
                    record["presentation_exc_spikes"] += int(spikes.sum())
                    record["final_exc_spikes"], record["final_inh_spikes"] = int(spikes.sum()), int(inh.sum())
                    record["below_min_on_final_attempt"] = bool(spikes.sum() < c.min_spikes)
                    if factor == 1 and learn:
                        require((spikes.sum() >= c.min_spikes) == (attempt + 1 == old["attempts"]),
                                "Original coarse retry decision did not replay")
        coarse = counts["1"]
        require(coarse[ARMS[1]]["final_exc_spikes"] == old["accepted_exc_spikes"] and
                coarse[ARMS[1]]["final_inh_spikes"] == old["accepted_inh_spikes"] and
                coarse[ARMS[0]]["final_exc_spikes"] == old["control_accepted_exc_spikes"],
                "Original coarse training spikes did not replay")
        elapsed = models[1][ARMS[1]].step_count * c.dt
        for factor, arms in models.items():
            for model in arms.values():
                require(model.step_count * model.config.dt == elapsed, "Training exposure differs across arms")
                np.testing.assert_array_equal(model.delays * model.config.dt, initial.delays * c.dt)
        history.append({"position": position, "training_index": old["training_index"],
                        "attempts": old["attempts"], "simulated_ms": elapsed, "counts": counts})
        if callback is not None:
            callback(position + 1, models, history)
    require(initial.state_digest() == before, "Source initial state changed")
    return models, history


def common_probe(network, images, row_ids, *, seed, base_config, factor=4):
    """One matched presentation on a common grid, including sub-threshold rows."""
    require(type(factor) is int and factor in FACTORS, "Unsupported probe factor")
    require(len(images) == len(row_ids) and len(images) > 0, "Probe image/ID mismatch")
    before = network.state_digest()
    target_dt = base_config.dt / factor
    model = copy.deepcopy(network)
    physical_delays = network.delays * network.config.dt
    delay_steps = physical_delays / target_dt
    require(np.array_equal(delay_steps, np.rint(delay_steps)), "Probe grid cannot represent physical delays")
    model.config = replace(network.config, dt=target_dt)
    model.config.validate()
    model.delays = np.rint(delay_steps).astype(np.int64)
    spikes, volts = [], []
    for image, row in zip(images, row_ids):
        model.reset_transients()
        tape = poisson_tape(image, base_config, seed=np.random.SeedSequence([seed, int(row), 0]))
        response, _, voltage = model.advance(inference.lift_tape(tape, factor), learn=False, adapt=False)
        model.rest(learn=False, adapt=False)
        spikes.append(response)
        volts.append(voltage)
    np.testing.assert_array_equal(model.weights, network.weights)
    np.testing.assert_array_equal(model.theta, network.theta)
    require(network.state_digest() == before, "Probe mutated the learned network")
    return {"spikes": np.asarray(spikes), "voltages": np.asarray(volts),
            "attempts": np.ones(len(images), dtype=np.int64)}


def probe_metrics(probes):
    def diff(left, right):
        return inference.response_difference(probes[left], probes[right])
    return {"mean_total_spikes": {k: float(v["spikes"].sum(axis=1).mean()) for k, v in probes.items()},
            "grid_changes": {f"{a}_to_{b}": {arm: diff(f"{a}__{arm}", f"{b}__{arm}") for arm in ARMS}
                             for a, b in ((1, 2), (2, 4), (1, 4))},
            "stdp_minus_control": {str(f): diff(f"{f}__{ARMS[0]}", f"{f}__{ARMS[1]}") for f in FACTORS}}


def make_plan(parent_root, output):
    if output.exists():
        raise FileExistsError(output)
    parent_root = parent_root.resolve()
    parent = inference.checked_plan(parent_root)
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    require(parent_summary["complete"] and parent_summary["plan_sha256"] == pilot.sha256(parent_root / "plan.json"),
            "Inference diagnostic must be complete")
    initial_rows = [r for r in parent["rows"] if r["condition"] == "initial"]
    pilot_root = Path(initial_rows[0]["checkpoint"]).parents[2]
    original = pilot.checked_plan(pilot_root)
    inference.require_legacy_protocol(pilot.ReferenceConfig.from_dict(original["reference_config"]))
    source_study = json.loads(Path(original["studies"][str(parent["seeds"][0])]["path"]).read_text())
    train, _, data = pilot.dataset(source_study)
    require(array_digest(train, data.train_y) == original["raw_train_sha256"], "Training pixels changed")
    files = {str(p): pilot.sha256(p) for p in (parent_root / "plan.json", parent_root / "summary.json",
                                            pilot_root / "plan.json")}
    records = {}
    for seed in parent["seeds"]:
        path = pilot_root / f"seed_{seed}" / "summary.json"
        study = json.loads(path.read_text())
        require(study["complete"] and study["plan_sha256"] == pilot.sha256(pilot_root / "plan.json"),
                "Incomplete source training run")
        files[str(path)] = pilot.sha256(path)
        prefix = study["training"]["records"][:50]
        order = np.random.default_rng(seed).permutation(len(train))[:50]
        np.testing.assert_array_equal([r["training_index"] for r in prefix], order)
        records[str(seed)] = prefix
    indices = inference.balanced_indices(parent["labels"], per_class=2, seed=20260915)
    fresh_indices = np.asarray(parent["indices"])[indices]
    with np.load(Path(parent["fresh_root"]) / "images.npz", allow_pickle=False) as images:
        probe_images, probe_labels, probe_ids = (images[k][fresh_indices] for k in ("images", "labels", "row_ids"))
    require(not set(probe_ids) & set(data.manifest["train_ids"]), "Training/probe rows overlap")
    sources = dict(parent["source_sha256"])
    sources["examples/mnist_paper_training_grid.py"] = pilot.sha256(__file__)
    output.mkdir(parents=True)
    np.savez_compressed(output / "images.npz", train_images=train, train_labels=data.train_y,
                        train_ids=data.manifest["train_ids"], probe_images=probe_images,
                        probe_labels=probe_labels, probe_ids=probe_ids)
    for name in sources:
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((pilot.REPO / name).read_bytes())
    plan = {"created_at": time.time(), "parent_root": str(parent_root), "seeds": parent["seeds"],
        "source_sha256": sources, "input_files": files, "initial_rows": initial_rows,
        "images_sha256": pilot.sha256(output / "images.npz"), "training_images": 50,
        "train_ids": data.manifest["train_ids"], "probe_ids": probe_ids.tolist(), "probe_labels": probe_labels.tolist(),
        "probe_selection_seed": 20260915, "records": records, "factors": list(FACTORS), "base_dt_ms": .5,
        "milestones": [10, 25, 50], "probe_factor": 4, "poisson_seed_offset": parent["poisson_seed_offset"],
        "interpretation": INTERPRETATION}
    atomic_json(output / "plan.json", plan)
    print("PLANNED: 50 training rows/seed, three grids, paired controls, 20 common-grid probes", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    inference.checked_plan(Path(plan["parent_root"]))
    for name, digest in plan["source_sha256"].items():
        require(pilot.sha256(pilot.REPO / name) == digest == pilot.sha256(output / "source_snapshot" / name),
                f"Study source changed: {name}")
    for path, digest in plan["input_files"].items():
        require(pilot.sha256(path) == digest, f"Study input changed: {path}")
    require(pilot.sha256(output / "images.npz") == plan["images_sha256"], "Image snapshot changed")
    require(plan["factors"] == list(FACTORS) and plan["training_images"] == 50
            and plan["milestones"] == [10, 25, 50] and plan["probe_factor"] == 4, "Protocol differs")
    with np.load(output / "images.npz", allow_pickle=False) as data:
        for key in ("train_ids", "probe_ids", "probe_labels"):
            np.testing.assert_array_equal(data[key], plan[key])
        require(not set(data["train_ids"]) & set(data["probe_ids"]), "Training/probe rows overlap")
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    require(seed in plan["seeds"], "Seed not in plan")
    folder = output / f"seed_{seed}"
    folder.mkdir()
    plan_hash = pilot.sha256(output / "plan.json")
    row = next(r for r in plan["initial_rows"] if r["seed"] == seed)
    initial = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
    inference.require_legacy_protocol(initial.config)
    require(initial.state_digest() == row["state_sha256"] and initial.config.dt == plan["base_dt_ms"],
            "Initial checkpoint differs")
    started = time.monotonic()
    result = {"complete": False, "seed": seed, "plan_sha256": plan_hash, "milestones": {}}
    atomic_json(folder / "summary.json", result)
    with np.load(output / "images.npz", allow_pickle=False) as data:
        train, probe_images = data["train_images"], data["probe_images"]
    def progress(n, models, history):
        if n in plan["milestones"]:
            result["milestones"][str(n)] = state_metrics(initial, models)
            atomic_json(folder / "summary.json", result)
        if n % 5 == 0:
            atomic_json(folder / "progress.json", {"completed_images": n, "history": history})
            print(f"TRAIN-GRID seed={seed}: {n}/{plan['training_images']}", flush=True)
    models, history = train_coupled(initial, train, plan["records"][str(seed)], seed=seed, callback=progress)
    result["training_seconds"] = time.monotonic() - started
    result["history"] = history
    result["checkpoints"] = {}
    probes = {}
    timer = time.monotonic()
    for name, network in [("initial", initial)] + [(f"{f}__{a}", models[f][a]) for f in FACTORS for a in ARMS]:
        if name != "initial":
            path = folder / "checkpoints" / name
            metadata = {"plan_sha256": plan_hash, "seed": seed, "name": name, "completed_images": len(history)}
            network.save(path, metadata)
            result["checkpoints"][name] = {"path": str(path.resolve()), "metadata": metadata,
                "sha256": directory_digest(path), "state_sha256": network.state_digest()}
        probes[name] = common_probe(network, probe_images, plan["probe_ids"], seed=seed + plan["poisson_seed_offset"],
                                    base_config=initial.config, factor=plan["probe_factor"])
        print(f"PROBE seed={seed}: {name}", flush=True)
    result["probe_seconds"] = time.monotonic() - timer
    path = folder / "probes.npz"
    payload = {f"{name}__{field}": values for name, response in probes.items() for field, values in response.items()}
    np.savez_compressed(path, **payload, row_ids=plan["probe_ids"], identity=json.dumps({"plan_sha256": plan_hash, "seed": seed}))
    result.update(probe_sha256=pilot.sha256(path), probe_metrics=probe_metrics(probes))
    checked_plan(output)
    require(pilot.sha256(output / "plan.json") == plan_hash and initial.state_digest() == row["state_sha256"],
            "Plan or initial state changed")
    result.update(complete=True, source_unchanged=True, coarse_training_exact_replay=True,
                  wall_seconds=time.monotonic() - started)
    atomic_json(folder / "summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    plan = checked_plan(output)
    plan_hash = pilot.sha256(output / "plan.json")
    studies = []
    for seed in plan["seeds"]:
        folder = output / f"seed_{seed}"
        result = json.loads((folder / "summary.json").read_text())
        require(result["complete"] and result["source_unchanged"] and result["coarse_training_exact_replay"]
                and result["plan_sha256"] == plan_hash and result["seed"] == seed, "Incomplete seed result")
        row = next(r for r in plan["initial_rows"] if r["seed"] == seed)
        initial = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
        models = {f: {} for f in FACTORS}
        for name, checkpoint in result["checkpoints"].items():
            require(directory_digest(checkpoint["path"]) == checkpoint["sha256"], "Learned checkpoint changed")
            network = ReferenceNetwork.load(checkpoint["path"], checkpoint["metadata"])
            require(network.state_digest() == checkpoint["state_sha256"], "Learned state changed")
            factor, arm = name.split("__")
            models[int(factor)][arm] = network
        require(state_metrics(initial, models) == result["milestones"]["50"], "Final state metrics differ")
        require(pilot.sha256(folder / "probes.npz") == result["probe_sha256"], "Probe cache changed")
        with np.load(folder / "probes.npz", allow_pickle=False) as data:
            require(json.loads(str(data["identity"])) == {"plan_sha256": plan_hash, "seed": seed}, "Probe identity differs")
            np.testing.assert_array_equal(data["row_ids"], plan["probe_ids"])
            probes = {name: {k: data[f"{name}__{k}"] for k in ("spikes", "voltages", "attempts")}
                      for name in ["initial"] + [f"{f}__{a}" for f in FACTORS for a in ARMS]}
        require(probe_metrics(probes) == result["probe_metrics"], "Probe metrics differ")
        studies.append(result)
    result = {"complete": True, "source_unchanged": True, "plan_sha256": plan_hash,
              "interpretation": INTERPRETATION, "seeds": plan["seeds"], "results": studies}
    checked_plan(output)
    require(pilot.sha256(output / "plan.json") == plan_hash, "Plan changed during summary")
    atomic_json(output / "summary.json", result)
    print(f"SUMMARIZED: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=Path("results/20260913-paper-grid-check"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.parent_root, args.output)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run needs --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
