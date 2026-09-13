"""Matched 50-image CPU diagnostic of v1, v2 and half-step v2.

Inputs and exposure come from the archived early-training study. Historical
sources are verified against their snapshots, never against current code or
rewritten hashes. No accuracy, classifier fitting, or automatic longer run.
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

from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork, poisson_tape
from examples.training_checkpoint import atomic_json, directory_digest


REPO = Path(__file__).resolve().parent.parent
VARIANTS = {"v1": (1, 1), "v2": (2, 8), "v2_half": (2, 16)}
ARMS = ("normalization_only", "stdp_normalized")
PAIRS = (("v1", "v2"), ("v2", "v2_half"), ("v1", "v2_half"))
INTERPRETATION = (
    "Matched early-training diagnostic: seeds 201/202/203, first 50 images in "
    "each original permutation, v1 at .5ms and v2 at .0625/.03125ms internally. "
    "Identical initial weights/theta, below-rest voltages, .5ms Bernoulli input "
    "events, delays, original v1 STDP retry exposure and normalization operations. "
    "Theta adapts in both arms; STDP runs only in the trained arm, including rest. "
    "State persists between training images. Finer models do not choose retries. "
    "Primary comparison is v2 to v2_half; v1 contrasts are secondary. Metrics at "
    "10/25/50 images describe weights, theta, spikes and voltage, not accuracy. "
    "Six learned states per seed are probed on the same 20 reused images at "
    "common v2 .03125ms resolution, frozen weights/theta, per-image electrical "
    "reset, one presentation, no retries or omitted silent images. No labels "
    "enter learning or scoring. Process CPU and elapsed times include matched "
    "presentation/rest and normalization, exclude tape generation/checkpointing; "
    "variants are interleaved and workers may run concurrently. These are not "
    "isolated throughput benchmarks. No numerical pass threshold, parameter "
    "selection, continuous-input convergence or Brian equivalence is asserted. "
    "The half-step grid is finite, not ground truth. No long run is triggered."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_copy(initial, variant):
    require(variant in VARIANTS, "Unknown dynamics variant")
    require(initial.step_count == 0 and not initial.pending_times.size and
            not initial.pending_synapses.size and not initial.pending_e.any() and
            not initial.pending_i.any(), "Version comparison requires an untouched initial state")
    require(initial.config.dynamics_version == 1, "Expected a historical v1 initial state")
    if variant == "v1":
        return copy.deepcopy(initial)
    version, substeps = VARIANTS[variant]
    result = ReferenceNetwork(replace(initial.config, dynamics_version=version, integration_substeps=substeps))
    for name, value in vars(initial).items():
        if isinstance(value, np.ndarray):
            setattr(result, name, value.copy())
    # Only v2 release arrays are new; every original initial-state array survives.
    return result


def state_metrics(initial, models):
    result = {"by_variant": {}, "changes": {}}
    for variant, arms in models.items():
        control, trained = (arms[a] for a in ARMS)
        result["by_variant"][variant] = {
            "stdp_minus_control_weight_relative_l1": float(np.abs(trained.weights-control.weights).sum()/np.abs(control.weights).sum()),
            "arms": {a: {"weight_relative_l1_from_initial": float(np.abs(n.weights-initial.weights).sum()/np.abs(initial.weights).sum()),
                          "mean_theta_mv": float(n.theta.mean()),
                          "weights_at_zero": int((n.weights == 0).sum()),
                          "weights_at_max": int((n.weights == n.config.weight_max).sum())} for a, n in arms.items()}}
    for a, b in PAIRS:
        left, right = models[a], models[b]
        np.testing.assert_array_equal(left[ARMS[0]].weights, right[ARMS[0]].weights)
        changes = {arm: {"weight_relative_l1": float(np.abs(right[arm].weights-left[arm].weights).sum()/np.abs(left[arm].weights).sum()),
                         "theta_rmse_mv": float(np.sqrt(np.mean((right[arm].theta-left[arm].theta)**2)))} for arm in ARMS}
        dl = left[ARMS[1]].weights-left[ARMS[0]].weights
        dr = right[ARMS[1]].weights-right[ARMS[0]].weights
        changes["learning_component_relative_l1"] = float(np.abs(dr-dl).sum()/np.abs(dl).sum()) if np.any(dl) else None
        result["changes"][f"{a}_to_{b}"] = changes
    return result


def response_difference(left, right):
    a, b = left["spikes"].astype(float), right["spikes"].astype(float)
    va, vb = left["voltages"], right["voltages"]
    delta = (vb-vb.mean(axis=-1, keepdims=True))-(va-va.mean(axis=-1, keepdims=True))
    return {"relative_spike_l1": float(np.abs(b-a).sum()/a.sum()) if a.sum() else None,
            "mean_absolute_spike_difference": float(np.abs(b-a).mean()),
            "same_spike_vector_fraction": float(np.all(a == b, axis=-1).mean()),
            "centered_voltage_rmse_mv": float(np.sqrt(np.mean(delta**2))),
            "mean_total_spike_change": float((b-a).sum(axis=-1).mean())}


def response_metrics(responses):
    return {"by_model": {name: {"mean_total_spikes": float(r["spikes"].sum(axis=1).mean()),
                               "silent_fraction": float(np.mean(r["spikes"].sum(axis=1) == 0))} for name, r in responses.items()},
            "changes": {f"{a}_to_{b}": {arm: response_difference(responses[f"{a}__{arm}"], responses[f"{b}__{arm}"])
                         for arm in ARMS} for a, b in PAIRS},
            "stdp_minus_control": {v: response_difference(responses[f"{v}__{ARMS[0]}"], responses[f"{v}__{ARMS[1]}"])
                                   for v in VARIANTS}}


def train_coupled(initial, images, records, *, seed, callback=None):
    require(len(records) > 0, "Empty training prefix")
    before = initial.state_digest()
    models = {v: {a: variant_copy(initial, v) for a in ARMS} for v in VARIANTS}
    traces = {f"{v}__{a}": {"spikes": [], "voltages": []} for v in VARIANTS for a in ARMS}
    costs = {k: {"cpu_seconds": 0., "wall_seconds": 0.} for k in traces}
    history = []
    for position, old in enumerate(records):
        require(old["position"] == position and type(old["training_index"]) is int and
                0 <= old["training_index"] < len(images) and type(old["attempts"]) is int and
                1 <= old["attempts"] <= initial.config.max_attempts, "Invalid original training record")
        counts = {k: {"all_presentation_exc_spikes": 0} for k in traces}
        for attempt in range(old["attempts"]):
            tape = poisson_tape(images[old["training_index"]], initial.config,
                seed=np.random.SeedSequence([seed, 0, position, attempt]), attempt=attempt)
            for variant, arms in models.items():
                for arm, network in arms.items():
                    key, learn = f"{variant}__{arm}", arm == ARMS[1]
                    wall, cpu = time.perf_counter(), time.process_time()
                    network.normalize()
                    spikes, inh, volts = network.advance(tape, learn=learn, adapt=True)
                    network.rest(learn=learn, adapt=True)
                    costs[key]["cpu_seconds"] += time.process_time()-cpu
                    costs[key]["wall_seconds"] += time.perf_counter()-wall
                    counts[key]["all_presentation_exc_spikes"] += int(spikes.sum())
                    counts[key].update(final_exc_spikes=int(spikes.sum()), final_inh_spikes=int(inh.sum()),
                                       below_min_on_final_attempt=bool(spikes.sum() < initial.config.min_spikes))
                    if variant == "v1" and learn:
                        require((spikes.sum() >= initial.config.min_spikes) == (attempt+1 == old["attempts"]),
                                "Historical retry decision did not replay")
                    if attempt+1 == old["attempts"]:
                        traces[key]["spikes"].append(spikes)
                        traces[key]["voltages"].append(volts)
        require(counts[f"v1__{ARMS[1]}"]["final_exc_spikes"] == old["accepted_exc_spikes"] and
                counts[f"v1__{ARMS[1]}"]["final_inh_spikes"] == old["accepted_inh_spikes"] and
                counts[f"v1__{ARMS[0]}"]["final_exc_spikes"] == old["control_accepted_exc_spikes"],
                "Historical training spikes did not replay")
        elapsed = models["v1"][ARMS[1]].step_count
        for arms in models.values():
            for n in arms.values():
                require(n.step_count == elapsed and n.config.dt == initial.config.dt, "Training exposure differs")
                np.testing.assert_array_equal(n.delays, initial.delays)
        history.append(dict(position=position, training_index=old["training_index"], attempts=old["attempts"], counts=counts))
        if callback:
            callback(position+1, models, history, costs)
    require(initial.state_digest() == before, "Initial state changed")
    return models, history, {k: {f: np.asarray(v) for f, v in r.items()} for k, r in traces.items()}, costs


def common_probe(network, images, row_ids, *, seed):
    require(len(images) == len(row_ids) and len(images) > 0, "Probe IDs differ")
    before = network.state_digest()
    model = ReferenceNetwork(replace(network.config, dynamics_version=2, integration_substeps=16))
    for name in ("weights", "theta", "delays"):
        setattr(model, name, getattr(network, name).copy())
    spikes, volts = [], []
    for image, row in zip(images, row_ids):
        model.reset_transients()
        tape = poisson_tape(image, model.config, seed=np.random.SeedSequence([seed, int(row), 0]))
        s, _, v = model.advance(tape, learn=False, adapt=False)
        spikes.append(s)
        volts.append(v)
        # No rest is needed: electrical state is discarded before the next row.
    require(network.state_digest() == before, "Probe changed source state")
    np.testing.assert_array_equal(model.weights, network.weights)
    np.testing.assert_array_equal(model.theta, network.theta)
    return dict(spikes=np.asarray(spikes), voltages=np.asarray(volts))


def make_plan(source_root, output):
    if output.exists():
        raise FileExistsError(output)
    source_root = source_root.resolve()
    parent_path = source_root/"plan.json"
    parent = json.loads(parent_path.read_text())
    completed = json.loads((source_root/"summary.json").read_text())
    require(completed["complete"] and completed["source_unchanged"] and completed["plan_sha256"] == sha256(parent_path),
            "Historical training diagnostic is incomplete")
    require(parent["seeds"] == [201, 202, 203] and parent["training_images"] == 50 and parent["base_dt_ms"] == .5,
            "Unexpected historical protocol")
    inputs = {str(p): sha256(p) for p in (parent_path, source_root/"summary.json", source_root/"images.npz")}
    require(inputs[str(source_root/"images.npz")] == parent["images_sha256"], "Historical image snapshot changed")
    for name, digest in parent["source_sha256"].items():
        archived = source_root/"source_snapshot"/name
        require(sha256(archived) == digest, f"Historical source snapshot changed: {name}")
        inputs[str(archived)] = digest
    for name, digest in parent["input_files"].items():
        require(sha256(name) == digest, f"Historical input changed: {name}")
        inputs[name] = digest
    rows = []
    for row in parent["initial_rows"]:
        seed = row["seed"]
        path = source_root/f"seed_{seed}"/"summary.json"
        study = json.loads(path.read_text())
        require(study["complete"] and study["source_unchanged"] and study["coarse_training_exact_replay"] and
                study["plan_sha256"] == sha256(parent_path) and study["seed"] == seed, "Historical seed is incomplete")
        inputs[str(path)] = sha256(path)
        initial = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
        require(initial.state_digest() == row["state_sha256"] and initial.config.dynamics_version == 1 and initial.step_count == 0,
                "Historical initialization differs")
        checkpoints = {a: study["checkpoints"][f"1__{a}"] for a in ARMS}
        for entry in [dict(path=row["checkpoint"], sha256=row["checkpoint_sha256"]), *checkpoints.values()]:
            require(directory_digest(entry["path"]) == entry["sha256"], "Historical checkpoint changed")
            for f in Path(entry["path"]).rglob("*"):
                if f.is_file():
                    inputs[str(f)] = sha256(f)
        require(len(parent["records"][str(seed)]) == 50, "Wrong prefix length")
        np.testing.assert_array_equal([r["training_index"] for r in parent["records"][str(seed)]],
                                       np.random.default_rng(seed).permutation(len(parent["train_ids"]))[:50])
        rows.append({k: row[k] for k in ("seed", "checkpoint", "checkpoint_metadata", "state_sha256")} |
                    {"legacy_final": checkpoints, "configs": {v: asdict(variant_copy(initial, v).config) for v in VARIANTS}})
    with np.load(source_root/"images.npz", allow_pickle=False) as data:
        for field in ("train_ids", "probe_ids", "probe_labels"):
            np.testing.assert_array_equal(data[field], parent[field])
        require(data["train_images"].shape == (1000, 784) and data["probe_images"].shape == (20, 784), "Unexpected data dimensions")
        require(not set(data["train_ids"]) & set(data["probe_ids"]), "Training/probe overlap")
        np.testing.assert_array_equal(np.bincount(data["probe_labels"], minlength=10), np.full(10, 2))
    names = ("examples/mnist_paper_version_check.py", "examples/mnist_paper_reference.py", "examples/training_checkpoint.py")
    hashes = {name: sha256(REPO/name) for name in names}
    plan = dict(created_at=time.time(), protocol_version=1, source_root=str(source_root),
                source_sha256=hashes, input_files=inputs, seeds=parent["seeds"], rows=rows,
                records=parent["records"], train_ids=parent["train_ids"], probe_ids=parent["probe_ids"],
                training_images=50, milestones=[10, 25, 50], variants={k: list(v) for k, v in VARIANTS.items()},
                probe_dynamics_version=2, probe_substeps=16, poisson_seed_offset=parent["poisson_seed_offset"],
                numpy=np.__version__, interpretation=INTERPRETATION)
    output.mkdir(parents=True)
    for name in names:
        target = output/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO/name).read_bytes())
    atomic_json(output/"plan.json", plan)
    print("PLANNED: three seeds, 50 matched images, v1/v2/v2_half, two learning arms, 20 common-grid probes", flush=True)


def checked_plan(output):
    plan = json.loads((output/"plan.json").read_text())
    require(plan["protocol_version"] == 1 and plan["variants"] == {k: list(v) for k, v in VARIANTS.items()} and
            plan["training_images"] == 50 and plan["milestones"] == [10, 25, 50] and
            plan["probe_dynamics_version"] == 2 and plan["probe_substeps"] == 16 and plan["numpy"] == np.__version__, "Protocol differs")
    for name, digest in plan["source_sha256"].items():
        require(sha256(REPO/name) == digest == sha256(output/"source_snapshot"/name), f"Current source changed: {name}")
    for name, digest in plan["input_files"].items():
        require(sha256(name) == digest, f"Input changed: {name}")
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    require(seed in plan["seeds"], "Seed not in plan")
    folder = output/f"seed_{seed}"
    folder.mkdir()
    plan_hash = sha256(output/"plan.json")
    row = next(r for r in plan["rows"] if r["seed"] == seed)
    initial = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
    require(initial.state_digest() == row["state_sha256"], "Initial state differs")
    started = time.monotonic()
    result = dict(complete=False, seed=seed, plan_sha256=plan_hash, milestones={})
    atomic_json(folder/"summary.json", result)
    with np.load(Path(plan["source_root"])/"images.npz", allow_pickle=False) as data:
        train, probe = data["train_images"], data["probe_images"]
    def progress(n, models, history, costs):
        if n in plan["milestones"]:
            result["milestones"][str(n)] = state_metrics(initial, models)
            atomic_json(folder/"summary.json", result)
        if n % 5 == 0:
            atomic_json(folder/"progress.json", dict(completed_images=n, history=history, costs=costs,
                wall_seconds=time.monotonic()-started))
            print(f"TRAIN-VERSION seed={seed}: {n}/50", flush=True)
    models, history, training_responses, costs = train_coupled(initial, train, plan["records"][str(seed)], seed=seed, callback=progress)
    result.update(training_seconds=time.monotonic()-started, history=history, costs=costs, checkpoints={})
    for v, arms in models.items():
        for arm, network in arms.items():
            require(asdict(network.config) == row["configs"][v], "Dynamics configuration changed")
            if v == "v1":
                old = row["legacy_final"][arm]
                require(network.state_digest() == old["state_sha256"], "Historical final full state did not replay")
            name = f"{v}__{arm}"
            path = folder/"checkpoints"/name
            metadata = dict(plan_sha256=plan_hash, seed=seed, name=name, completed_images=50)
            network.save(path, metadata)
            result["checkpoints"][name] = dict(path=str(path.resolve()), metadata=metadata,
                sha256=directory_digest(path), state_sha256=network.state_digest())
    atomic_json(folder/"summary.json", result)
    probes = {}
    timer = time.monotonic()
    for v, arms in models.items():
        for arm, network in arms.items():
            name = f"{v}__{arm}"
            probes[name] = common_probe(network, probe, plan["probe_ids"], seed=seed+plan["poisson_seed_offset"])
            print(f"PROBE-VERSION seed={seed}: {name}", flush=True)
    result["probe_seconds"] = time.monotonic()-timer
    payload = {f"{kind}__{name}__{field}": value for kind, group in (("training", training_responses), ("probe", probes))
               for name, response in group.items() for field, value in response.items()}
    np.savez_compressed(folder/"responses.npz", **payload, probe_ids=plan["probe_ids"],
                        identity=json.dumps(dict(plan_sha256=plan_hash, seed=seed)))
    result.update(responses_sha256=sha256(folder/"responses.npz"), training_response_metrics=response_metrics(training_responses),
                  probe_metrics=response_metrics(probes))
    checked_plan(output)
    require(sha256(output/"plan.json") == plan_hash and initial.state_digest() == row["state_sha256"], "Plan or initialization changed")
    result.update(complete=True, source_unchanged=True, legacy_full_state_replay=True, wall_seconds=time.monotonic()-started)
    atomic_json(folder/"summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output/"summary.json").exists():
        raise FileExistsError(output/"summary.json")
    plan = checked_plan(output)
    plan_hash = sha256(output/"plan.json")
    studies = []
    for row in plan["rows"]:
        seed = row["seed"]
        folder = output/f"seed_{seed}"
        result = json.loads((folder/"summary.json").read_text())
        require(result["complete"] and result["source_unchanged"] and result["legacy_full_state_replay"] and
                result["plan_sha256"] == plan_hash and result["seed"] == seed, "Incomplete seed result")
        initial = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
        models = {v: {} for v in VARIANTS}
        require(set(result["checkpoints"]) == {f"{v}__{a}" for v in VARIANTS for a in ARMS}, "Missing checkpoint")
        for name, item in result["checkpoints"].items():
            require(directory_digest(item["path"]) == item["sha256"], "Learned checkpoint changed")
            network = ReferenceNetwork.load(item["path"], item["metadata"])
            v, arm = name.split("__")
            require(network.state_digest() == item["state_sha256"] and asdict(network.config) == row["configs"][v], "Learned state differs")
            if v == "v1":
                require(network.state_digest() == row["legacy_final"][arm]["state_sha256"], "Legacy replay differs")
            models[v][arm] = network
        require(state_metrics(initial, models) == result["milestones"]["50"], "Final state metrics differ")
        require(sha256(folder/"responses.npz") == result["responses_sha256"], "Responses changed")
        with np.load(folder/"responses.npz", allow_pickle=False) as data:
            require(json.loads(str(data["identity"])) == dict(plan_sha256=plan_hash, seed=seed), "Response identity differs")
            np.testing.assert_array_equal(data["probe_ids"], plan["probe_ids"])
            for kind, metric in (("training", "training_response_metrics"), ("probe", "probe_metrics")):
                responses = {f"{v}__{a}": {f: data[f"{kind}__{v}__{a}__{f}"] for f in ("spikes", "voltages")}
                             for v in VARIANTS for a in ARMS}
                require(response_metrics(responses) == result[metric], "Response metrics differ")
        studies.append(result)
    checked_plan(output)
    require(sha256(output/"plan.json") == plan_hash, "Plan changed")
    atomic_json(output/"summary.json", dict(complete=True, source_unchanged=True, plan_sha256=plan_hash,
                seeds=plan["seeds"], interpretation=INTERPRETATION, results=studies))
    print(f"SUMMARIZED: {output/'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("results/20260913-paper-training-grid"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.source_root, args.output)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
