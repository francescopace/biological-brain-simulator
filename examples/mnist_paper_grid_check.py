"""Paired timestep sensitivity of frozen reference networks and frozen decoders.

Replay existing 0.5 ms input events at the same physical times on finer grids.
This diagnoses inference on reused images, not training convergence or a new
accuracy benchmark. No fitting, learning, parameter selection or default change.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from examples import mnist_paper_fresh_evaluation as fresh
from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork, poisson_tape
from examples.training_checkpoint import array_digest, atomic_json


FACTORS = (1, 2, 4)
POLICIES = ("fixed_exposure", "adaptive_retry")
FIELDS = ("spikes", "voltages", "attempts", "failed")
INTERPRETATION = (
    "Frozen-inference grid sensitivity on 100 reused, class-balanced images, "
    "three fixed networks per condition and all six previously frozen decoders. "
    "No training, refitting, selection or canonical-test evaluation. Original "
    "0.5 ms Bernoulli-Poisson events are replayed at identical physical times on "
    "0.5/0.25/0.125 ms grids; input delays and all physical parameters are fixed. "
    "Primary fixed_exposure scores the original coarse accepted attempt on every "
    "grid, even when below the spike threshold. Secondary adaptive_retry scores "
    "each grid's first accepted attempt; exhaustion is an explicit abstention "
    "counted as an error, never an omitted image. Threshold, recurrent-event, "
    "refractory and voltage-sampling discretizations change with the grid. "
    "Networks and decoders were trained at 0.5 ms, so this is not convergence of "
    "training, a continuous-time Poisson test or equivalence to Brian/the paper. "
    "No pass tolerance is asserted. Bootstrap intervals resample shared images "
    "within digit with seeds paired, conditional on these networks/readouts; "
    "secondary intervals are unadjusted. The finest grid is not ground truth."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_legacy_protocol(config):
    require(config.dynamics_version == 1 and config.integration_substeps == 1,
            "Historical grid protocol requires v1 dynamics; v2 needs a new substep-aware protocol")


def balanced_indices(labels, *, per_class=10, seed=20260914):
    labels = np.asarray(labels)
    require(labels.ndim == 1 and labels.dtype.kind in "iu" and
            np.isin(labels, np.arange(10)).all() and type(per_class) is int and per_class > 0,
            "Expected digit labels and a positive class budget")
    rng = np.random.default_rng(seed)
    indices = []
    for digit in range(10):
        candidates = np.flatnonzero(labels == digit)
        require(len(candidates) >= per_class, f"Not enough rows for digit {digit}")
        indices.extend(rng.permutation(candidates)[:per_class].tolist())
    rng.shuffle(indices)
    return indices


def lift_tape(tape, factor):
    require(type(factor) is int and factor in FACTORS, "Unsupported grid factor")
    tape = np.asarray(tape)
    require(tape.ndim == 2 and tape.dtype == bool and len(tape) > 0, "Expected boolean input tape")
    if factor == 1:
        return tape
    lifted = np.zeros((len(tape) * factor, tape.shape[1]), dtype=bool)
    lifted[::factor] = tape
    return lifted


def refined_network(network, factor):
    require(type(factor) is int and factor in FACTORS, "Unsupported grid factor")
    refined = copy.deepcopy(network)
    refined.config = replace(network.config, dt=network.config.dt / factor)
    refined.config.validate()
    refined.delays = network.delays.astype(np.int64) * factor
    refined.reset_transients()
    np.testing.assert_array_equal(refined.delays * refined.config.dt, network.delays * network.config.dt)
    return refined


def paired_responses(network, images, row_ids, original_attempts, *, seed, progress=None):
    """Run both retry policies on shared trajectories without repeating prefixes."""
    require_legacy_protocol(network.config)
    images, row_ids, original_attempts = map(np.asarray, (images, row_ids, original_attempts))
    c = network.config
    require(images.ndim == 2 and images.shape == (len(row_ids), c.n_input) and len(images) > 0
            and row_ids.ndim == 1 and row_ids.dtype.kind in "iu" and np.all(row_ids >= 0)
            and len(set(row_ids.tolist())) == len(row_ids), "Expected aligned unique image rows")
    require(original_attempts.shape == row_ids.shape and original_attempts.dtype.kind in "iu"
            and np.all((original_attempts >= 1) & (original_attempts <= c.max_attempts)),
            "Original retry counts exceed the protocol")
    before = network.state_digest()
    grids = {f: refined_network(network, f) for f in FACTORS}
    result = {p: {f: {"spikes": np.zeros((len(images), c.n_exc), dtype=np.int32),
                     "voltages": np.zeros((len(images), c.n_exc)),
                     "attempts": np.zeros(len(images), dtype=np.int64),
                     "failed": np.zeros(len(images), dtype=bool)} for f in FACTORS} for p in POLICIES}
    for i, (image, row_id, exposure) in enumerate(zip(images, row_ids, original_attempts)):
        for grid in grids.values():
            grid.reset_transients()
        fixed_done, adaptive_done = set(), set()
        for attempt in range(c.max_attempts):
            tape = poisson_tape(image, c, seed=np.random.SeedSequence([seed, int(row_id), attempt]), attempt=attempt)
            for factor, grid in grids.items():
                if factor in fixed_done and factor in adaptive_done:
                    continue
                spikes, _, volts = grid.advance(lift_tape(tape, factor), learn=False, adapt=False)
                grid.rest(learn=False, adapt=False)
                accepted = spikes.sum() >= c.min_spikes
                save_fixed = attempt + 1 == exposure
                save_adaptive = factor not in adaptive_done and (accepted or attempt + 1 == c.max_attempts)
                for policy, save in zip(POLICIES, (save_fixed, save_adaptive)):
                    if save:
                        target = result[policy][factor]
                        target["spikes"][i], target["voltages"][i] = spikes, volts
                        target["attempts"][i] = attempt + 1
                        target["failed"][i] = policy == "adaptive_retry" and not accepted
                if save_fixed:
                    fixed_done.add(factor)
                if save_adaptive:
                    adaptive_done.add(factor)
            if len(fixed_done) == len(adaptive_done) == len(FACTORS):
                break
        require(len(fixed_done) == len(adaptive_done) == len(FACTORS), "Incomplete paired trajectory")
        if progress is not None and ((i + 1) % 10 == 0 or i + 1 == len(images)):
            progress(i + 1, len(images))
    for grid in grids.values():
        np.testing.assert_array_equal(grid.weights, network.weights)
        np.testing.assert_array_equal(grid.theta, network.theta)
        np.testing.assert_array_equal(grid.delays * grid.config.dt, network.delays * c.dt)
    require(network.state_digest() == before, "Source network mutated")
    return result


def predictions(states, assignments, response):
    predicted = fresh.predict_all(states, assignments, response["spikes"], response["voltages"])
    for value in predicted.values():
        value[response["failed"]] = -1
    return predicted


def response_difference(source, target):
    a, b = source["spikes"].astype(float), target["spikes"].astype(float)
    va, vb = source["voltages"], target["voltages"]
    ca, cb = va - va.mean(axis=-1, keepdims=True), vb - vb.mean(axis=-1, keepdims=True)
    return {"relative_spike_l1": float(np.abs(b - a).sum() / max(a.sum(), 1.)),
            "same_spike_vector_fraction": float(np.all(a == b, axis=-1).mean()),
            "centered_voltage_rmse_mv": float(np.sqrt(np.mean((cb - ca) ** 2))),
            "mean_total_spike_change": float((b - a).sum(axis=-1).mean()),
            "changed_attempt_fraction": float((source["attempts"] != target["attempts"]).mean())}


def source_response(row, plan):
    with np.load(row["fresh_cache"], allow_pickle=False) as data:
        require(json.loads(str(data["identity"])) == {"plan_sha256": plan["fresh_plan_sha256"],
                "state_sha256": row["state_sha256"]}, "Fresh cache identity differs")
        indices = plan["indices"]
        np.testing.assert_array_equal(data["row_ids"][indices], plan["row_ids"])
        np.testing.assert_array_equal(data["labels"][indices], plan["labels"])
        result = {k: data[k][indices] for k in FIELDS[:-1]}
    result["failed"] = np.zeros(len(indices), dtype=bool)
    return result


def check_replay(responses, baseline):
    for policy in POLICIES:
        for field in FIELDS:
            np.testing.assert_array_equal(responses[policy][1][field], baseline[field])


def make_plan(fresh_root, output):
    if output.exists():
        raise FileExistsError(output)
    fresh_root = fresh_root.resolve()
    source = fresh.checked_plan(fresh_root)
    source_hash = fresh.sha256(fresh_root / "plan.json")
    completed = json.loads((fresh_root / "summary.json").read_text())
    require(completed["complete"] and completed["source_unchanged"] and
            completed["plan_sha256"] == source_hash, "Fresh evaluation is not complete")
    indices = balanced_indices(source["labels"])
    files = {str(fresh_root / name): fresh.sha256(fresh_root / name)
             for name in ("plan.json", "summary.json", "images.npz")}
    rows = []
    for row in source["rows"]:
        envelope = json.loads((Path(row["checkpoint"]) / "metadata.json").read_text())
        require_legacy_protocol(ReferenceConfig.from_dict(envelope["config"]))
        seed_summary = fresh_root / f"seed_{row['seed']}" / "summary.json"
        study = json.loads(seed_summary.read_text())
        require(study["complete"] and study["source_unchanged"] and study["plan_sha256"] == source_hash
                and study["seed"] == row["seed"], "Fresh seed is incomplete or misidentified")
        files[str(seed_summary)] = fresh.sha256(seed_summary)
        cache = fresh_root / f"seed_{row['seed']}" / f"{row['condition']}-responses.npz"
        scores = study["results"][row["condition"]]
        require(fresh.sha256(cache) == scores["cache_sha256"], "Fresh response cache changed")
        files[str(cache)] = scores["cache_sha256"]
        rows.append({k: v for k, v in row.items() if k not in ("source_cache", "old_predictions")} |
                    {"fresh_cache": str(cache), "baseline_predictions": {
                        m: np.asarray(scores["methods"][m]["predictions"])[indices].tolist() for m in fresh.DECODERS}})
    sources = dict(source["source_sha256"])
    sources["examples/mnist_paper_grid_check.py"] = fresh.sha256(__file__)
    plan = {"created_at": time.time(), "fresh_root": str(fresh_root), "fresh_plan_sha256": source_hash,
        "tuning_plan_sha256": source["tuning_plan_sha256"], "selection_sha256": source["selection_sha256"],
        "seeds": source["seeds"], "conditions": list(fresh.CONDITIONS), "methods": list(fresh.DECODERS),
        "rows": rows, "indices": indices, "row_ids": np.asarray(source["row_ids"])[indices].tolist(),
        "labels": np.asarray(source["labels"])[indices].tolist(), "selection_seed": 20260914, "per_class": 10,
        "factors": list(FACTORS), "base_dt_ms": .5, "policies": list(POLICIES),
        "poisson_seed_offset": source["poisson_seed_offset"], "input_files": files, "source_sha256": sources,
        "bootstrap_seed": 20260914, "bootstrap_repeats": 5000, "interpretation": INTERPRETATION}
    for row in rows:
        original = source_response(row, plan)
        replayed = predictions(fresh.load_decoders(row, plan), row["assignments"], original)
        for method, pred in replayed.items():
            np.testing.assert_array_equal(pred, row["baseline_predictions"][method])
    output.mkdir(parents=True)
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((fresh.REPO / name).read_bytes())
    atomic_json(output / "plan.json", plan)
    print(f"PLANNED: {len(indices)} reused images, factors={FACTORS}, two retry policies", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    source = fresh.checked_plan(Path(plan["fresh_root"]))
    for name, digest in plan["source_sha256"].items():
        require(fresh.sha256(fresh.REPO / name) == digest == fresh.sha256(output / "source_snapshot" / name),
                f"Study source changed: {name}")
    for name, digest in plan["input_files"].items():
        require(fresh.sha256(name) == digest, f"Study input changed: {name}")
    require(plan["indices"] == balanced_indices(source["labels"], per_class=plan["per_class"],
            seed=plan["selection_seed"]), "Diagnostic cohort changed")
    np.testing.assert_array_equal(np.asarray(source["row_ids"])[plan["indices"]], plan["row_ids"])
    np.testing.assert_array_equal(np.asarray(source["labels"])[plan["indices"]], plan["labels"])
    require(plan["factors"] == list(FACTORS) and plan["policies"] == list(POLICIES)
            and plan["methods"] == list(fresh.DECODERS) and plan["conditions"] == list(fresh.CONDITIONS),
            "Grid protocol differs")
    return plan


def run_seed(output, seed):
    plan = checked_plan(output)
    require(seed in plan["seeds"], "Seed not in plan")
    folder = output / f"seed_{seed}"
    folder.mkdir()
    plan_hash = fresh.sha256(output / "plan.json")
    started = time.monotonic()
    result = {"complete": False, "seed": seed, "plan_sha256": plan_hash, "results": {}}
    atomic_json(folder / "summary.json", result)
    with np.load(Path(plan["fresh_root"]) / "images.npz", allow_pickle=False) as data:
        images = data["images"][plan["indices"]]
    labels = np.asarray(plan["labels"])
    for row in [r for r in plan["rows"] if r["seed"] == seed]:
        condition = row["condition"]
        network = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
        require(network.state_digest() == row["state_sha256"] and network.config.dt == plan["base_dt_ms"],
                "Source network or base timestep differs")
        baseline = source_response(row, plan)
        states = fresh.load_decoders(row, plan)
        timer = time.monotonic()
        responses = paired_responses(network, images, plan["row_ids"], baseline["attempts"],
            seed=seed + plan["poisson_seed_offset"],
            progress=lambda done, total: print(f"GRID seed={seed} {condition}: {done}/{total}", flush=True))
        check_replay(responses, baseline)
        payload, scores = {}, {}
        for policy in POLICIES:
            scores[policy] = {}
            for factor in FACTORS:
                response = responses[policy][factor]
                pred = predictions(states, row["assignments"], response)
                if factor == 1:
                    for method in fresh.DECODERS:
                        np.testing.assert_array_equal(pred[method], row["baseline_predictions"][method])
                prefix = f"{policy}__{factor}__"
                payload.update({prefix + name: value for name, value in response.items()})
                scores[policy][str(factor)] = {
                    "methods": {m: {"predictions": v.tolist(), "accuracy": float((v == labels).mean())}
                                for m, v in pred.items()},
                    "extra_attempts": int((response["attempts"] - 1).sum()),
                    "below_min_spikes": int((response["spikes"].sum(axis=1) < network.config.min_spikes).sum()),
                    "failed_images": int(response["failed"].sum()),
                    "mean_total_spikes": float(response["spikes"].sum(axis=1).mean())}
        artifact = folder / f"{condition}-responses.npz"
        np.savez_compressed(artifact, **payload, labels=labels, row_ids=plan["row_ids"],
            identity=json.dumps({"plan_sha256": plan_hash, "state_sha256": row["state_sha256"]}))
        require(network.state_digest() == row["state_sha256"], "Source network changed")
        result["results"][condition] = {"cache_sha256": fresh.sha256(artifact), "scores": scores,
            "baseline_exact_replay": True, "inference_seconds": time.monotonic() - timer,
            "frozen_parameters_sha256": array_digest(network.weights, network.theta, network.delays)}
        atomic_json(folder / "summary.json", result)
        print(f"DONE seed={seed} {condition}: joint fixed=" + json.dumps(
            {f: scores["fixed_exposure"][str(f)]["methods"]["joint_cv"]["accuracy"] for f in FACTORS}), flush=True)
    checked_plan(output)
    require(fresh.sha256(output / "plan.json") == plan_hash, "Plan changed during run")
    result.update(complete=True, source_unchanged=True, wall_seconds=time.monotonic() - started)
    atomic_json(folder / "summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    plan = checked_plan(output)
    plan_hash = fresh.sha256(output / "plan.json")
    labels = np.asarray(plan["labels"])
    records = {p: {f: {c: [] for c in fresh.CONDITIONS} for f in FACTORS} for p in POLICIES}
    for seed in plan["seeds"]:
        study = json.loads((output / f"seed_{seed}" / "summary.json").read_text())
        require(study["complete"] and study["source_unchanged"] and study["seed"] == seed
                and study["plan_sha256"] == plan_hash and set(study["results"]) == set(fresh.CONDITIONS),
                "Incomplete or mismatched seed result")
        for row in [r for r in plan["rows"] if r["seed"] == seed]:
            condition = row["condition"]
            saved = study["results"][condition]
            network = ReferenceNetwork.load(row["checkpoint"], row["checkpoint_metadata"])
            require(network.state_digest() == row["state_sha256"] and saved["baseline_exact_replay"]
                    and saved["frozen_parameters_sha256"] == array_digest(network.weights, network.theta, network.delays),
                    "Frozen parameter verification differs")
            states = fresh.load_decoders(row, plan)
            artifact = output / f"seed_{seed}" / f"{condition}-responses.npz"
            require(fresh.sha256(artifact) == saved["cache_sha256"], "Response cache changed")
            responses = {p: {} for p in POLICIES}
            with np.load(artifact, allow_pickle=False) as data:
                require(json.loads(str(data["identity"])) == {"plan_sha256": plan_hash,
                        "state_sha256": row["state_sha256"]}, "Grid cache identity differs")
                np.testing.assert_array_equal(data["row_ids"], plan["row_ids"])
                np.testing.assert_array_equal(data["labels"], labels)
                for policy in POLICIES:
                    for factor in FACTORS:
                        response = {k: data[f"{policy}__{factor}__{k}"] for k in FIELDS}
                        pred = predictions(states, row["assignments"], response)
                        score_set = saved["scores"][policy][str(factor)]
                        below = response["spikes"].sum(axis=1) < network.config.min_spikes
                        require(score_set["below_min_spikes"] == int(below.sum()) and
                                score_set["failed_images"] == int(response["failed"].sum()) and
                                score_set["extra_attempts"] == int((response["attempts"] - 1).sum()),
                                "Retry statistics differ")
                        for method in fresh.DECODERS:
                            score = saved["scores"][policy][str(factor)]["methods"][method]
                            np.testing.assert_array_equal(pred[method], score["predictions"])
                            require(float((pred[method] == labels).mean()) == score["accuracy"], "Score differs")
                        responses[policy][factor] = response
                        records[policy][factor][condition].append(response | {"predictions": pred, "below": int(below.sum())})
            check_replay(responses, source_response(row, plan))
    def interval(values):
        return fresh.paired_interval(values, labels, repeats=plan["bootstrap_repeats"], seed=plan["bootstrap_seed"])
    correct, preds = {}, {}
    for p in POLICIES:
        for f in FACTORS:
            for c in fresh.CONDITIONS:
                for m in fresh.DECODERS:
                    preds[p, f, c, m] = np.asarray([r["predictions"][m] for r in records[p][f][c]])
                    correct[p, f, c, m] = (preds[p, f, c, m] == labels).astype(float)
    metrics, contrasts, changes, retry_effects = {}, {}, {}, {}
    for policy in POLICIES:
        metrics[policy], contrasts[policy], changes[policy] = {}, {}, {}
        for factor in FACTORS:
            metrics[policy][str(factor)] = {c: {
                "accuracy": {m: float(correct[policy, factor, c, m].mean()) for m in fresh.DECODERS},
                "mean_total_spikes": float(np.mean([r["spikes"].sum(axis=1).mean() for r in records[policy][factor][c]])),
                "extra_attempts": [int((r["attempts"] - 1).sum()) for r in records[policy][factor][c]],
                "failed_images": [int(r["failed"].sum()) for r in records[policy][factor][c]],
                "below_min_spikes": [r["below"] for r in records[policy][factor][c]]
            } for c in fresh.CONDITIONS}
            contrasts[policy][str(factor)] = {m: {c: interval(correct[policy, factor, "stdp_normalized", m] -
                correct[policy, factor, c, m]) for c in ("initial", "normalization_only")}
                for m in ("joint_cv", "class_average")}
        for start, end in ((1, 2), (2, 4), (1, 4)):
            pair = f"{start}_to_{end}"
            changes[policy][pair] = {}
            for condition in fresh.CONDITIONS:
                a, b = records[policy][start][condition], records[policy][end][condition]
                stacked_a = {k: np.stack([r[k] for r in a]) for k in FIELDS}
                stacked_b = {k: np.stack([r[k] for r in b]) for k in FIELDS}
                changes[policy][pair][condition] = response_difference(stacked_a, stacked_b) | {
                    "methods": {m: {"accuracy_delta_pp": interval(correct[policy, end, condition, m] -
                        correct[policy, start, condition, m]), "prediction_disagreement_fraction":
                        float((preds[policy, end, condition, m] != preds[policy, start, condition, m]).mean())}
                        for m in fresh.DECODERS}}
    for factor in FACTORS:
        retry_effects[str(factor)] = {c: {m: {
            "accuracy_delta_pp": interval(correct["adaptive_retry", factor, c, m] - correct["fixed_exposure", factor, c, m]),
            "prediction_disagreement_fraction": float((preds["adaptive_retry", factor, c, m] !=
                                                       preds["fixed_exposure", factor, c, m]).mean())}
            for m in fresh.DECODERS} for c in fresh.CONDITIONS}
    result = {"complete": True, "source_unchanged": True, "baseline_exact_replay": True,
        "plan_sha256": plan_hash, "images": len(labels), "seeds": plan["seeds"],
        "interpretation": INTERPRETATION, "metrics": metrics, "contrasts": contrasts,
        "grid_changes": changes, "adaptive_minus_fixed": retry_effects}
    checked_plan(output)
    require(fresh.sha256(output / "plan.json") == plan_hash, "Plan changed during summary")
    atomic_json(output / "summary.json", result)
    print(f"SUMMARIZED: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, default=Path("results/20260913-paper-fresh-evaluation"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args.fresh_root, args.output)
    elif args.mode == "run":
        if args.seed is None:
            parser.error("run needs --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
