"""Frozen weight/theta factorial probe of the early-training grid checkpoints.

No training, classifier fitting, accuracy calculation or default changes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from examples import mnist_paper_training_grid as training
from examples.mnist_paper_reference import ReferenceNetwork
from examples.training_checkpoint import atomic_json, directory_digest


FACTORS = training.FACTORS
ARMS = training.ARMS
FIELDS = ("spikes", "voltages", "attempts")
PAIRS = ((2, 4), (1, 2), (1, 4))
require = training.require
sha256 = training.pilot.sha256
INTERPRETATION = (
    "Frozen post-training weight/theta swaps on all three 50-image training-grid "
    "checkpoints per condition and seed. Full 3x3 weight-grid/theta-grid factorial "
    "within the STDP and no-STDP arms; no swaps between seeds or arms. Same 20 "
    "reused balanced probes, fixed .125 ms inference, coupled .5 ms Poisson events, "
    "one presentation, no retries, transient reset and frozen parameters. Original "
    "diagonal cells must exactly reproduce saved probes. Identical no-STDP weights "
    "must give identical responses at fixed theta. Primary training-grid contrast "
    "is .25 to .125 ms; .5 to .25 and .5 to .125 are secondary. For each pair, "
    "report weight-only, theta-only, joint and interaction response changes at "
    "both endpoint anchors. Signed components sum; their norms do not. No unique "
    "percentage attribution of response change to weights/theta is asserted. "
    "This interventions-at-readout diagnostic does not retrain with frozen theta, "
    "undo theta's influence on learned weights, measure accuracy, establish grid "
    "convergence, prove a bug or select a better model/timestep. No refitting or "
    "parameter selection. Checkpoints and production defaults are unchanged."
)


def hybrid(weight_source, theta_source):
    """Copy only theta across physically compatible checkpoints, never alias it."""
    require(replace(weight_source.config, dt=theta_source.config.dt) == theta_source.config,
            "Checkpoint physical configurations differ")
    np.testing.assert_array_equal(weight_source.delays * weight_source.config.dt,
                                  theta_source.delays * theta_source.config.dt)
    require(theta_source.theta.shape == weight_source.theta.shape and np.isfinite(theta_source.theta).all(),
            "Invalid theta donor")
    result = copy.deepcopy(weight_source)
    result.theta = theta_source.theta.copy()
    return result


def key(arm, weight_factor, theta_factor):
    return f"{arm}__w{weight_factor}__t{theta_factor}"


def delta_metrics(spikes, volts, baseline):
    centered = volts - volts.mean(axis=-1, keepdims=True)
    denominator = float(np.abs(baseline["spikes"]).sum())
    return {"relative_spike_l1": float(np.abs(spikes).sum() / denominator) if denominator else None,
            "mean_absolute_spike_difference": float(np.abs(spikes).sum(axis=-1).mean()),
            "same_spike_vector_fraction": float(np.all(spikes == 0, axis=-1).mean()),
            "mean_total_spike_change": float(spikes.sum(axis=-1).mean()),
            "centered_voltage_rmse_mv": float(np.sqrt(np.mean(centered ** 2)))}


def factorial_metrics(a, b, c, d):
    """A=(Wa,Ta), B=(Wb,Ta), C=(Wa,Tb), D=(Wb,Tb)."""
    components = {}
    for field in ("spikes", "voltages"):
        aa, bb, cc, dd = (x[field].astype(np.float64) for x in (a, b, c, d))
        require(aa.ndim == 2 and aa.shape == bb.shape == cc.shape == dd.shape
                and aa.size > 0 and all(np.isfinite(x).all() for x in (aa, bb, cc, dd)),
                "Invalid factorial response arrays")
        components[field] = {"weights_only": bb - aa, "theta_only": cc - aa,
            "joint": dd - aa, "interaction": (dd - cc) - (bb - aa),
            "weights_after_theta": dd - cc, "theta_after_weights": dd - bb}
        np.testing.assert_allclose(components[field]["joint"], components[field]["weights_only"] +
            components[field]["theta_only"] + components[field]["interaction"], rtol=0, atol=1e-12)
    return {name: delta_metrics(components["spikes"][name], components["voltages"][name], a)
            for name in components["spikes"]}


def all_metrics(responses):
    return {arm: {f"{a}_to_{b}": factorial_metrics(
        responses[key(arm, a, a)], responses[key(arm, b, a)],
        responses[key(arm, a, b)], responses[key(arm, b, b)]) for a, b in PAIRS} for arm in ARMS}


def verify_replays(responses, baseline):
    for arm in ARMS:
        for factor in FACTORS:
            for field in FIELDS:
                np.testing.assert_array_equal(responses[key(arm, factor, factor)][field],
                                              baseline[f"{factor}__{arm}__{field}"])
    for theta in FACTORS:
        for weight in FACTORS[1:]:
            for field in FIELDS:
                np.testing.assert_array_equal(responses[key(ARMS[0], weight, theta)][field],
                                              responses[key(ARMS[0], 1, theta)][field])


def make_plan(source_root, output):
    if output.exists():
        raise FileExistsError(output)
    source_root = source_root.resolve()
    parent = training.checked_plan(source_root)
    source_hash = sha256(source_root / "plan.json")
    summary = json.loads((source_root / "summary.json").read_text())
    require(summary["complete"] and summary["source_unchanged"] and summary["plan_sha256"] == source_hash,
            "Source training diagnostic is incomplete")
    paths = [source_root / name for name in ("plan.json", "summary.json", "images.npz")]
    rows = []
    for seed in parent["seeds"]:
        path = source_root / f"seed_{seed}" / "summary.json"
        result = json.loads(path.read_text())
        require(result["complete"] and result["source_unchanged"] and result["seed"] == seed and
                result["plan_sha256"] == source_hash, "Source seed result differs")
        require(set(result["checkpoints"]) == {f"{f}__{a}" for f in FACTORS for a in ARMS}, "Missing source state")
        cache = path.parent / "probes.npz"
        require(sha256(cache) == result["probe_sha256"], "Source probes changed")
        paths.extend((path, cache))
        for name, checkpoint in result["checkpoints"].items():
            require(checkpoint["metadata"] == {"plan_sha256": source_hash, "seed": seed,
                "name": name, "completed_images": 50}, "Source checkpoint identity differs")
            require(directory_digest(checkpoint["path"]) == checkpoint["sha256"], "Source checkpoint changed")
        rows.append({"seed": seed, "checkpoints": result["checkpoints"], "baseline_probes": str(cache)})
    sources = dict(parent["source_sha256"])
    sources["examples/mnist_paper_theta_swap.py"] = sha256(__file__)
    output.mkdir(parents=True)
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((training.pilot.REPO / name).read_bytes())
    plan = {"created_at": time.time(), "source_root": str(source_root), "source_plan_sha256": source_hash,
        "source_sha256": sources, "input_files": {str(p): sha256(p) for p in paths}, "rows": rows,
        "seeds": parent["seeds"], "factors": list(FACTORS), "arms": list(ARMS),
        "pairs": [list(p) for p in PAIRS], "probe_ids": parent["probe_ids"], "probe_factor": parent["probe_factor"],
        "base_dt_ms": parent["base_dt_ms"], "poisson_seed_offset": parent["poisson_seed_offset"],
        "interpretation": INTERPRETATION}
    atomic_json(output / "plan.json", plan)
    print("PLANNED: 3x3 weight/theta swaps, two arms, three seeds, same 20 probes", flush=True)


def checked_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    parent = training.checked_plan(Path(plan["source_root"]))
    for name, digest in plan["source_sha256"].items():
        require(sha256(training.pilot.REPO / name) == digest == sha256(output / "source_snapshot" / name),
                f"Source changed: {name}")
    for path, digest in plan["input_files"].items():
        require(sha256(path) == digest, f"Input changed: {path}")
    for row in plan["rows"]:
        for checkpoint in row["checkpoints"].values():
            require(directory_digest(checkpoint["path"]) == checkpoint["sha256"], "Checkpoint changed")
    require(plan["probe_ids"] == parent["probe_ids"] and plan["probe_factor"] == 4 and
            plan["factors"] == list(FACTORS) and plan["arms"] == list(ARMS) and
            plan["pairs"] == [list(p) for p in PAIRS] and plan["seeds"] == parent["seeds"], "Protocol differs")
    return plan


def load_models(row):
    models = {a: {} for a in ARMS}
    for name, checkpoint in row["checkpoints"].items():
        model = ReferenceNetwork.load(checkpoint["path"], checkpoint["metadata"])
        require(model.state_digest() == checkpoint["state_sha256"], "Checkpoint state differs")
        factor, arm = name.split("__")
        models[arm][int(factor)] = model
    for factor in FACTORS[1:]:
        np.testing.assert_array_equal(models[ARMS[0]][1].weights, models[ARMS[0]][factor].weights)
    return models


def baseline_arrays(row, plan):
    with np.load(row["baseline_probes"], allow_pickle=False) as data:
        require(json.loads(str(data["identity"])) == {"plan_sha256": plan["source_plan_sha256"], "seed": row["seed"]},
                "Source probe identity differs")
        np.testing.assert_array_equal(data["row_ids"], plan["probe_ids"])
        return {k: data[k] for k in data.files if k not in ("identity", "row_ids")}


def run_seed(output, seed):
    plan = checked_plan(output)
    require(seed in plan["seeds"], "Seed not in plan")
    folder = output / f"seed_{seed}"
    folder.mkdir()
    plan_hash = sha256(output / "plan.json")
    row = next(r for r in plan["rows"] if r["seed"] == seed)
    models = load_models(row)
    baseline = baseline_arrays(row, plan)
    before = {f"{a}__{f}": m.state_digest() for a, group in models.items() for f, m in group.items()}
    base_config = models[ARMS[0]][1].config
    training.inference.require_legacy_protocol(base_config)
    require(base_config.dt == plan["base_dt_ms"], "Base grid differs")
    with np.load(Path(plan["source_root"]) / "images.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["probe_ids"], plan["probe_ids"])
        images = data["probe_images"]
    started = time.monotonic()
    result = {"complete": False, "seed": seed, "plan_sha256": plan_hash}
    atomic_json(folder / "summary.json", result)
    responses = {}
    for arm in ARMS:
        for weight in FACTORS:
            for theta in FACTORS:
                name = key(arm, weight, theta)
                model = hybrid(models[arm][weight], models[arm][theta])
                responses[name] = training.common_probe(model, images, plan["probe_ids"],
                    seed=seed + plan["poisson_seed_offset"], base_config=base_config, factor=plan["probe_factor"])
                print(f"SWAP seed={seed}: {name} ({len(responses)}/18)", flush=True)
    verify_replays(responses, baseline)
    path = folder / "responses.npz"
    payload = {f"{name}__{field}": value for name, response in responses.items() for field, value in response.items()}
    np.savez_compressed(path, **payload, row_ids=plan["probe_ids"], identity=json.dumps({"plan_sha256": plan_hash, "seed": seed}))
    require(before == {f"{a}__{f}": m.state_digest() for a, group in models.items() for f, m in group.items()},
            "Loaded checkpoints were mutated")
    checked_plan(output)
    require(sha256(output / "plan.json") == plan_hash, "Plan changed during run")
    result.update(complete=True, source_unchanged=True, diagonal_exact_replay=True, fixed_theta_control_exact=True,
        responses_sha256=sha256(path), metrics=all_metrics(responses), wall_seconds=time.monotonic() - started)
    atomic_json(folder / "summary.json", result)
    print(f"COMPLETE seed={seed}", flush=True)


def summarize(output):
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    plan = checked_plan(output)
    plan_hash = sha256(output / "plan.json")
    studies = []
    for row in plan["rows"]:
        seed = row["seed"]
        folder = output / f"seed_{seed}"
        study = json.loads((folder / "summary.json").read_text())
        require(study["complete"] and study["source_unchanged"] and study["seed"] == seed and
                study["plan_sha256"] == plan_hash and study["diagonal_exact_replay"] and
                study["fixed_theta_control_exact"], "Incomplete or mismatched seed")
        require(sha256(folder / "responses.npz") == study["responses_sha256"], "Response cache changed")
        with np.load(folder / "responses.npz", allow_pickle=False) as data:
            require(json.loads(str(data["identity"])) == {"plan_sha256": plan_hash, "seed": seed}, "Cache identity differs")
            np.testing.assert_array_equal(data["row_ids"], plan["probe_ids"])
            responses = {key(a, w, t): {f: data[f"{key(a,w,t)}__{f}"] for f in FIELDS}
                         for a in ARMS for w in FACTORS for t in FACTORS}
        verify_replays(responses, baseline_arrays(row, plan))
        require(all_metrics(responses) == study["metrics"], "Factorial metrics differ")
        studies.append(study)
    result = {"complete": True, "source_unchanged": True, "plan_sha256": plan_hash,
              "interpretation": INTERPRETATION, "results": studies}
    checked_plan(output)
    require(sha256(output / "plan.json") == plan_hash, "Plan changed during summary")
    atomic_json(output / "summary.json", result)
    print(f"SUMMARIZED: {output / 'summary.json'}", flush=True)


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
            parser.error("run needs --seed")
        run_seed(args.output, args.seed)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
