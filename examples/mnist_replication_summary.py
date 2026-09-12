"""Verify a preregistered fresh-split study and summarize paired seed results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples import mnist_benchmark as mn
from examples.mnist_learning_check import CONDITIONS, DECODERS, LearningConfig, probe_decoders
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


def paired_effect(differences, labels, *, seed=20260915, resamples=5000):
    """Bootstrap shared images, not seed-image pairs as independent samples."""
    differences = np.asarray(differences, dtype=np.float64)
    labels = np.asarray(labels)
    if (differences.ndim != 2 or not differences.shape[0] or not labels.size
            or differences.shape[1] != labels.size or resamples < 1
            or not np.isfinite(differences).all()):
        raise ValueError("Expected finite seed-by-image differences and aligned labels")
    per_image = differences.mean(axis=0)
    rng = np.random.default_rng(seed)
    means = np.zeros(resamples)
    for digit in np.unique(labels):
        ids = np.flatnonzero(labels == digit)
        sampled = rng.choice(ids, size=(resamples, len(ids)), replace=True)
        means += per_image[sampled].sum(axis=1) / len(labels)
    interval = np.quantile(means * 100, [.025, .975]).tolist()
    gains = (differences.mean(axis=1) * 100).tolist()
    return {
        "seed_gains_pp": gains, "mean_gain_pp": float(np.mean(gains)),
        "conditional_image_bootstrap_95_pp": interval,
        "wins_per_seed": (differences > 0).sum(axis=1).tolist(),
        "losses_per_seed": (differences < 0).sum(axis=1).tolist(),
        "positive_each_seed": bool(all(gain > 0 for gain in gains)),
        "registered_positive_criterion_met": bool(all(gain > 0 for gain in gains) and interval[0] > 0),
    }


def summarize(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    excluded_bytes = (root / "excluded_rows.json").read_bytes()
    assert hashlib.sha256(excluded_bytes).hexdigest() == plan["exclusions"]["sha256"]
    excluded = set(json.loads(excluded_bytes))
    for previous in plan["exclusions"]["sources"]:
        assert hashlib.sha256(Path(previous["path"]).read_bytes()).hexdigest() == previous["sha256"]
    seeds, config = plan["seeds"], plan["config"]
    repo = Path(__file__).resolve().parent.parent
    rows, signatures, datasets, labels, checkpoint_checks = {}, [], [], [], []
    for seed in seeds:
        study_root = root / f"seed_{seed}"
        study = json.loads((study_root / "summary.json").read_text())
        assert study["complete"] and study["source_unchanged"]
        assert study["started_at"] >= plan["created_at"]
        signature = study["signature"]
        assert signature["seeds"] == [seed] and signature["config"] == config
        assert set(signature["excluded_ids"]) == excluded
        signatures.append({k: v for k, v in signature.items() if k != "seeds"})
        dataset = study["dataset"]
        datasets.append(dataset)
        train, valid = set(dataset["train_ids"]), set(dataset["validation_ids"])
        assert len(train) == config["train_per_class"] * 10
        assert len(valid) == config["validation_per_class"] * 10
        assert not train & valid and not (train | valid) & excluded
        assert min(train | valid) >= 0 and max(train | valid) < 60000
        assert set(dataset["readout_ids"]) <= train
        assert len(set(dataset["readout_ids"])) == config["readout_per_class"] * 10
        assert set(dataset["excluded_ids"]) == excluded
        for name, digest in signature["source_sha256"].items():
            for path in (repo / name, study_root / "source_snapshot" / name):
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert len(study["results"]) == 3
        assert {r["condition"] for r in study["results"]} == set(CONDITIONS)
        assert len({r["initial_sha256"] for r in study["results"]}) == 1
        for row in study["results"]:
            condition = row["condition"]
            assert row["seed"] == seed
            count = 0 if condition == "initial" else len(train) * config["epochs"]
            assert row["completed_samples"] == count
            folder = study_root / f"seed_{seed}" / condition
            checkpoint = folder / "checkpoints" / f"sample_{count:08d}"
            metadata = json.loads((checkpoint / "progress.json").read_text())
            protocol = metadata["progress"]["protocol"]
            assert protocol == {"config": config, "seed": seed, "condition": condition,
                                "train_sha256": dataset["train_sha256"],
                                "source_sha256": signature["source_sha256"]}
            before = directory_digest(checkpoint)
            brain, progress = load_training_checkpoint(checkpoint, protocol)
            assert brain.step_count == count * (config["train_steps"] + config["rest_steps"])
            assert simulation_digest(brain) == row["final_sha256"]
            assert all(r.v.device.type == "cpu" for r in brain.regions.values())
            if condition == "initial":
                assert row["initial_sha256"] == row["final_sha256"]
            else:
                assert progress["initial_sha256"] == row["initial_sha256"]
                updates = progress["weight_deltas"]
                assert len(updates) == count
                assert updates == json.loads((folder / "weight_deltas.json").read_text())
                if condition == "normalization_only":
                    assert all(u["raw_stdp_l1"] == 0 for u in updates)
                else:
                    assert any(u["raw_stdp_l1"] > 0 for u in updates)
            assert directory_digest(checkpoint) == before
            del brain
            with np.load(folder / "responses.npz", allow_pickle=False) as cache:
                identity = json.loads(str(cache["identity"]))
                assert identity == {"brain_sha256": row["final_sha256"],
                                    "inference_steps": config["inference_steps"],
                                    "readout_sha256": dataset["readout_sha256"],
                                    "validation_sha256": dataset["validation_sha256"]}
                y = cache["validation_y"].copy()
                labels.append(y)
                assert np.array_equal(np.bincount(y), np.full(10, config["validation_per_class"]))
                assert np.array_equal(np.bincount(cache["readout_y"]), np.full(10, config["readout_per_class"]))
                assert np.isfinite(cache["validation_voltages"]).all()
                assert row["validation_silent_samples"] == int((cache["validation_spikes"].sum(axis=1) == 0).sum())
                readout = mn.ReadoutResponses(cache["exc_indices"], cache["readout_spikes"], cache["readout_voltages"])
                validation = mn.ReadoutResponses(cache["exc_indices"], cache["validation_spikes"], cache["validation_voltages"])
                assert probe_decoders(readout, cache["readout_y"], validation, y,
                                      LearningConfig(**config), tuple(range(10))) == row["decoders"]
            for decoder in DECODERS:
                score = row["decoders"][decoder]
                assert len(score["predictions"]) == len(y)
                assert float(np.mean(np.asarray(score["predictions"]) == y)) == score["accuracy"]
            rows[seed, condition] = row
            checkpoint_checks.append({"seed": seed, "condition": condition, "sha256": before})
    assert all(s == signatures[0] for s in signatures)
    assert all(d == datasets[0] for d in datasets)
    assert all(np.array_equal(y, labels[0]) for y in labels)
    y = labels[0]
    means = {c: {d: float(np.mean([rows[s, c]["decoders"][d]["accuracy"] for s in seeds]))
                 for d in DECODERS} for c in CONDITIONS}
    effects = {}
    for control in ("normalization_only", "initial"):
        effects[control] = {}
        for decoder in DECODERS:
            differences = [
                (np.asarray(rows[s, "stdp_normalized"]["decoders"][decoder]["predictions"]) == y).astype(int)
                - (np.asarray(rows[s, control]["decoders"][decoder]["predictions"]) == y).astype(int)
                for s in seeds]
            effects[control][decoder] = paired_effect(differences, y, seed=plan["uncertainty"]["seed"],
                                                     resamples=5000)
    return {"complete": True, "seeds": seeds, "means": means, "paired_effects": effects,
            "primary_criterion_met": effects["normalization_only"]["ridge_spikes"]["registered_positive_criterion_met"],
            "initialization_criterion_met": effects["initial"]["ridge_spikes"]["registered_positive_criterion_met"],
            "plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
            "dataset": datasets[0], "checkpoint_integrity": checkpoint_checks,
            "silent_images": {str(s): {c: rows[s, c]["validation_silent_samples"] for c in CONDITIONS} for s in seeds},
            "source_and_checkpoints_unchanged": True,
            "interpretation": f"Primary contrast: ridge_spikes STDP minus normalization. Other decoders are secondary, not alternative primary endpoints. Intervals condition on these {len(seeds)} fixed networks and one split; {len(y)} shared images are not {len(y) * len(seeds)} independent observations. No canonical test evaluation or default change."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    result = summarize(args.study)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("means", "primary_criterion_met", "initialization_criterion_met")}), flush=True)


if __name__ == "__main__":
    main()
