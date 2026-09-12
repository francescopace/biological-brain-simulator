"""Post-hoc state/readout diagnostics on immutable completed MNIST caches.

The balanced label-deletion check refits only decoders on 90 of the 100 existing
labelled images. Its ranges describe these perturbations, not confidence intervals.
Old/new cohorts change seeds and datasets together and cannot isolate those causes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples import mnist_benchmark as mn
from examples.mnist_learning_check import ridge_predictions
from examples.mnist_optimization_check import simulation_digest
from examples.mnist_readout_features import readout_features
from examples.mnist_readout_transfer_check import features
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


def balanced_deletions(labels, repeats=100, seed=20260916):
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    if not groups or any(len(g) != 10 for g in groups):
        raise ValueError("Expected ten labelled examples per class")
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(g, size=9, replace=False) for g in groups]) for _ in range(repeats)]


def response(cache, prefix):
    return mn.ReadoutResponses(cache["exc_indices"], cache[prefix + "_spikes"], cache[prefix + "_voltages"])


def activity(cache):
    train, valid = cache["readout_spikes"], cache["validation_spikes"]
    unseen = train.sum(axis=0) == 0
    return {"mean_validation_spikes": float(valid.sum(axis=1).mean()),
            "mean_active_neurons_per_image": float((valid > 0).sum(axis=1).mean()),
            "neurons_active_in_readout": int((~unseen).sum()),
            "constant_spike_features": int((train.var(axis=0) == 0).sum()),
            "unseen_neuron_fraction_of_validation_spikes": float(valid[:, unseen].sum() / valid.sum()),
            "silent_validation_images": int((valid.sum(axis=1) == 0).sum())}


def summarize_cohort(rows):
    result = {}
    for decoder in ("ridge_spikes", "ridge_spikes_centered_voltage"):
        matrices = np.array([r["cross_readouts"][decoder]["accuracy_matrix"] for r in rows])
        gains = np.array([r["label_deletion"][decoder]["gain_pp"] for r in rows])
        averaged = gains.mean(axis=0)
        result[decoder] = {"mean_accuracy_matrix": matrices.mean(axis=0).tolist(),
            "fixed_control_decoder_gain_pp": float(((matrices[:, 0, 1] - matrices[:, 0, 0]) * 100).mean()),
            "subsequent_refit_component_pp": float(((matrices[:, 1, 1] - matrices[:, 0, 1]) * 100).mean()),
            "diagonal_gain_pp": float(((matrices[:, 1, 1] - matrices[:, 0, 0]) * 100).mean()),
            "deletion_mean_gain_pp": float(averaged.mean()),
            "deletion_gain_5_95_percentiles_pp": np.quantile(averaged, [.05, .95]).tolist(),
            "deletion_fraction_positive": float((averaged > 0).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    sources = [("old", Path("results/20260912-learning-1000-wta64")),
               ("old", Path("results/20260912-learning-1000-wta64-replication"))]
    sources += [("new", Path(f"results/20260912-wta64-fresh-replication/seed_{s}")) for s in (201, 202, 203)]
    file_hashes, checkpoint_hashes, results = {}, {}, []
    def read_json(path):
        content = path.read_bytes()
        file_hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)
    for group, path in sources:
        study = read_json(path / "summary.json")
        assert study["complete"]
        for seed in study["signature"]["seeds"]:
            print(f"ANALYZE {group} seed={seed}", flush=True)
            rows = {r["condition"]: r for r in study["results"] if r["seed"] == seed}
            caches, models, updates = {}, {}, {}
            for condition in ("normalization_only", "stdp_normalized"):
                folder = path / f"seed_{seed}" / condition
                cache_path = folder / "responses.npz"
                file_hashes[str(cache_path)] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                with np.load(cache_path, allow_pickle=False) as cache:
                    assert json.loads(str(cache["identity"]))["brain_sha256"] == rows[condition]["final_sha256"]
                    caches[condition] = {k: cache[k].copy() for k in cache.files if k != "identity"}
                checkpoint = folder / "checkpoints" / "sample_00001000"
                checkpoint_hashes[str(checkpoint)] = directory_digest(checkpoint)
                metadata = read_json(checkpoint / "progress.json")
                model, progress = load_training_checkpoint(checkpoint, metadata["progress"]["protocol"])
                assert simulation_digest(model) == rows[condition]["final_sha256"]
                models[condition], updates[condition] = model, progress["weight_deltas"]
            c, t = caches["normalization_only"], caches["stdp_normalized"]
            assert np.array_equal(c["readout_y"], t["readout_y"]) and np.array_equal(c["validation_y"], t["validation_y"])
            assert np.array_equal(c["exc_indices"], t["exc_indices"])
            subsets = balanced_deletions(c["readout_y"])
            result = {"cohort": group, "seed": seed, "cross_readouts": {}, "label_deletion": {},
                      "activity": {k: activity(v) for k, v in caches.items()},
                      "pixel_ridge_accuracy": study["pixel_ridge_baseline"]["accuracy"]}
            for decoder in ("ridge_spikes", "ridge_spikes_centered_voltage"):
                train = [features(response(z, "readout"), decoder) for z in (c, t)]
                valid = [features(response(z, "validation"), decoder) for z in (c, t)]
                matrix = np.empty((2, 2))
                for i, condition in enumerate(("normalization_only", "stdp_normalized")):
                    _, classifier = ridge_predictions(train[i], c["readout_y"], valid[i], 1.)
                    for j in range(2):
                        predictions = classifier.predict(valid[j])
                        matrix[i, j] = float((predictions == c["validation_y"]).mean())
                        if i == j and decoder in rows[condition]["decoders"]:
                            assert predictions.tolist() == rows[condition]["decoders"][decoder]["predictions"]
                result["cross_readouts"][decoder] = {"rows_fit_columns_evaluate": ["normalization_only", "stdp_normalized"],
                                                      "accuracy_matrix": matrix.tolist()}
                gains = []
                for indices in subsets:
                    accuracy = []
                    for i in range(2):
                        predictions, _ = ridge_predictions(train[i][indices], c["readout_y"][indices], valid[i], 1.)
                        accuracy.append(float((predictions == c["validation_y"]).mean()))
                    gains.append(100 * (accuracy[1] - accuracy[0]))
                result["label_deletion"][decoder] = {"gain_pp": gains,
                    "gain_5_95_percentiles_pp": np.quantile(gains, [.05, .95]).tolist(),
                    "fraction_positive": float((np.asarray(gains) > 0).mean())}
            left, right = [models[k].get_projection("input", "cortex") for k in ("normalization_only", "stdp_normalized")]
            assert left.n_synapses == right.n_synapses
            ns = left.n_synapses
            for attr in ("syn_pre", "syn_post", "syn_alive", "syn_delay"):
                assert np.array_equal(getattr(left, attr)[:ns].numpy(), getattr(right, attr)[:ns].numpy())
            weights = left.syn_weight[:ns].numpy().astype(np.float64)
            after = right.syn_weight[:ns].numpy().astype(np.float64)
            delta = after - weights
            theta = {name: (models["stdp_normalized"].regions[name].theta - r.theta).numpy()
                     for name, r in models["normalization_only"].regions.items()}
            trace = updates["stdp_normalized"]
            nonzero = [r for r in trace[1:] if r["raw_normalization_cosine"] is not None]
            result["state"] = {"relative_weight_l1_vs_control_pct": float(np.abs(delta).sum() / np.abs(weights).sum() * 100),
                "changed_weights": int(np.count_nonzero(delta)), "live_synapses": ns,
                "weights_at_min": int((after == right.syn_min_weight[:ns].numpy()).sum()),
                "weights_at_max": int((after == right.syn_max_weight[:ns].numpy()).sum()),
                "theta_abs_mean_difference": {k: float(np.abs(v).mean()) for k, v in theta.items()},
                "theta_abs_max_difference": {k: float(np.abs(v).max()) for k, v in theta.items()},
                "median_raw_normalization_cosine_excluding_first_image": float(np.median([r["raw_normalization_cosine"] for r in nonzero])),
                "median_net_to_raw_l1_excluding_first_image": float(np.median([r["total_to_raw_l1"] for r in nonzero]))}
            result["changed_responses"] = {prefix: {
                "spike_cells_pct": float(np.mean(c[prefix + "_spikes"] != t[prefix + "_spikes"]) * 100),
                "images_with_spike_changes_pct": float(np.mean(np.any(c[prefix + "_spikes"] != t[prefix + "_spikes"], axis=1)) * 100)}
                for prefix in ("readout", "validation")}
            for condition, model in models.items():
                assert simulation_digest(model) == rows[condition]["final_sha256"]
            results.append(result)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in file_hashes.items())
    assert all(directory_digest(p) == h for p, h in checkpoint_hashes.items())
    payload = {"complete": True, "results": results, "summary": {g: summarize_cohort([r for r in results if r["cohort"] == g]) for g in ("old", "new")},
        "input_sha256": file_hashes, "checkpoint_sha256": checkpoint_hashes,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources_unchanged": True,
        "interpretation": "Post-hoc cached diagnostics. The fixed-decoder/refit components telescope along one chosen intervention order, not a unique causal attribution. Label-deletion fits use nine existing examples per digit, same subsets for paired models and shared per cohort; percentiles are perturbation ranges, not population confidence intervals. No SNN simulation/training, validation-based parameter selection, canonical test scoring or source mutation."}
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
