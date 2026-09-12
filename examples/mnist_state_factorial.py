"""Separate saved feedforward weights from adaptive thresholds without training.

Cross the three completed learning-control states on their original train-only
validation split. Diagonal cases must reproduce the saved responses exactly.
These interventions diagnose the saved states, not alternative training rules.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import (
    CONDITIONS, LearningConfig, cached_responses, prepare_dataset, probe_decoders,
)
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


def factorial_variant(initial, weight_source, theta_source):
    """Copy one projection's weights and all regional theta, never the donors."""
    for donor in (weight_source, theta_source):
        if (initial.dt != donor.dt or list(initial.regions) != list(donor.regions)
                or len(initial.projections) != len(donor.projections)):
            raise ValueError("Factorial donors must share the same network")
        for name, left in initial.regions.items():
            right = donor.regions[name]
            if (left.n_neurons != right.n_neurons or left.dt != right.dt
                    or left.integration_method != right.integration_method
                    or left.integration_max_step != right.integration_max_step):
                raise ValueError("Factorial donors must share neuron dynamics")
            for attr in ("a", "b", "c", "d", "neuron_type", "neuron_alive"):
                if not torch.equal(getattr(left, attr), getattr(right, attr)):
                    raise ValueError(f"Factorial donors differ in {name}.{attr}")
        for left, right in zip([*initial.regions.values(), *initial.projections],
                               [*donor.regions.values(), *donor.projections]):
            for name in ("source_name", "target_name"):
                if getattr(left, name, None) != getattr(right, name, None):
                    raise ValueError("Factorial donors differ in projection identity")
            if left.n_synapses != right.n_synapses:
                raise ValueError("Factorial donors differ in synapse count")
            for attr in ("syn_pre", "syn_post", "syn_alive", "syn_delay", "syn_modulation",
                         "syn_attenuation", "syn_min_weight", "syn_max_weight"):
                if not torch.equal(getattr(left, attr), getattr(right, attr)):
                    raise ValueError(f"Factorial donors differ in {attr}")
            if left is not initial.get_projection("input", "cortex"):
                if not torch.equal(left.syn_weight, right.syn_weight):
                    raise ValueError("Only input-to-cortex weights may differ")
    model = copy.deepcopy(initial)
    target = model.get_projection("input", "cortex")
    source = weight_source.get_projection("input", "cortex")
    target.syn_weight[:target.n_synapses].copy_(source.syn_weight[:source.n_synapses])
    for name, region in model.regions.items():
        region.theta.copy_(theta_source.regions[name].theta)
    return model


def response_summary(responses):
    spikes = responses.spikes
    return {
        "silent_samples": int(np.sum(spikes.sum(axis=1) == 0)),
        "mean_spikes_per_sample": float(spikes.sum(axis=1).mean()),
        "mean_active_neurons_per_sample": float((spikes > 0).sum(axis=1).mean()),
        "active_neurons_across_samples": int(np.sum(spikes.sum(axis=0) > 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    study = json.loads((args.study / "summary.json").read_text())
    if not study["complete"]:
        parser.error("The source study must be complete")
    rows = {r["condition"]: r for r in study["results"] if r["seed"] == args.seed}
    if set(rows) != set(CONDITIONS) or len({r["initial_sha256"] for r in rows.values()}) != 1:
        parser.error("All three controls must share the same initialization")
    config = LearningConfig(**study["signature"]["config"])
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_learning_check.py",
               root / "examples/mnist_benchmark.py", root / "examples/training_checkpoint.py",
               root / "examples/mnist_optimization_check.py", *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    if data.manifest != study["dataset"]:
        raise ValueError("Source study dataset mismatch")
    models, directories, digests = {}, {}, {}
    for condition, row in rows.items():
        directory = args.study / f"seed_{args.seed}" / condition / "checkpoints" / f"sample_{row['completed_samples']:08d}"
        protocol = json.loads((directory / "progress.json").read_text())["progress"]["protocol"]
        models[condition], _ = load_training_checkpoint(directory, protocol)
        if simulation_digest(models[condition]) != row["final_sha256"]:
            raise ValueError("Saved state does not match the completed study")
        directories[condition], digests[condition] = directory, directory_digest(directory)
    original_states = {c: simulation_digest(b) for c, b in models.items()}
    payload = {
        "complete": False, "seed": args.seed, "study": str(args.study.resolve()),
        "dataset": data.manifest, "source_sha256": hashes, "checkpoint_sha256": digests,
        "readout_samples": len(data.readout_indices), "validation_samples": len(data.validation_y),
        "interpretation": "Evaluation-only interventions on existing states. Same labelled readout "
                          "and validation rows in every case; no alternative training run "
                          "or canonical test claim. Every diagonal must reproduce cached responses.",
        "results": [], "theta": {},
    }
    for condition, model in models.items():
        payload["theta"][condition] = {name: {"mean": float(r.theta[:r.n_neurons].mean()),
                                             "max": float(r.theta[:r.n_neurons].max())}
                                          for name, r in model.regions.items()}
    # Reproduce the diagonal first, so omitted state cannot masquerade as a factor effect.
    cases = [(c, c) for c in CONDITIONS]
    cases += [(w, t) for w in CONDITIONS for t in CONDITIONS if w != t]
    args.output.mkdir(parents=True)
    for weights, theta in cases:
        print(f"START weights={weights} theta={theta}", flush=True)
        start = time.perf_counter()
        model = factorial_variant(models["initial"], models[weights], models[theta])
        readout, validation = cached_responses(model, data, config, args.output / f"{weights}--{theta}.npz")
        if weights == theta:
            saved = args.study / f"seed_{args.seed}" / weights / "responses.npz"
            with np.load(saved, allow_pickle=False) as old:
                for prefix, response in (("readout", readout), ("validation", validation)):
                    for attr in ("spikes", "voltages"):
                        if old[f"{prefix}_{attr}"].tobytes() != getattr(response, attr).tobytes():
                            raise AssertionError(f"Diagonal failed for {weights}: {prefix}.{attr}")
                np.testing.assert_array_equal(old["exc_indices"], readout.exc_indices)
        row = {"weights": weights, "theta": theta, "wall_s": time.perf_counter() - start,
               "validation": response_summary(validation),
               "decoders": probe_decoders(readout, data.train_y[data.readout_indices], validation,
                                           data.validation_y, config, mn.CLASSES)}
        payload["results"].append(row)
        atomic_json(args.output / "summary.json", payload)
        print("END " + json.dumps({d: v["accuracy"] for d, v in row["decoders"].items()}), flush=True)
    assert original_states == {c: simulation_digest(b) for c, b in models.items()}
    assert digests == {c: directory_digest(p) for c, p in directories.items()}
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    payload.update(complete=True, source_unchanged=True, diagonals_bitwise_equal=True)
    atomic_json(args.output / "summary.json", payload)
    print("COMPLETE: source states unchanged, all three diagonals exact", flush=True)


if __name__ == "__main__":
    main()
