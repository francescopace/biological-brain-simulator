"""Compare the CPU Heun loop against the identical PyTorch algorithm.

Training uses new compact excitatory-only MNIST models in ABBA order.
Inference compares complete responses of all labelled readout/validation rows.
No canonical test rows are used, and neither arm changes the integration rule.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, cached_responses, prepare_dataset, probe_decoders
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json
import src.integration as integration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--training-samples", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.training_samples < 1:
        parser.error("Use a new output directory and a positive training sample count")
    study = json.loads((args.study / "summary.json").read_text())
    config = LearningConfig(**study["signature"]["config"])
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_benchmark.py", root / "examples/mnist_learning_check.py",
               root / "examples/mnist_optimization_check.py", root / "examples/training_checkpoint.py",
               *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    assert data.manifest == study["dataset"]
    brain = mn.build_brain(seed=args.seed)
    assert all(r.v.device.type == "cpu" for r in brain.regions.values())
    initial_hash = simulation_digest(brain)
    args.output.mkdir(parents=True)
    payload = {"complete": False, "seed": args.seed, "source_sha256": hashes,
               "config": asdict(config), "dataset": data.manifest,
               "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
               "interpretation": "CPU kernel equivalence, not an integrator or accuracy comparison. "
               "Timings share host load; training uses ABBA order, inference uses one run per backend.",
               "training": [], "inference": []}
    for enabled in (False, True, True, False):
        model = copy.deepcopy(brain)
        start = time.perf_counter()
        with patch.object(integration, "CPU_HEUN_ENABLED", enabled), patch.object(mn, "REST_STEPS", config.rest_steps):
            mn.train_unsupervised(model, data.train_X[:args.training_samples], epochs=1,
                                  train_present_steps=config.train_steps, seed=args.seed, log_every=0)
        row = {"cpu_kernel": enabled, "wall_s": time.perf_counter() - start,
               "samples": min(len(data.train_X), args.training_samples),
               "simulation_sha256": simulation_digest(model)}
        payload["training"].append(row)
        atomic_json(args.output / "summary.json", payload)
        print("TRAIN " + json.dumps(row), flush=True)
    assert len({r["simulation_sha256"] for r in payload["training"]}) == 1
    outputs = []
    for enabled in (False, True):
        start = time.perf_counter()
        with patch.object(integration, "CPU_HEUN_ENABLED", enabled):
            readout, validation = cached_responses(brain, data, config, args.output / f"responses_{enabled}.npz")
        row = {"cpu_kernel": enabled, "wall_s": time.perf_counter() - start,
               "decoders": probe_decoders(readout, data.train_y[data.readout_indices], validation,
                                          data.validation_y, config, mn.CLASSES)}
        outputs.append((readout, validation))
        payload["inference"].append(row)
        atomic_json(args.output / "summary.json", payload)
        print("INFERENCE " + json.dumps({"cpu_kernel": enabled, "wall_s": row["wall_s"]}), flush=True)
    for left, right in zip(*outputs):
        for attr in ("exc_indices", "spikes", "voltages"):
            np.testing.assert_array_equal(getattr(left, attr), getattr(right, attr))
    assert payload["inference"][0]["decoders"] == payload["inference"][1]["decoders"]
    assert simulation_digest(brain) == initial_hash
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    times = {enabled: statistics.median(r["wall_s"] for r in payload["training"] if r["cpu_kernel"] == enabled)
             for enabled in (False, True)}
    payload.update(complete=True, training_speedup=times[False] / times[True],
                   inference_speedup=payload["inference"][0]["wall_s"] / payload["inference"][1]["wall_s"],
                   training_state_and_rng_equal=True, inference_responses_and_predictions_equal=True)
    atomic_json(args.output / "summary.json", payload)
    print("COMPLETE " + json.dumps({k: payload[k] for k in ("training_speedup", "inference_speedup")}), flush=True)


if __name__ == "__main__":
    main()
