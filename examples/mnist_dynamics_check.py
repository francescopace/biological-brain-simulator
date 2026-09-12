"""Paired 2x2 inference ablation: integration method versus input-to-INH wiring.

Load one saved model; never retrain it or regenerate its excitatory weights.
Disabling existing inhibitory-target edges isolates wiring from RNG/topology
generation. Therefore these timings do not measure the smaller new builder.
All decoder fitting and validation use the prior train-only study's rows.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, cached_responses, prepare_dataset, probe_decoders
from examples.mnist_numerics_check import probe_images
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest
from src.neuron import NeuronType
from src.persistence import load_brain


def variant(source, method, exc_only, max_step):
    model = copy.deepcopy(source)
    model.integration_method = method
    model.integration_max_step = max_step
    for region in model.regions.values():
        region.integration_method = method
        region.integration_max_step = max_step
    proj = model.get_projection("input", "cortex")
    n = proj.n_synapses
    inh = model.regions["cortex"].neuron_type[proj.syn_post[:n].long()] == NeuronType.INHIBITORY.value
    if exc_only:
        proj.syn_alive[:n][inh] = False
    return model


def run_case(source, data, config, folder, *, method, exc_only, max_step, probe_samples=8):
    model = variant(source, method, exc_only, max_step)
    before = simulation_digest(model)
    start = time.perf_counter()
    readout, validation = cached_responses(model, data, config, folder / "responses.npz")
    response_s = time.perf_counter() - start
    proj = model.get_projection("input", "cortex")
    result = {
        "method": method, "exc_only": exc_only, "max_step": max_step,
        "model_sha256": before, "live_feedforward": int(proj.syn_alive[:proj.n_synapses].sum()),
        "response_wall_s": response_s,
        "decoders": probe_decoders(readout, data.train_y[data.readout_indices], validation,
                                   data.validation_y, config, mn.CLASSES),
        "mean_exc_spikes": float(validation.spikes.sum(axis=1).mean()),
        "silent_validation_images": int((validation.spikes.sum(axis=1) == 0).sum()),
    }
    if probe_samples:
        result["numerical_probes"] = probe_images(model, data.validation_X[:probe_samples], config.inference_steps)
    assert simulation_digest(model) == before, "Inference changed the source model"
    atomic_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--check-refinement", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if not np.isfinite(args.max_step) or args.max_step <= 0:
        parser.error("max-step must be finite and positive")
    study = json.loads((args.study / "summary.json").read_text())
    config = LearningConfig(**study["signature"]["config"])
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_benchmark.py", root / "examples/mnist_learning_check.py",
               root / "examples/mnist_numerics_check.py", root / "examples/mnist_optimization_check.py",
               root / "examples/training_checkpoint.py", *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_hash = directory_digest(args.checkpoint)
    source = load_brain(args.checkpoint)
    original = simulation_digest(source)
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    if data.manifest != study["dataset"]:
        raise ValueError("Study split or preprocessing does not match")
    args.output.mkdir(parents=True)
    payload = {"pid": os.getpid(), "complete": False, "started_at": time.time(),
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_hash,
               "config": asdict(config), "dataset": data.manifest, "source_sha256": hashes,
               "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
               "interpretation": "Inference-only intervention on one checkpoint, not retraining or a held-out test. "
               "Disabled edges retain storage; timings are not compact-topology speedups. "
               "Runs share host load with the existing full benchmark.", "results": []}
    atomic_json(args.output / "summary.json", payload)
    cases = [("legacy_euler", False, args.max_step), ("heun", False, args.max_step),
             ("legacy_euler", True, args.max_step), ("heun", True, args.max_step)]
    if args.check_refinement:
        cases.append(("heun", True, args.max_step / 2))
    for i, (method, exc_only, max_step) in enumerate(cases):
        print(f"START {i+1}/{len(cases)} method={method} exc_only={exc_only} h={max_step}", flush=True)
        result = run_case(source, data, config, args.output / f"case_{i}",
                          method=method, exc_only=exc_only, max_step=max_step)
        payload["results"].append(result)
        atomic_json(args.output / "summary.json", payload)
        print("END " + json.dumps({"method": method, "exc_only": exc_only, "h": max_step,
              "wall_s": result["response_wall_s"], "mean_spikes": result["mean_exc_spikes"],
              "accuracy": {k: v["accuracy"] for k, v in result["decoders"].items()}}), flush=True)
    assert simulation_digest(source) == original
    assert directory_digest(args.checkpoint) == source_hash
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    payload.update(complete=True, finished_at=time.time(), checkpoint_unchanged=True)
    atomic_json(args.output / "summary.json", payload)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
