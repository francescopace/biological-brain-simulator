"""Measure matched-pair activation and cortical competition without training.

Pulse calibration and image activity do not use image labels or accuracy. The
coupling candidates change only matched exc-to-inh weights and their bounds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, prepare_dataset
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest
from src.neuron import FiringPattern, NeuronType
from src.persistence import load_brain
from src.region import Region, RegionType


def coupling_variant(source, weight):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Coupling must be finite and nonnegative")
    model = copy.deepcopy(source)
    cortex = model.regions["cortex"]
    ns = cortex.n_synapses
    pre, post = cortex.syn_pre[:ns].long(), cortex.syn_post[:ns].long()
    pairs = ((cortex.neuron_type[pre] == NeuronType.EXCITATORY.value)
             & (cortex.neuron_type[post] == NeuronType.INHIBITORY.value) & cortex.syn_alive[:ns])
    exc = torch.where(cortex.neuron_type[:cortex.n_neurons] == NeuronType.EXCITATORY.value)[0]
    inh = torch.where(cortex.neuron_type[:cortex.n_neurons] == NeuronType.INHIBITORY.value)[0]
    count = min(len(exc), len(inh))
    if not torch.equal(pre[pairs], exc[:count]) or not torch.equal(post[pairs], inh[:count]):
        raise ValueError("Coupling intervention requires the benchmark's matched-pair topology")
    cortex.syn_weight[:ns][pairs] = weight
    cortex.syn_max_weight[:ns][pairs] = max(10., weight)
    return model


def paired_impulse(weight, *, resource=1., integration_max_step=.1, steps=30):
    """Force one source spike; measure its paired partner through the real delay/STP path."""
    cortex = Region("pair", RegionType.ASSOCIATION, max_neurons=2,
                    integration_max_step=integration_max_step)
    cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
    mn.wire_cortex_microcircuit(cortex, exc_to_inh_weight=weight)
    cortex.syn_resource[0] = resource
    cortex.v[0] = 35.
    source_steps, partner_steps = [], []
    for step in range(1, steps + 1):
        cortex.step(float(step), step)
        if bool(cortex.fired[0]):
            source_steps.append(step)
        if bool(cortex.fired[1]):
            partner_steps.append(step)
    return {"weight": weight, "initial_resource": resource,
            "integration_max_step": integration_max_step,
            "source_spike_steps": source_steps, "partner_spike_steps": partner_steps,
            "stored_weight": float(cortex.syn_weight[0]),
            "stored_max_weight": float(cortex.syn_max_weight[0])}


def image_activity(source, images, steps):
    expected = simulation_digest(source)
    model = mn._inference_brain(source)
    cortex = model.regions["cortex"]
    n = cortex.n_neurons
    exc = cortex.neuron_type[:n] == NeuronType.EXCITATORY.value
    rows = []
    for image in images:
        mn.reset_inference_state(model)
        before = cortex.total_spikes[:n].clone()
        start = model.time
        mn.present_sample(model, image, steps, learn=False, collect_responses=False)
        counts = cortex.total_spikes[:n] - before
        row = {}
        for name, mask in (("exc", exc), ("inh", ~exc)):
            row[name + "_spikes"] = int(counts[mask].sum())
            row[name + "_active_neurons"] = int((counts[mask] > 0).sum())
            # This is the last spike per cell, not a first-spike latency trace.
            times = cortex.last_spike_time[:n][mask & (counts > 0)] - start
            row[name + "_mean_last_spike_ms"] = float(times.mean()) if times.numel() else None
        rows.append(row)
    assert simulation_digest(source) == expected
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--weights", type=float, nargs="+", default=[8., 24., 32., 64.])
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        parser.error("Use a new output file and a positive sample count")
    study = json.loads((args.study / "summary.json").read_text())
    config = LearningConfig(**study["signature"]["config"])
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_benchmark.py",
               root / "examples/mnist_learning_check.py", root / "examples/mnist_optimization_check.py",
               root / "examples/training_checkpoint.py", *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    if data.manifest != study["dataset"] or args.samples > len(data.train_X):
        raise ValueError("Study dataset or requested sample count mismatch")
    checkpoint = args.study / f"seed_{args.seed}/initial/checkpoints/sample_00000000/brain"
    checkpoint_hash = directory_digest(checkpoint)
    source = load_brain(checkpoint)
    source_hash = simulation_digest(source)
    result = {"complete": False, "source_sha256": hashes, "checkpoint_sha256": checkpoint_hash,
              "sample_ids": data.manifest["train_ids"][:args.samples], "image_steps": config.train_steps,
              "interpretation": "No training, no accuracy measurement and no source mutation. "
                                "Candidates are screened on pulse activation and train-image activity only.",
              "pulse_calibration": [], "image_activity": []}
    for weight in args.weights:
        for resource in (1., .9, .5):
            for cap in (.1, .05):
                result["pulse_calibration"].append(paired_impulse(weight, resource=resource,
                                                                 integration_max_step=cap))
        model = coupling_variant(source, weight)
        row = {"weight": weight, "samples": image_activity(model, data.train_X[:args.samples], config.train_steps)}
        result["image_activity"].append(row)
        atomic_json(args.output, result)
        print("WEIGHT " + str(weight) + " mean exc/inh spikes: " + str([
            float(np.mean([r[key] for r in row["samples"]])) for key in ("exc_spikes", "inh_spikes")]), flush=True)
    assert source_hash == simulation_digest(source)
    assert checkpoint_hash == directory_digest(checkpoint)
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result.update(complete=True, source_unchanged=True)
    atomic_json(args.output, result)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
