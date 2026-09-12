"""Compare event selection kernels on real spike histories, without training a saved model."""

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
from examples.mnist_diagnosis import ProtocolConfig, load_dataset, patched_benchmark_globals
from examples.mnist_state_diagnosis import checkpoint_digest
from src.persistence import load_brain
from src.synaptic_events import SynapseEventIndex


def torch_scan(fired, indices, alive):
    return torch.where(fired[indices] & alive)[0]


def numpy_scan(fired, indices, alive):
    return torch.from_numpy(np.flatnonzero(fired.numpy()[indices.numpy()] & alive.numpy()))


class CPUAdjacency:
    """Experimental index for fixed endpoint arrays; not a dynamic topology cache."""

    def __init__(self, indices, n_neurons):
        values = indices.numpy()
        self.order = np.argsort(values, kind="stable")
        self.pointers = np.concatenate(([0], np.bincount(values, minlength=n_neurons).cumsum()))
        self.grouped = bool(np.all(values[1:] >= values[:-1]))

    def select(self, fired, indices, alive):
        neurons = np.flatnonzero(fired.numpy())
        if not len(neurons):
            return torch.empty(0, dtype=torch.int64)
        active = np.concatenate([self.order[self.pointers[n]:self.pointers[n + 1]] for n in neurons])
        active = active[alive.numpy()[active]]
        if not self.grouped:
            active.sort()  # Preserve COO order, including floating-point addition order.
        return torch.from_numpy(active)


def compare(indices, alive, frames):
    start = time.perf_counter()
    adjacency = CPUAdjacency(indices, len(frames[0]))
    build_s = time.perf_counter() - start
    checked = SynapseEventIndex()
    kernels = {"torch_scan": torch_scan, "numpy_scan": numpy_scan,
               "adjacency": adjacency.select, "checked_adjacency": checked.select}
    expected = [torch_scan(f, indices, alive) for f in frames]
    for kernel in kernels.values():
        for frame, reference in zip(frames, expected):
            assert torch.equal(kernel(frame, indices, alive), reference)
    times = {name: [] for name in kernels}
    names = list(kernels)
    for order in (names, names[::-1], names):
        for name in order:
            start = time.perf_counter()
            for frame in frames:
                kernels[name](frame, indices, alive)
            times[name].append(time.perf_counter() - start)
    return {
        "synapses": len(indices), "frames": len(frames), "adjacency_build_s": build_s,
        "mean_active_synapses": float(np.mean([len(e) for e in expected])),
        "all_indices_and_order_equal": True,
        "seconds": times,
        "median_seconds": {name: float(np.median(values)) for name, values in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    if min(args.samples, args.steps) < 1:
        parser.error("samples and steps must be positive")
    if mn.DEVICE.type != "cpu":
        parser.error("This comparison requires BRAIN_DEVICE=cpu")
    source_hash = checkpoint_digest(args.checkpoint)
    config = ProtocolConfig(**json.loads(args.reference.read_text())["config"])
    brain = copy.deepcopy(load_brain(args.checkpoint))
    frames = {name: [] for name in brain.regions}
    with patched_benchmark_globals(config):
        X, _, _, _ = load_dataset(config)
        for x in X[:args.samples]:
            for _ in range(args.steps):
                brain.stimulate("input", x)
                brain.step()
                mn.apply_feedforward_stdp(brain)
                for name, region in brain.regions.items():
                    frames[name].append(region.fired[:region.n_neurons].clone())
            mn.reset_brain_state(brain)
    cortex = brain.regions["cortex"]
    projection = brain.get_projection("input", "cortex")
    payload = {
        "checkpoint_sha256": source_hash, "threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "index_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent.parent / "src/synaptic_events.py").read_bytes()
        ).hexdigest(),
        "interpretation": "Microkernel timings; full MNIST is running concurrently. No model selection.",
        "cases": {},
    }
    for label, target, attr, region in (
        ("cortex_propagation", cortex, "syn_pre", "cortex"),
        ("projection_propagation_and_ltd", projection, "syn_pre", "input"),
        ("projection_ltp", projection, "syn_post", "cortex"),
    ):
        ns = target.n_synapses
        print(f"START {label}: {ns} synapses", flush=True)
        payload["cases"][label] = compare(getattr(target, attr)[:ns], target.syn_alive[:ns], frames[region])
        print(json.dumps(payload["cases"][label]), flush=True)
    assert checkpoint_digest(args.checkpoint) == source_hash
    payload["checkpoint_unchanged"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
