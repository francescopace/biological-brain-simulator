"""Bounded CPU/MPS timing on identical MNIST tensors, outside process setup.

Run --prepare under BRAIN_DEVICE=cpu, then independent --trial processes in
CPU/MPS/MPS/CPU order. Only load the locally generated, trusted fixture.
Native noise generators are reseeded, not migrated between backend formats;
training therefore shares images and initial tensors, not random noise draws.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, prepare_dataset
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import array_digest, atomic_json
from src.device import DEVICE


def targets(brain):
    return [brain, brain.encoder, brain.memory, brain.growth, brain.homeostasis,
            brain.stdp, brain.reward_stdp, brain.metaplasticity,
            *brain.regions.values(), *brain.projections]


def tensor_digest(brain):
    digest = hashlib.sha256()
    for index, target in enumerate(targets(brain)):
        for key, value in sorted(vars(target).items()):
            if isinstance(value, torch.Tensor):
                digest.update(str((index, key, value.shape, value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def place_fixture(brain, device):
    """Move this fixed, mechanism-disabled fixture, not general checkpoints."""
    if brain.memory.traces or brain.stats_history or brain.oscillators.enabled:
        raise ValueError("Expected a fresh MNIST timing fixture")
    for target in targets(brain):
        for key, value in list(vars(target).items()):
            if isinstance(value, torch.Tensor):
                setattr(target, key, value.to(device))
            elif isinstance(value, torch.Generator):
                setattr(target, key, torch.Generator(device=device).manual_seed(value.initial_seed()))
    return brain


def synchronize():
    if DEVICE.type == "mps":
        torch.mps.synchronize()


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    paths = [*sorted((root / "src").glob("*.py")),
             *sorted((root / "examples").glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(args):
    if DEVICE.type != "cpu":
        raise ValueError("Prepare the canonical fixture on CPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = LearningConfig(train_per_class=100, exc_to_inh_weight=args.coupling)
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False,
                         parser="liac-arff", data_home=".sklearn_data")
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    # One fixed train-only image per digit in each disjoint group; no accuracy fit.
    train_ids = [int(np.flatnonzero(data.train_y == digit)[0]) for digit in range(10)]
    infer_ids = [int(np.flatnonzero(data.validation_y == digit)[0]) for digit in range(10)]
    train_x, infer_x = data.train_X[train_ids], data.validation_X[infer_ids]
    brain = mn.build_brain(seed=101, exc_to_inh_weight=args.coupling)
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hashes(), "tensor_sha256": tensor_digest(brain),
        "images_sha256": array_digest(train_x, infer_x),
        "train_rows": [data.manifest["train_ids"][i] for i in train_ids],
        "inference_rows": [data.manifest["validation_ids"][i] for i in infer_ids],
        "intensity_target_l1": data.manifest["intensity_target_l1"],
        "neurons": sum(r.n_neurons for r in brain.regions.values()),
        "synapses": sum(t.n_synapses for t in [*brain.regions.values(), *brain.projections]),
        "feedforward_synapses": brain.projections[0].n_synapses,
        "exc_to_inh_weight": args.coupling, "train_noise": brain.encoder.noise_level,
        "dt_ms": brain.dt, "integration": brain.integration_method,
        "integration_max_step_ms": brain.integration_max_step,
        "norm_target": mn.compute_norm_target(brain),
        "scope": "Timing only; no decoder fit or accuracy estimate. Canonical test split unused.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(brain, args.output / "fixture.pt")
    np.savez(args.output / "images.npz", train=train_x, inference=infer_x)
    manifest["fixture_file_sha256"] = hashlib.sha256((args.output / "fixture.pt").read_bytes()).hexdigest()
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}), flush=True)


def train(brain, images, args, norm_target):
    for x in images:
        mn.present_sample(brain, x, args.train_steps, learn=True, collect_responses=False)
        mn.normalize_feedforward_weights(brain, norm_target)
        mn.reset_brain_state(brain, rest_steps=args.rest_steps)


def infer(brain, images, args):
    return [mn.present_inference_sample(brain, x, args.infer_steps) for x in images]


def run_trial(args):
    if DEVICE.type not in ("cpu", "mps"):
        raise ValueError("Only CPU/MPS are included in this comparison")
    if DEVICE.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run with access to Metal, outside the sandbox")
    destination = args.output / (args.trial + ".json")
    if destination.exists() or destination.with_suffix(".npz").exists():
        raise FileExistsError(destination)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert source_hashes() == manifest["source_sha256"], "Sources changed since preparation"
    assert hashlib.sha256((args.output / "fixture.pt").read_bytes()).hexdigest() == manifest["fixture_file_sha256"]
    # This is our own local fixture, not an untrusted/user-supplied pickle.
    source = torch.load(args.output / "fixture.pt", map_location="cpu", weights_only=False)
    source_hash = simulation_digest(source)
    brain = place_fixture(copy.deepcopy(source), DEVICE)
    assert tensor_digest(brain) == manifest["tensor_sha256"], "Initial tensors differ"
    with np.load(args.output / "images.npz") as data:
        assert array_digest(data["train"], data["inference"]) == manifest["images_sha256"]
        train_x = data["train"][:args.train_samples].copy()
        infer_x = data["inference"][:args.infer_samples].copy()
    result = {
        "trial": args.trial, "backend": str(DEVICE), "complete": False,
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "initial_tensor_sha256": tensor_digest(brain),
        "train_samples": len(train_x), "inference_samples": len(infer_x),
        "train_steps": args.train_steps, "rest_steps": args.rest_steps,
        "inference_steps": args.infer_steps,
        "timing_scope": "Production presentation, STDP, normalization and rest; independent inference with CPU response export. Setup, copies, warmup and diagnostics excluded. MPS synchronized before/after each segment.",
        "rng_policy": "Native backend generators, same seeds; noise draws and cross-backend trajectories may differ. Inference noise is zero, starting from the common initial fixture, not separately trained states.",
    }
    synchronize()
    start = time.perf_counter()
    warm = copy.deepcopy(brain)
    train(warm, train_x[:1], args, manifest["norm_target"])
    warm = mn._inference_brain(brain)
    infer(warm, infer_x[:1], args)
    synchronize()
    result["warmup_s"] = time.perf_counter() - start
    del warm
    print("WARMUP " + json.dumps({"trial": args.trial, "seconds": result["warmup_s"]}), flush=True)
    model = copy.deepcopy(brain)
    synchronize()
    start = time.perf_counter()
    train(model, train_x, args, manifest["norm_target"])
    synchronize()
    result["training_s"] = time.perf_counter() - start
    assert model.step_count == len(train_x) * (args.train_steps + args.rest_steps)
    result["training_state_sha256"] = simulation_digest(model)
    result["training_spikes"] = {name: int(r.total_spikes[:r.n_neurons].sum().item())
                                 for name, r in model.regions.items()}
    assert all(bool(torch.isfinite(r.v[:r.n_neurons]).all()) for r in model.regions.values())
    result["training_weights_finite"] = bool(torch.isfinite(model.projections[0].syn_weight).all())
    assert result["training_weights_finite"]
    atomic_json(destination, result)
    print("TRAIN " + json.dumps({"trial": args.trial, "seconds": result["training_s"]}), flush=True)
    model = mn._inference_brain(brain)
    synchronize()
    start = time.perf_counter()
    responses = infer(model, infer_x, args)
    synchronize()
    result["inference_s"] = time.perf_counter() - start
    assert model.step_count == len(infer_x) * args.infer_steps
    spikes = np.stack([r[0] for r in responses])
    voltages = np.stack([r[1] for r in responses])
    assert np.isfinite(voltages).all()
    result["inference_response_sha256"] = array_digest(spikes, voltages)
    result["inference_spikes"] = int(spikes.sum())
    np.savez(destination.with_suffix(".npz"), spikes=spikes, voltages=voltages)
    assert tensor_digest(brain) == manifest["tensor_sha256"]
    assert simulation_digest(source) == source_hash
    assert source_hashes() == manifest["source_sha256"]
    result.update(complete=True, initial_tensors_equal=True, source_unchanged=True)
    atomic_json(destination, result)
    print("COMPLETE " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--trial")
    parser.add_argument("--coupling", type=float, default=64.0)
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--infer-samples", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--rest-steps", type=int, default=25)
    args = parser.parse_args()
    if not (1 <= args.train_samples <= 10 and 1 <= args.infer_samples <= 10
            and args.train_steps > 0 and args.infer_steps > 0 and args.rest_steps >= 0):
        parser.error("Use 1–10 samples, positive presentation steps and nonnegative rest")
    if args.trial and Path(args.trial).name != args.trial:
        parser.error("Trial must be a filename, not a path")
    prepare(args) if args.prepare else run_trial(args)


if __name__ == "__main__":
    main()
