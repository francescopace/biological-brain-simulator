"""Check inhibitory integration instability on scalar probes and saved-network states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.training_checkpoint import atomic_json, directory_digest
from src.neuron import FiringPattern, NeuronType
from src.persistence import load_brain
from src.region import MAX_DELAY_STEPS, Region, RegionType


def fine_step(v, u, current, a, b, *, duration=1.0, substep=0.02):
    """Resolve first threshold crossings with a finer Euler reference."""
    v, u = v.clone(), u.clone()
    fired = torch.zeros_like(v, dtype=torch.bool)
    steps = int(np.ceil(duration / substep))
    h = duration / steps
    for _ in range(steps):
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + current) * h
        du = a * (b * v - u) * h
        v = torch.where(fired, v, v + dv)
        u = torch.where(fired, u, u + du)
        fired |= v >= 30.0
        v = torch.clamp(v, max=30.0)
    return fired, v, u


def scalar_probes():
    rows = []
    for current in (-50.0, -100.0, -200.0, -300.0, -1000.0):
        for method, dt in (("legacy_euler", 1.0), ("legacy_euler", 0.05), ("heun", 1.0)):
            region = Region("probe", RegionType.ASSOCIATION, max_neurons=1, dt=dt,
                            integration_method=method)
            region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
            minimum = -65.0
            for step in range(round(100 / dt)):
                region.current[0] = current
                region.step((step + 1) * dt, step + 1)
                minimum = min(minimum, float(region.v[0]))
            rows.append({"current": current, "dt": dt, "method": method,
                         "spikes_in_100ms": int(region.total_spikes[0]),
                         "minimum_v": minimum, "final_v": float(region.v[0])})
    return rows


def probe_images(brain, images, n_steps):
    model = mn._inference_brain(brain)
    cortex = model.regions["cortex"]
    n = cortex.n_neurons
    exc = (cortex.neuron_type[:n] == NeuronType.EXCITATORY.value) & cortex.neuron_alive[:n]
    per_image = []
    for image in images:
        mn.reset_inference_state(model)
        metrics = {"exc_spikes": 0, "exc_spikes_at_I_below_minus_200": 0,
                   "not_reproduced_at_0_02ms": 0, "not_reproduced_at_0_01ms": 0,
                   "minimum_net_current": 0.0}
        for _ in range(n_steps):
            slot = (model.step_count + 1) % MAX_DELAY_STEPS
            current = (cortex.current[:n] + cortex.spike_buffer[slot, :n] - cortex.theta[:n]).clone()
            v, u = cortex.v[:n].clone(), cortex.u[:n].clone()
            model.stimulate("input", image)
            model.step()
            fired = cortex.fired[:n] & exc
            suspect = fired & (current < -200.0)
            metrics["exc_spikes"] += int(fired.sum())
            metrics["exc_spikes_at_I_below_minus_200"] += int(suspect.sum())
            metrics["minimum_net_current"] = min(metrics["minimum_net_current"], float(current.min()))
            if torch.any(suspect):
                for h, key in ((0.02, "not_reproduced_at_0_02ms"), (0.01, "not_reproduced_at_0_01ms")):
                    reference, _, _ = fine_step(v[suspect], u[suspect], current[suspect],
                                                cortex.a[:n][suspect], cortex.b[:n][suspect],
                                                duration=model.dt, substep=h)
                    metrics[key] += int((~reference).sum())
        per_image.append(metrics)
    return per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        parser.error("Use a new output path and a positive sample count")
    study = json.loads((args.study / "summary.json").read_text())
    root = Path(__file__).resolve().parent.parent
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    selected = study["dataset"]["validation_ids"][:args.samples]
    images = mn.downsample_images(np.asarray(raw.data[selected], dtype=np.float64) / 255.0,
                                   factor=mn.DOWNSAMPLE,
                                   target_l1=study["dataset"]["intensity_target_l1"])
    del raw
    source_hash = directory_digest(args.checkpoint)
    brain = load_brain(args.checkpoint)
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_hash,
              "validation_ids": selected, "scalar": scalar_probes(),
              "interpretation": "One-step reference probes from the coarse trajectory, not an accuracy comparison."}
    result["images"] = probe_images(brain, images, study["signature"]["config"]["inference_steps"])
    result["totals"] = {key: sum(row[key] for row in result["images"]) for key in
                        ("exc_spikes", "exc_spikes_at_I_below_minus_200",
                         "not_reproduced_at_0_02ms", "not_reproduced_at_0_01ms")}
    assert source_hash == directory_digest(args.checkpoint)
    result["checkpoint_unchanged"] = True
    atomic_json(args.output, result)
    print(json.dumps(result["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
