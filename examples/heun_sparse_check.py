"""ABBA comparison of integration, region steps or synaptic recovery with references."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch


def dense_heun(v, u, current, a, b, steps, h):
    """Preserve the pre-threshold-shortcut loop as the timing reference."""
    v, u = v.numpy().copy(), u.numpy().copy()
    current, a, b = current.numpy(), a.numpy(), b.numpy()
    fired = v >= 30.0
    v = np.minimum(v, 30.0)
    for _ in range(steps):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        du = a * (b * v - u)
        predictor_v = np.minimum(v + h * dv, 30.0)
        predictor_u = u + h * du
        next_v = v + (0.5 * h) * (
            dv + 0.04 * predictor_v * predictor_v + 5.0 * predictor_v
            + 140.0 - predictor_u + current
        )
        next_u = u + (0.5 * h) * (du + a * (b * predictor_v - predictor_u))
        crossed = next_v >= 30.0
        fraction = np.clip((30.0 - v) / np.maximum(next_v - v, 1e-12), 0.0, 1.0)
        next_u = np.where(crossed, u + fraction * (next_u - u), next_u)
        u = np.where(fired, u, next_u)
        v = np.where(fired, v, np.minimum(next_v, 30.0))
        fired |= crossed
    return torch.from_numpy(v), torch.from_numpy(u), torch.from_numpy(fired)


def main():
    import examples.mnist_benchmark as mn
    from examples.mnist_learning_check import LearningConfig, prepare_dataset
    from examples.mnist_optimization_check import simulation_digest
    from examples.training_checkpoint import atomic_json
    from src.persistence import load_brain
    import src.integration as integration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--region-step", action="store_true",
                      help="Compare Region.step against its retained pre-empty-shortcut version")
    mode.add_argument("--housekeeping", action="store_true",
                      help="Compare in-place synaptic recovery in both regions and projections")
    mode.add_argument("--preflight", action="store_true",
                      help="Compare CPU integration validation and copy overhead")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    study = json.loads((args.study / "summary.json").read_text())
    config = LearningConfig(**study["signature"]["config"])
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=".sklearn_data")
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    assert data.manifest == study["dataset"]
    brain = load_brain(args.study / "seed_101/initial/checkpoints/sample_00000000/brain")
    original_hash = simulation_digest(brain)
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_benchmark.py", root / "examples/_utils.py",
               root / "examples/mnist_learning_check.py", root / "examples/training_checkpoint.py",
               root / "examples/mnist_optimization_check.py", root / "examples/mnist_diagnosis.py",
               root / "examples/mnist_state_diagnosis.py", *sorted((root / "src").glob("*.py"))]
    if args.preflight:
        from examples.heun_preflight_reference import integrate as reference
        import src.region as region_module
        target, attribute = region_module, "integrate"
        comparison = "heun_cpu_preflight"
        sources.append(root / "examples/heun_preflight_reference.py")
    elif args.housekeeping:
        from examples.synapse_housekeeping_check import reference
        import src.synapse as synaptic_state
        target, attribute = synaptic_state, "advance_synapse_state"
        comparison = "synaptic_housekeeping"
        sources.append(root / "examples/synapse_housekeeping_check.py")
    elif args.region_step:
        from examples.region_step_reference import dense_step
        from src.region import Region
        target, attribute, reference = Region, "step", dense_step
        comparison = "region_step_reference"
        sources.append(root / "examples/region_step_reference.py")
    else:
        target, attribute, reference = integration, "_heun_cpu", dense_heun
        comparison = "heun_threshold_paths"
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"complete": False, "training": [], "inference": [], "source_sha256": hashes,
              "comparison": comparison,
              "interpretation": "ABBA comparison on copies, sharing host load. No accuracy tuning or source edits."}
    args.output.mkdir(parents=True)
    candidate = getattr(target, attribute)
    expected = None
    for sparse in (False, True, True, False):
        with patch.object(target, attribute, candidate if sparse else reference):
            model = copy.deepcopy(brain)
            start = time.perf_counter()
            with patch.object(mn, "REST_STEPS", config.rest_steps):
                mn.train_unsupervised(model, data.train_X[:20], epochs=1,
                                      train_present_steps=config.train_steps, seed=101, log_every=0)
            row = {"sparse": sparse, "wall_s": time.perf_counter() - start,
                   "state_sha256": simulation_digest(model)}
            result["training"].append(row)
            print("TRAIN " + json.dumps(row), flush=True)
            model = mn._inference_brain(brain)
            start = time.perf_counter()
            with patch.object(mn, "ASSIGN_PRESENT_STEPS", config.inference_steps):
                responses = mn.collect_readout_responses(model, data.validation_X[:40])
            result["inference"].append({"sparse": sparse, "wall_s": time.perf_counter() - start})
            if expected is None:
                expected = responses
            for key in ("exc_indices", "spikes", "voltages"):
                assert getattr(expected, key).tobytes() == getattr(responses, key).tobytes(), key
            atomic_json(args.output / "summary.json", result)
    assert len({r["state_sha256"] for r in result["training"]}) == 1
    assert simulation_digest(brain) == original_hash
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for mode in ("training", "inference"):
        timing = {flag: statistics.median(r["wall_s"] for r in result[mode] if r["sparse"] == flag)
                  for flag in (False, True)}
        result[mode + "_speedup"] = timing[False] / timing[True]
    result.update(complete=True, states_and_rng_equal=True, responses_bitwise_equal=True)
    atomic_json(args.output / "summary.json", result)
    print("COMPLETE " + json.dumps({k: result[k] for k in ("training_speedup", "inference_speedup")}), flush=True)


if __name__ == "__main__":
    main()
