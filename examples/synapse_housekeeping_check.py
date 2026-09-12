"""Check and time synaptic recovery without allocating intermediate state arrays."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from examples.training_checkpoint import array_digest, atomic_json
from src.synapse import advance_synapse_state


def reference(target, ns, recovery_rate=.005, transmission_decay=.999):
    s = slice(0, ns)
    target.syn_resource[s] = torch.clamp(target.syn_resource[s] + recovery_rate, max=1.0)
    target.syn_facilitation[s] *= .98
    target.syn_age[s] += 1
    target.syn_recent[s] *= transmission_decay


def inplace_candidate(target, ns, recovery_rate=.005, transmission_decay=.999):
    advance_synapse_state(target, ns, recovery_rate, transmission_decay)


def make_state(ns, dtype=torch.float32):
    rng = torch.Generator().manual_seed(31)
    return SimpleNamespace(
        syn_resource=torch.rand(ns + 7, generator=rng, dtype=dtype),
        syn_facilitation=torch.rand(ns + 7, generator=rng, dtype=dtype),
        syn_age=torch.arange(ns + 7, dtype=torch.int32),
        syn_recent=torch.rand(ns + 7, generator=rng, dtype=dtype),
    )


def state_digest(target):
    return array_digest(*(getattr(target, key).detach().cpu().numpy() for key in sorted(vars(target))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--sizes", nargs="+", type=int, default=[0, 47000, 160000])
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1 or min(args.sizes) < 0:
        parser.error("Use a new output file and nonnegative sizes/positive iterations")
    torch.set_num_threads(4)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    kernel = Path(__file__).resolve().parent.parent / "src/synapse.py"
    kernel_hash = hashlib.sha256(kernel.read_bytes()).hexdigest()
    result = {"complete": False, "source_sha256": source_hash, "torch": torch.__version__,
              "kernel_sha256": kernel_hash,
              "threads": torch.get_num_threads(), "iterations": args.iterations, "cases": [],
              "interpretation": "ABBA kernel timings under current host load; not a whole-brain speedup."}
    for ns in args.sizes:
        initial = make_state(ns)
        before = state_digest(initial)
        trials = []
        for fast in (False, True, True, False):
            model = copy.deepcopy(initial)
            function = inplace_candidate if fast else reference
            warm = copy.deepcopy(initial)
            for _ in range(10):
                function(warm, ns)
            start = time.perf_counter()
            for _ in range(args.iterations):
                function(model, ns)
            trials.append({"fast": fast, "wall_s": time.perf_counter() - start,
                           "state_sha256": state_digest(model)})
        assert len({r["state_sha256"] for r in trials}) == 1
        assert state_digest(initial) == before
        medians = {flag: statistics.median(r["wall_s"] for r in trials if r["fast"] == flag)
                   for flag in (False, True)}
        row = {"synapses": ns, "trials": trials, "speedup": medians[False] / medians[True]}
        result["cases"].append(row)
        atomic_json(args.output, result)
        print("CASE " + json.dumps({"synapses": ns, "speedup": row["speedup"]}), flush=True)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == kernel_hash
    result.update(complete=True, states_bitwise_equal=True)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
