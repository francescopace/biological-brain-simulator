"""Bounded deterministic validation of corrected v2 dynamics; no MNIST data.

Compare the same initial pulses and frozen parameters to the v1 grid and an
independent event-located ODE solve. This does not establish MNIST convergence
or accuracy. Use a new output directory; historical artifacts are never edited.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from examples import mnist_paper_timing_check as timing
from examples.mnist_paper_reference import ReferenceNetwork
from examples.training_checkpoint import atomic_json


class _RecordingNetwork(ReferenceNetwork):
    def triplet_post(self, fired, now, *, learn):
        # Both populations have been thresholded when this method is called.
        # Observe each internal boundary without changing the event scheduler.
        for name, mask in (("exc", fired), ("inh", self.pending_i)):
            for cell in np.flatnonzero(mask):
                self.events[f"{name}_spikes_ms"][cell].append(float(now))
        super().triplet_post(fired, now, learn=learn)


def spike_trace(source, duration=20., *, strong_drive=False, learn=False, adapt=False):
    before = source.state_digest()
    network = _RecordingNetwork.__new__(_RecordingNetwork)
    network.__dict__.update(copy.deepcopy(vars(source)))
    c = network.config
    if not np.isclose(duration / c.dt, round(duration / c.dt)) or duration <= 0:
        raise ValueError("Duration must be positive and grid aligned")
    network.events = {f"{name}_spikes_ms": [[] for _ in range(c.n_exc)] for name in ("exc", "inh")}
    for _ in range(round(duration / c.dt)):
        if strong_drive:
            network.ge_e[:] = network.ge_i[:] = 10000.
        network.advance(np.zeros((1, c.n_input), bool), learn=learn, adapt=adapt)
    assert source.state_digest() == before
    return network.events, network


def refractory_check(dt, substeps):
    source = timing.make_network(dt, dynamics_version=2, integration_substeps=substeps)
    events, _ = spike_trace(source, duration=40., strong_drive=True)
    h = dt / substeps
    result = {}
    for name, refractory in (("exc", source.config.refractory_e), ("inh", source.config.refractory_i)):
        ticks = np.rint(np.asarray(events[f"{name}_spikes_ms"][0]) / h).astype(np.int64)
        intervals = np.diff(ticks)
        expected = round(refractory / h) + 1
        result[name] = {"expected_interval_ticks": expected, "observed_interval_ticks": intervals.tolist(),
                        "mismatched_intervals": int(np.count_nonzero(intervals != expected))}
    return result


def run(output):
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parent.parent
    names = ("examples/mnist_paper_dynamics_check.py", "examples/mnist_paper_timing_check.py",
             "examples/mnist_paper_reference.py", "examples/training_checkpoint.py")
    hashes = {name: timing.sha256(repo / name) for name in names}
    output.mkdir(parents=True)
    for name in names:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    config = timing.make_network(dynamics_version=2, integration_substeps=8).config
    plan = {"created_at": time.time(), "source_sha256": hashes, "config": asdict(config),
            "cases": timing.cases(), "duration_ms": 20., "refractory_grids": timing.REFRACTORY_GRIDS,
            "refractory_substeps": [1, 8], "numpy": np.__version__, "scipy": timing.scipy.__version__,
            "oracle": {"rtol": 1e-10, "atol": 1e-12, "max_step": .05},
            "tight_oracle": {"rtol": 1e-12, "atol": 1e-13, "max_step": .025},
            "interpretation": __doc__}
    atomic_json(output / "plan.json", plan)
    plan_hash = timing.sha256(output / "plan.json")
    started = time.monotonic()
    result = {"complete": False, "plan_sha256": plan_hash, "cases": {}}
    atomic_json(output / "summary.json", result)
    for case in plan["cases"]:
        args = {k: v for k, v in case.items() if k != "name"}
        legacy = timing.make_network(**args)
        reference = timing.event_trace(legacy, **plan["oracle"])
        tight = timing.compare_events(reference, timing.event_trace(legacy, **plan["tight_oracle"]))
        assert tight["same_per_cell_spike_counts"] and tight["max_abs_spike_time_error_ms"] < 1e-8
        row = {"event_reference": reference, "tight_oracle_check": tight}
        for name, source in (("v1", legacy), ("v2", timing.make_network(**args, dynamics_version=2, integration_substeps=8))):
            events, _ = spike_trace(source)
            row[name] = {"events": events, "comparison": timing.compare_events(reference, events)}
        result["cases"][case["name"]] = row
    result["refractory"] = {f"dt={dt}/substeps={s}": refractory_check(dt, s)
                            for dt in plan["refractory_grids"] for s in plan["refractory_substeps"]}
    assert all(p["mismatched_intervals"] == 0 for row in result["refractory"].values() for p in row.values())
    for name, digest in hashes.items():
        assert timing.sha256(repo / name) == digest == timing.sha256(output / "source_snapshot" / name)
    assert timing.sha256(output / "plan.json") == plan_hash
    result.update(complete=True, source_unchanged=True, wall_seconds=time.monotonic() - started)
    atomic_json(output / "summary.json", result)
    print(f"COMPLETE: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
