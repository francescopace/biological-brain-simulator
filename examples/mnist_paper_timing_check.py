"""Deterministic spike/refractory audit; no MNIST training or model changes.

The independent event-located oracle covers frozen conductance equations with
initial conductance pulses, clamped refractory voltage and zero-delay recurrence.
It is not a reproduction of Brian or a replacement for the production simulator.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from examples.mnist_paper_reference import ReferenceConfig, ReferenceNetwork
from examples.training_checkpoint import atomic_json


GRIDS = (.5, .25, .125, .0625)
REFRACTORY_GRIDS = (*GRIDS, .2, .1, .05)
DURATION_MS = 20.
INTERPRETATION = (
    "Deterministic audit of frozen spike timing, refractory release and recurrent "
    "delivery in tiny networks. No dataset, training, fitting or accuracy. Compare "
    "the unchanged reference to an independent DOP853 event-located solution of "
    "the same frozen conductance equations with initial pulses, absolute clamped "
    "refractory intervals and immediate recurrence at spike times. The oracle is "
    "checked with tighter tolerances and a smaller step cap, not asserted to be "
    "Brian-equivalent or ground truth for MNIST. Strong-drive refractory tests "
    "separately compare integer step intervals to the expected refractory-grid "
    "budget plus one integration interval. Decimal-grid defects are reported, "
    "not fixed. Results do not quantify a contribution to MNIST accuracy or "
    "justify changing learning parameters. Source and defaults remain unchanged."
)


def make_network(dt=.5, *, ge_e=(0., 0.), ge_i=(0., 0.), recurrent=False,
                 dynamics_version=1, integration_substeps=1):
    # This audit explicitly characterizes v1, including its known defects.
    config = replace(ReferenceConfig(dynamics_version=dynamics_version, integration_substeps=integration_substeps), dt=dt, n_input=1, n_exc=2, incoming_sum=.5,
                     presentation_ms=DURATION_MS, rest_ms=0., input_delay_max=0.,
                     exc_to_inh=10.4 if recurrent else 0., inh_to_exc=17. if recurrent else 0.)
    network = ReferenceNetwork(config, seed=201)
    network.reset_transients()
    network.weights[:] = 0.
    network.delays[:] = 0
    network.ge_e[:] = ge_e
    network.ge_i[:] = ge_i
    return network


def grid_trace(source, duration=DURATION_MS, *, strong_drive=False):
    """Observe one-step calls, checking recurrence before conductance decay."""
    before = source.state_digest()
    network = copy.deepcopy(source)
    c = network.config
    if c.integration_substeps != 1:
        raise ValueError("This legacy observer requires one substep; use dynamics_check.spike_trace for v2")
    if not np.isclose(duration / c.dt, round(duration / c.dt)):
        raise ValueError("Duration must be grid aligned")
    trace = {"exc_spikes_ms": [[] for _ in range(c.n_exc)], "inh_spikes_ms": [[] for _ in range(c.n_exc)],
             "exc_steps": [[] for _ in range(c.n_exc)], "inh_steps": [[] for _ in range(c.n_exc)]}
    for _ in range(round(duration / c.dt)):
        if strong_drive:
            network.ge_e[:] = network.ge_i[:] = 10000.
        expected_ge_i = (network.ge_i + network.pending_e * c.exc_to_inh) * np.exp(-c.dt / c.tau_ge)
        expected_gi_e = (network.gi_e + (network.pending_i.sum() - network.pending_i.astype(int)) * c.inh_to_exc) * np.exp(-c.dt / c.tau_gi)
        exc, inh, _ = network.advance(np.zeros((1, c.n_input), dtype=bool), learn=False, adapt=False)
        np.testing.assert_allclose(network.ge_i, expected_ge_i, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(network.gi_e, expected_gi_e, rtol=1e-13, atol=1e-13)
        for name, counts, times in (("exc", exc, network.last_post), ("inh", inh, network.last_inh)):
            for cell in np.flatnonzero(counts):
                trace[f"{name}_spikes_ms"][cell].append(float(times[cell]))
                trace[f"{name}_steps"][cell].append(network.step_count)
    assert source.state_digest() == before
    return trace, network


def event_trace(source, duration=DURATION_MS, *, rtol=1e-10, atol=1e-12, max_step=.05):
    """Independent ODE/event implementation, without ReferenceNetwork.advance."""
    c, n = source.config, source.config.n_exc
    if source.step_count or source.pending_e.any() or source.pending_i.any() or source.pending_times.size:
        raise ValueError("Oracle accepts an untouched pulse initial condition")
    state = np.concatenate([getattr(source, k) for k in ("v_e", "ge_e", "gi_e", "v_i", "ge_i", "gi_i")])
    thresholds = np.concatenate((np.full(n, c.threshold_e) + source.theta - c.theta_offset, np.full(n, c.threshold_i)))
    resets = np.r_[np.full(n, c.reset_e), np.full(n, c.reset_i)]
    refractory = np.r_[np.full(n, c.refractory_e), np.full(n, c.refractory_i)]
    release = np.zeros(2 * n)
    v_indices = np.r_[np.arange(n), 3 * n + np.arange(n)]
    now = 0.
    trace = {"exc_spikes_ms": [[] for _ in range(n)], "inh_spikes_ms": [[] for _ in range(n)]}
    for _ in range(10000):
        if now >= duration - 1e-12:
            return trace
        active = release <= now + 1e-12
        later = release[release > now + 1e-12]
        stop = min(duration, float(later.min()) if later.size else duration)
        def rhs(t, y):
            ve, ge, gi, vi, gie, gii = y.reshape(6, n)
            dve = ((c.rest_e-ve) + ge*(c.reversal_e-ve) + gi*(c.reversal_ie-ve)) / c.tau_e
            dvi = ((c.rest_i-vi) + gie*(c.reversal_e-vi) + gii*(c.reversal_ii-vi)) / c.tau_i
            return np.concatenate((np.where(active[:n], dve, 0.), -ge/c.tau_ge, -gi/c.tau_gi,
                                   np.where(active[n:], dvi, 0.), -gie/c.tau_ge, -gii/c.tau_gi))
        events = []
        for cell in np.flatnonzero(active):
            def threshold(t, y, cell=int(cell)):
                return y[v_indices[cell]] - thresholds[cell]
            threshold.terminal, threshold.direction = True, 1.
            events.append(threshold)
        solution = solve_ivp(rhs, (now, stop), state, method="DOP853", events=events,
                             rtol=rtol, atol=atol, max_step=max_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        now, state = float(solution.t[-1]), solution.y[:, -1].copy()
        if solution.status != 1:
            continue
        # A terminal event may stop before reporting a simultaneous peer event.
        fired = active & (state[v_indices] >= thresholds - 1e-9)
        if not fired.any():
            raise RuntimeError("Threshold event without a firing cell")
        for cell in np.flatnonzero(fired):
            trace["exc_spikes_ms" if cell < n else "inh_spikes_ms"][cell % n].append(now)
        state[v_indices[fired]] = resets[fired]
        release[fired] = now + refractory[fired]
        # Emitted spikes affect conductances at their actual event timestamp.
        state[4*n:5*n] += fired[:n] * c.exc_to_inh
        state[2*n:3*n] += (fired[n:].sum() - fired[n:].astype(int)) * c.inh_to_exc
    raise RuntimeError("Oracle event budget exhausted")


def compare_events(reference, observed):
    deltas = []
    same = True
    for key in ("exc_spikes_ms", "inh_spikes_ms"):
        if len(reference[key]) != len(observed[key]):
            raise ValueError("Event traces have different neuron counts")
        for left, right in zip(reference[key], observed[key]):
            if len(left) != len(right):
                same = False
            else:
                deltas.extend(np.asarray(right) - np.asarray(left))
    return {"same_per_cell_spike_counts": same,
            "max_abs_spike_time_error_ms": float(max(map(abs, deltas), default=0.)) if same else None,
            "exc_counts": [len(s) for s in observed["exc_spikes_ms"]],
            "inh_counts": [len(s) for s in observed["inh_spikes_ms"]]}


def refractory_check(dt):
    source = make_network(dt)
    trace, _ = grid_trace(source, duration=40., strong_drive=True)
    result = {}
    for name, refractory in (("exc", source.config.refractory_e), ("inh", source.config.refractory_i)):
        expected = round(refractory / dt) + 1
        intervals = np.diff(trace[f"{name}_steps"][0])
        result[name] = {"expected_interval_steps": expected, "observed_interval_steps": intervals.tolist(),
                        "mismatched_intervals": int(np.count_nonzero(intervals != expected)),
                        "excess_steps": (intervals - expected).tolist()}
    return result


def cases():
    result = [{"name": f"isolated_e_{g}", "ge_e": [g, 0.], "ge_i": [0., 0.], "recurrent": False}
              for g in (20., 25., 40., 80.)]
    result += [{"name": f"isolated_i_{g}", "ge_e": [0., 0.], "ge_i": [g, 0.], "recurrent": False}
               for g in (10.4, 20.)]
    result += [{"name": f"competition_80_{g}", "ge_e": [80., g], "ge_i": [0., 0.], "recurrent": True}
               for g in (25., 30., 35., 40., 80.)]
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parent.parent
    names = ("examples/mnist_paper_timing_check.py", "examples/mnist_paper_reference.py", "examples/training_checkpoint.py")
    hashes = {name: sha256(repo / name) for name in names}
    output.mkdir(parents=True)
    for name in names:
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / name).read_bytes())
    plan = {"created_at": time.time(), "source_sha256": hashes, "cases": cases(), "grids": list(GRIDS),
            "refractory_grids": list(REFRACTORY_GRIDS), "duration_ms": DURATION_MS, "refractory_duration_ms": 40.,
            "base_config": asdict(make_network().config), "numpy": np.__version__, "scipy": scipy.__version__,
            "oracle": {"rtol": 1e-10, "atol": 1e-12, "max_step": .05},
            "tight_oracle": {"rtol": 1e-12, "atol": 1e-13, "max_step": .025}, "interpretation": INTERPRETATION}
    atomic_json(output / "plan.json", plan)
    plan_hash = sha256(output / "plan.json")
    started = time.monotonic()
    result = {"complete": False, "plan_sha256": plan_hash, "interpretation": INTERPRETATION, "cases": {}}
    atomic_json(output / "summary.json", result)
    for case in plan["cases"]:
        config = {k: v for k, v in case.items() if k != "name"}
        source = make_network(**config)
        reference = event_trace(source, **plan["oracle"])
        tighter = event_trace(source, **plan["tight_oracle"])
        agreement = compare_events(reference, tighter)
        if not agreement["same_per_cell_spike_counts"] or agreement["max_abs_spike_time_error_ms"] > 1e-8:
            raise RuntimeError("Independent oracle did not stabilize")
        values = {"event_reference": reference, "tight_oracle_check": agreement, "grids": {}}
        for dt in GRIDS:
            trace, _ = grid_trace(make_network(dt, **config))
            values["grids"][str(dt)] = {"events": trace, "comparison": compare_events(reference, trace)}
        result["cases"][case["name"]] = values
        atomic_json(output / "summary.json", result)
        print(f"TIMING: {case['name']}", flush=True)
    result["refractory"] = {str(dt): refractory_check(dt) for dt in REFRACTORY_GRIDS}
    for name, digest in hashes.items():
        assert sha256(repo / name) == digest == sha256(output / "source_snapshot" / name)
    assert sha256(output / "plan.json") == plan_hash
    result.update(complete=True, source_unchanged=True, recurrent_delivery_checks_passed=True,
                  wall_seconds=time.monotonic() - started,
                  refractory_mismatch_grids=[dt for dt, values in result["refractory"].items()
                      if any(v["mismatched_intervals"] for v in values.values())])
    atomic_json(output / "summary.json", result)
    print(f"COMPLETE: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
