"""Isolated NumPy reference for the released Diehl--Cook MNIST triplet demo.

Equations/parameters: doi:10.3389/fncom.2015.00099 and
github.com/peter-u-diehl/stdp-mnist (Diehl&Cook_spiking_MNIST.py and generator).
This is an independent implementation, not Brian-1 trajectory equivalence and
not the power-law rule associated with the paper's 87.0% result. Units: ms, mV,
and conductance relative to 1 nS. No dependency on the production Brain model.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class ReferenceConfig:
    n_input: int = 784
    n_exc: int = 400
    dt: float = .5
    presentation_ms: float = 350.
    rest_ms: float = 150.
    tau_e: float = 100.
    tau_i: float = 10.
    tau_ge: float = 1.
    tau_gi: float = 2.
    rest_e: float = -65.
    rest_i: float = -60.
    reset_e: float = -65.
    reset_i: float = -45.
    threshold_e: float = -52.
    threshold_i: float = -40.
    reversal_e: float = 0.
    reversal_ie: float = -100.
    reversal_ii: float = -85.
    refractory_e: float = 5.
    refractory_i: float = 2.
    theta_initial: float = 20.
    theta_offset: float = 20.
    theta_plus: float = .05
    theta_tau: float = 1e7
    weight_max: float = 1.
    incoming_sum: float = 78.
    exc_to_inh: float = 10.4
    inh_to_exc: float = 17.
    input_delay_max: float = 10.
    trace_pre_tau: float = 20.
    trace_post1_tau: float = 20.
    trace_post2_tau: float = 40.
    eta_pre: float = .0001
    eta_post: float = .01
    min_spikes: int = 5
    max_attempts: int = 10
    dynamics_version: int = 2
    integration_substeps: int = 8

    @classmethod
    def from_dict(cls, values):
        """Unversioned historical protocols always select the original dynamics."""
        values = dict(values)
        if "dynamics_version" not in values:
            if "integration_substeps" in values:
                raise ValueError("Substeps require an explicit dynamics version")
            values.update(dynamics_version=1, integration_substeps=1)
        return cls(**values)

    def state_config(self):
        """Keep the original v1 checkpoint schema and state hashes unchanged."""
        values = asdict(self)
        if self.dynamics_version == 1:
            values.pop("dynamics_version")
            values.pop("integration_substeps")
        return values

    def validate(self):
        if type(self.dynamics_version) is not int or self.dynamics_version not in (1, 2):
            raise ValueError("Unsupported dynamics version")
        if type(self.integration_substeps) is not int or self.integration_substeps < 1:
            raise ValueError("integration_substeps must be a positive integer")
        if self.dynamics_version == 1 and self.integration_substeps != 1:
            raise ValueError("Legacy dynamics require one integration substep")
        for name in ("n_input", "n_exc", "min_spikes", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Parameters must be finite")
        for name in ("dt", "presentation_ms", "tau_e", "tau_i", "tau_ge", "tau_gi",
                     "theta_tau", "weight_max", "incoming_sum", "trace_pre_tau",
                     "trace_post1_tau", "trace_post2_tau"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("rest_ms", "theta_plus", "exc_to_inh", "inh_to_exc", "input_delay_max",
                     "eta_pre", "eta_post", "refractory_e", "refractory_i"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("presentation_ms", "rest_ms", "input_delay_max", "refractory_e", "refractory_i"):
            if not math.isclose(getattr(self, name) / self.dt, round(getattr(self, name) / self.dt)):
                raise ValueError(f"{name} must be a whole number of timesteps")
        if self.incoming_sum > self.n_input * self.weight_max:
            raise ValueError("Incoming weight budget exceeds bounded capacity")


def conductance_step(v, ge, gi, *, dt, tau, rest, reversal_e, reversal_i, tau_ge=1., tau_gi=2.):
    """Exponential midpoint: exact voltage flow with midpoint conductances.

    Conductances decay exactly between grid-aligned arrival events. This is a
    second-order approximation to the coupled voltage/conductance equations,
    not the Brian-1 scheduler/integrator. Tests compare an independent ODE solve.
    """
    mid_e, mid_i = ge * math.exp(-dt / (2 * tau_ge)), gi * math.exp(-dt / (2 * tau_gi))
    total = 1. + mid_e + mid_i
    equilibrium = (rest + mid_e * reversal_e + mid_i * reversal_i) / total
    return (equilibrium + (v - equilibrium) * np.exp(-dt * total / tau),
            ge * math.exp(-dt / tau_ge), gi * math.exp(-dt / tau_gi))


def poisson_tape(image, config, *, seed, attempt=0):
    """Bernoulli grid approximation to independent pixel-rate Poisson processes.

    Raw pixels are in [0,255]. Base rate is pixel/4 Hz; each retry adds pixel/8
    Hz, as input_intensity=2,3,... in the released demo. Never clip probabilities.
    """
    image = np.asarray(image, dtype=np.float64)
    if (image.shape != (config.n_input,) or not np.isfinite(image).all()
            or np.any(image < 0) or np.any(image > 255) or type(attempt) is not int or attempt < 0):
        raise ValueError("Expected finite raw pixels in [0,255] and a nonnegative attempt")
    probability = image * (2 + attempt) / 8 * config.dt / 1000.
    if np.any(probability >= 1.):
        raise ValueError("Poisson grid too coarse for this input rate")
    rng = np.random.default_rng(seed)
    return rng.random((round(config.presentation_ms / config.dt), config.n_input)) < probability


class ReferenceNetwork:
    """Dense input, one-to-one E->I, all-to-all I->E excluding the paired cell."""

    def __init__(self, config=ReferenceConfig(), *, seed=201):
        config.validate()
        self.config = config
        rng = np.random.default_rng(seed)
        self.weights = (rng.random((config.n_input, config.n_exc)) + .01) * .3
        # Uniform delays rounded down to the timestep grid, including zero.
        self.delays = np.floor(rng.uniform(0., config.input_delay_max,
                                size=self.weights.shape) / config.dt).astype(np.int32)
        self.theta = np.full(config.n_exc, config.theta_initial)
        self.reset_transients()
        # The released demo starts below rest. Independent evaluation uses rest.
        self.v_e -= 40.
        self.v_i -= 40.

    def reset_transients(self):
        c = self.config
        self.step_count = 0
        self.v_e, self.v_i = np.full(c.n_exc, c.rest_e), np.full(c.n_exc, c.rest_i)
        for name in ("ge_e", "gi_e", "ge_i", "gi_i"):
            setattr(self, name, np.zeros(c.n_exc))
        self.last_pre = np.full(self.weights.shape, -1e15)
        self.last_post = np.full(c.n_exc, -1e15)
        self.last_inh = np.full(c.n_exc, -1e15)
        self.pending_e = np.zeros(c.n_exc, dtype=bool)
        self.pending_i = np.zeros(c.n_exc, dtype=bool)
        self.pending_times = np.empty(0, dtype=np.int64)
        self.pending_synapses = np.empty(0, dtype=np.int64)
        if c.dynamics_version == 2:
            # Integer release deadlines, in internal integration ticks. These
            # are separate from the millisecond timestamps used by STDP.
            self.release_e = np.zeros(c.n_exc, dtype=np.int64)
            self.release_i = np.zeros(c.n_exc, dtype=np.int64)

    def state_digest(self):
        digest = hashlib.sha256(json.dumps(self.config.state_config(), sort_keys=True).encode())
        digest.update(str(self.step_count).encode())
        for name, value in sorted(vars(self).items()):
            if isinstance(value, np.ndarray):
                digest.update(str((name, value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
        return digest.hexdigest()

    def normalize(self):
        """Divisive column normalization followed by explicit bound enforcement."""
        sums = self.weights.sum(axis=0)
        if np.any(sums <= 0) or not np.isfinite(sums).all():
            raise ValueError("Cannot normalize a zero/nonfinite input column")
        self.weights *= self.config.incoming_sum / sums
        np.clip(self.weights, 0., self.config.weight_max, out=self.weights)

    def _events(self, tape):
        """Build a sparse arrival schedule; no dense synapse scan per timestep."""
        times, inputs = np.nonzero(tape)
        n = self.config.n_exc
        arrivals = (times[:, None] + self.delays[inputs] + self.step_count).ravel()
        synapses = (inputs[:, None] * n + np.arange(n)).ravel()
        arrivals = np.concatenate((self.pending_times, arrivals))
        synapses = np.concatenate((self.pending_synapses, synapses))
        end = self.step_count + len(tape)
        future = arrivals >= end
        self.pending_times, self.pending_synapses = arrivals[future], synapses[future]
        arrivals, synapses = arrivals[~future], synapses[~future]
        order = np.argsort(arrivals, kind="stable")
        arrivals, synapses = arrivals[order], synapses[order]
        counts = np.bincount(arrivals - self.step_count, minlength=len(tape))
        return synapses, np.concatenate(([0], np.cumsum(counts)))

    def triplet_pre(self, synapses, now, *, learn):
        """Arrival: transmit the pre-update weight, then reset trace and apply LTD."""
        c = self.config
        weights, previous = self.weights.ravel(), self.last_pre.ravel()
        posts = synapses % c.n_exc
        self.ge_e += np.bincount(posts, weights=weights[synapses], minlength=c.n_exc)
        if learn:
            weights[synapses] = np.maximum(0., weights[synapses] - c.eta_pre *
                np.exp(-(now - self.last_post[posts]) / c.trace_post1_tau))
        previous[synapses] = now

    def triplet_post(self, fired, now, *, learn):
        c = self.config
        columns = np.flatnonzero(fired)
        if learn and columns.size:
            pre = np.exp(-(now - self.last_pre[:, columns]) / c.trace_pre_tau)
            post2_before = np.exp(-(now - self.last_post[columns]) / c.trace_post2_tau)
            self.weights[:, columns] = np.minimum(c.weight_max,
                self.weights[:, columns] + c.eta_post * pre * post2_before)
        self.last_post[columns] = now

    def advance(self, tape, *, learn=False, adapt=False):
        """Advance fixed input events; learning/adaptation are explicit and separate.

        At a grid boundary: recurrent and external arrivals, presynaptic STDP,
        voltage/conductance integration, threshold/reset, postsynaptic STDP.
        In v2 this entire sequence runs on dt/integration_substeps, including
        recurrence, STDP and theta. Input tapes, delays, pending arrival times
        and step_count retain their outer dt units. Voltage features average
        all internal endpoint samples. Calls finish only at outer boundaries.
        V1 preserves the original single-step dynamics for historical replay.
        Refractory voltage is clamped to reset; conductances keep decaying.
        """
        tape = np.asarray(tape)
        if tape.ndim != 2 or tape.shape[1] != self.config.n_input or tape.dtype != bool or not len(tape):
            raise ValueError("Expected nonempty boolean timestep-by-input spike tape")
        c = self.config
        synapses, bounds = self._events(tape)
        counts_e, counts_i = np.zeros(c.n_exc, dtype=np.int32), np.zeros(c.n_exc, dtype=np.int32)
        voltage_sum = np.zeros(c.n_exc)
        substeps = c.integration_substeps
        dt = c.dt / substeps
        refractory_e = round(c.refractory_e / dt)
        refractory_i = round(c.refractory_i / dt)
        for internal in range(len(tape) * substeps):
            i, substep = divmod(internal, substeps)
            tick = self.step_count * substeps + substep
            now = self.step_count * c.dt if c.dynamics_version == 1 else tick * dt
            self.ge_i += self.pending_e * c.exc_to_inh
            self.gi_e += (self.pending_i.sum() - self.pending_i.astype(np.int64)) * c.inh_to_exc
            if substep == 0:
                self.triplet_pre(synapses[bounds[i]:bounds[i + 1]], now, learn=learn)
            if adapt:
                self.theta *= math.exp(-dt / c.theta_tau)
            self.v_e, self.ge_e, self.gi_e = conductance_step(self.v_e, self.ge_e, self.gi_e,
                dt=dt, tau=c.tau_e, rest=c.rest_e, reversal_e=c.reversal_e,
                reversal_i=c.reversal_ie, tau_ge=c.tau_ge, tau_gi=c.tau_gi)
            self.v_i, self.ge_i, self.gi_i = conductance_step(self.v_i, self.ge_i, self.gi_i,
                dt=dt, tau=c.tau_i, rest=c.rest_i, reversal_e=c.reversal_e,
                reversal_i=c.reversal_ii, tau_ge=c.tau_ge, tau_gi=c.tau_gi)
            # A spike is timestamped at this interval's end.
            if c.dynamics_version == 1:
                end = now + dt
                blocked_e = now - self.last_post < c.refractory_e
                blocked_i = now - self.last_inh < c.refractory_i
            else:
                end = (tick + 1) * dt
                blocked_e = tick < self.release_e
                blocked_i = tick < self.release_i
            self.v_e[blocked_e], self.v_i[blocked_i] = c.reset_e, c.reset_i
            self.pending_e = (~blocked_e) & (self.v_e > c.threshold_e + self.theta - c.theta_offset)
            self.pending_i = (~blocked_i) & (self.v_i > c.threshold_i)
            self.v_e[self.pending_e], self.v_i[self.pending_i] = c.reset_e, c.reset_i
            self.triplet_post(self.pending_e, end, learn=learn)
            self.last_inh[self.pending_i] = end
            if c.dynamics_version == 2:
                self.release_e[self.pending_e] = tick + 1 + refractory_e
                self.release_i[self.pending_i] = tick + 1 + refractory_i
            if adapt:
                self.theta[self.pending_e] += c.theta_plus
            counts_e += self.pending_e
            counts_i += self.pending_i
            voltage_sum += self.v_e
            if substep == substeps - 1:
                self.step_count += 1
        if not np.isfinite(self.v_e).all() or not np.isfinite(self.weights).all():
            raise FloatingPointError("Nonfinite reference state")
        return counts_e, counts_i, voltage_sum / (len(tape) * substeps)

    def rest(self, *, learn=False, adapt=False):
        steps = round(self.config.rest_ms / self.config.dt)
        if steps:
            self.advance(np.zeros((steps, self.config.n_input), dtype=bool), learn=learn, adapt=adapt)

    def save(self, path, metadata):
        """Publish an immutable, pickle-free complete state at any step boundary."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".paper-state-", dir=path.parent) as folder:
            staged = Path(folder) / "state"
            staged.mkdir()
            arrays = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
            np.savez_compressed(staged / "arrays.npz", **arrays)
            envelope = {"version": self.config.dynamics_version, "config": self.config.state_config(), "step_count": self.step_count,
                "state_sha256": self.state_digest(), "metadata": metadata,
                "arrays_sha256": hashlib.sha256((staged / "arrays.npz").read_bytes()).hexdigest()}
            (staged / "metadata.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
            if path.exists():
                raise FileExistsError(path)
            os.rename(staged, path)

    @classmethod
    def load(cls, path, expected_metadata):
        path = Path(path)
        envelope = json.loads((path / "metadata.json").read_text())
        if envelope["version"] not in (1, 2) or envelope["metadata"] != expected_metadata:
            raise ValueError("Reference checkpoint protocol mismatch")
        if hashlib.sha256((path / "arrays.npz").read_bytes()).hexdigest() != envelope["arrays_sha256"]:
            raise ValueError("Reference checkpoint integrity mismatch")
        network = cls(ReferenceConfig.from_dict(envelope["config"]))
        if network.config.dynamics_version != envelope["version"]:
            raise ValueError("Reference checkpoint dynamics version mismatch")
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            expected = {k for k, v in vars(network).items() if isinstance(v, np.ndarray)}
            if set(arrays.files) != expected:
                raise ValueError("Reference checkpoint state keys differ")
            for name in arrays.files:
                setattr(network, name, arrays[name].copy())
        network.step_count = envelope["step_count"]
        if network.state_digest() != envelope["state_sha256"]:
            raise ValueError("Reference checkpoint state mismatch")
        return network


def frozen_responses(network, images, row_ids, *, seed):
    """Independent-image pilot evaluation with per-row keyed Poisson draws.

    Unlike the original demo, reset electrical state before each image. Theta,
    weights and RNG-independent source state remain fixed. Retries retain state
    within that image; a finite retry cap fails rather than scoring an omission.
    """
    if len(images) != len(row_ids):
        raise ValueError("Image and row-ID counts differ")
    before = network.state_digest()
    frozen = copy.deepcopy(network)
    spikes, volts, attempts = [], [], []
    for image, row in zip(images, row_ids):
        frozen.reset_transients()
        for attempt in range(network.config.max_attempts):
            tape = poisson_tape(image, network.config, seed=np.random.SeedSequence([seed, int(row), attempt]), attempt=attempt)
            response, _, voltage = frozen.advance(tape)
            frozen.rest()
            if response.sum() >= network.config.min_spikes:
                spikes.append(response)
                volts.append(voltage)
                attempts.append(attempt + 1)
                break
        else:
            raise RuntimeError(f"Input row {row} exhausted the frozen retry budget")
    assert network.state_digest() == before
    return np.asarray(spikes), np.asarray(volts), np.asarray(attempts)


def class_average_predictions(train, labels, validation, *, classes=tuple(range(10))):
    """Assign each responsive neuron using training labels; average rates by class.

    Unresponsive neurons remain unassigned. All-zero class scores abstain (-1),
    avoiding the released demo's arbitrary class for silent examples/neurons.
    """
    train, labels, validation = np.asarray(train), np.asarray(labels), np.asarray(validation)
    if (train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]
            or labels.shape != (len(train),) or not len(train) or not len(classes)
            or not np.isfinite(train).all() or not np.isfinite(validation).all()
            or np.any(train < 0) or np.any(validation < 0) or not np.isin(labels, classes).all()):
        raise ValueError("Expected aligned nonnegative responses and known class labels")
    if any(not np.any(labels == cls) for cls in classes):
        raise ValueError("Each class needs readout examples")
    means = np.asarray([train[labels == cls].mean(axis=0) for cls in classes])
    assignments = np.asarray(classes)[means.argmax(axis=0)].astype(np.int64)
    assignments[means.max(axis=0) == 0] = -1
    scores = np.zeros((len(validation), len(classes)))
    for j, cls in enumerate(classes):
        mask = assignments == cls
        if np.any(mask):
            scores[:, j] = validation[:, mask].mean(axis=1)
    predictions = np.asarray(classes)[scores.argmax(axis=1)].astype(np.int64)
    predictions[scores.max(axis=1) == 0] = -1
    return predictions, assignments, scores
