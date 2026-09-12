"""Izhikevich integration within one externally scheduled simulation step.

The stable path uses SSP RK2 (Heun), not a change to the network timestep.
Only the first threshold crossing is retained: neurons wait for Region's
end-of-step reset, and synapses still receive at most one event per neuron.
"""

from __future__ import annotations

import math

import numpy as np
import torch


METHODS = {"heun", "legacy_euler"}
DEFAULT_MAX_STEP = 0.1
CPU_HEUN_ENABLED = True


def validate_integration(dt, method, max_step):
    if method not in METHODS:
        raise ValueError(f"Unknown integration method: {method!r}")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("integration_max_step must be finite and positive")


def _can_use_cpu_heun(v, u, current, a, b):
    return (
        CPU_HEUN_ENABLED and v.ndim > 0 and v.dtype in (torch.float32, torch.float64)
        and all(x.device.type == "cpu" and x.dtype == v.dtype and x.shape == v.shape
                and not x.requires_grad and not x.is_neg() and not x.is_conj()
                for x in (v, u, current, a, b))
    )


def integrate(v, u, current, a, b, dt, *, method="heun", max_step=DEFAULT_MAX_STEP):
    """Return voltage, recovery and first-crossing flags without changing inputs.

    For a,b >= 0 and voltage <= 30, u is bounded above by max(u0, 30*b).
    The lower root of dv/dt at that upper bound provides a lower voltage
    barrier. Limit h*abs(d(dv/dt)/dv) below 1 on that interval. Both Euler
    stages of SSP RK2 then respect the lower barrier; strong inhibition
    cannot produce the quadratic rebound seen with coarse explicit Euler.
    ``max_step`` additionally controls accuracy and threshold resolution.

    Stage voltages are capped at the spike threshold (not at an arbitrary
    negative voltage). Recovery at a crossing is linearly interpolated, then
    frozen until the external reset. This is a one-spike-per-network-step
    contract, not a continuous-time reset/multiple-spike solver.
    """
    validate_integration(dt, method, max_step)
    cpu_ready = method == "heun" and v.numel() > 0 and _can_use_cpu_heun(v, u, current, a, b)
    if cpu_ready:
        # Read-only views avoid small torch dispatches. The CPU kernel already
        # owns copies of v/u; cloning here as well would duplicate that work.
        arrays = tuple(x.numpy() for x in (v, u, current, a, b))
        if not all(bool(np.isfinite(x).all()) for x in arrays):
            raise ValueError("Non-finite neuron state or current")
        if bool((arrays[3] < 0).any()) or bool((arrays[4] < 0).any()):
            raise ValueError("Heun stability bound requires nonnegative a and b")
    else:
        v, u = v.clone(), u.clone()
    if method == "legacy_euler":
        # Preserve historical arithmetic and end-only threshold detection.
        steps = max(1, int(dt / 0.5))
        h = dt / steps
        for _ in range(steps):
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + current) * h
            du = a * (b * v - u) * h
            v += dv
            u += du
        return v, u, v >= 30.0

    if not v.numel():
        return v, u, torch.zeros_like(v, dtype=torch.bool)
    if not cpu_ready:
        if not all(bool(torch.isfinite(x).all()) for x in (v, u, current, a, b)):
            raise ValueError("Non-finite neuron state or current")
        if bool((a < 0).any()) or bool((b < 0).any()):
            raise ValueError("Heun stability bound requires nonnegative a and b")

    upper_u = torch.maximum(u, 30.0 * b)
    discriminant = torch.clamp(25.0 - 0.16 * (140.0 - upper_u + current), min=0.0)
    lower_v = torch.minimum(v, (-5.0 - torch.sqrt(discriminant)) / 0.08)
    # 7.4 is the voltage derivative at the threshold, 0.08*30 + 5.
    rate = max(7.4, float((-0.08 * lower_v - 5.0).max().detach()), float(a.max().detach()))
    steps = math.ceil(dt / min(max_step, 0.9 / rate))
    if steps > 100_000:
        raise ValueError("Neuron state requires more than 100,000 internal substeps")
    h = dt / steps
    if cpu_ready or _can_use_cpu_heun(v, u, current, a, b):
        return _heun_cpu(v, u, current, a, b, steps, h)
    fired = v >= 30.0
    v = torch.clamp(v, max=30.0)
    for _ in range(steps):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
        du = a * (b * v - u)
        predictor_v = torch.clamp(v + h * dv, max=30.0)
        predictor_u = u + h * du
        next_v = v + (0.5 * h) * (
            dv + 0.04 * predictor_v * predictor_v + 5.0 * predictor_v
            + 140.0 - predictor_u + current
        )
        next_u = u + (0.5 * h) * (du + a * (b * predictor_v - predictor_u))
        crossed = next_v >= 30.0
        fraction = torch.clamp((30.0 - v) / torch.clamp(next_v - v, min=1e-12), 0.0, 1.0)
        next_u = torch.where(crossed, u + fraction * (next_u - u), next_u)
        u = torch.where(fired, u, next_u)
        v = torch.where(fired, v, torch.clamp(next_v, max=30.0))
        # torch.where saves the mask for backward; do not mutate that version.
        fired = fired | crossed
    return v, u, fired


def _heun_cpu(v, u, current, a, b, steps, h):
    """Same Heun operations in the same order, with less CPU dispatch overhead.

    Stability bounds remain in the shared caller. No fused/fast-math operations
    are used; regression checks require bitwise agreement with the torch loop.
    The inputs are read-only and outputs own their NumPy storage.
    """
    v, u = v.numpy().copy(), u.numpy().copy()
    current, a, b = current.numpy(), a.numpy(), b.numpy()
    fired = v >= 30.0
    has_fired = bool(fired.any())
    v = np.minimum(v, 30.0)
    if has_fired and bool(fired.all()):
        return torch.from_numpy(v), torch.from_numpy(u), torch.from_numpy(fired)
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
        crossed = (next_v >= 30.0) & ~fired
        has_crossed = bool(crossed.any())
        if has_crossed:
            fraction = np.clip((30.0 - v) / np.maximum(next_v - v, 1e-12), 0.0, 1.0)
            next_u = np.where(crossed, u + fraction * (next_u - u), next_u)
        if has_fired:
            u = np.where(fired, u, next_u)
            v = np.where(fired, v, next_v)
        else:
            v, u = next_v, next_u
        if has_crossed:
            v = np.minimum(v, 30.0)
            fired |= crossed
            has_fired = True
            # The external timestep still finishes in Region.step; only
            # already-frozen, first-crossing internal dynamics are omitted.
            if bool(fired.all()):
                break
    return torch.from_numpy(v), torch.from_numpy(u), torch.from_numpy(fired)
