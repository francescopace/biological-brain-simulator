"""Retained integration entry point before CPU validation/copy shortcuts.

The Heun kernel is shared: this reference isolates only preflight changes.
"""

import math

import torch

import src.integration as integration


def integrate(v, u, current, a, b, dt, *, method="heun", max_step=0.1):
    integration.validate_integration(dt, method, max_step)
    v, u = v.clone(), u.clone()
    if method == "legacy_euler":
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
    if not all(bool(torch.isfinite(x).all()) for x in (v, u, current, a, b)):
        raise ValueError("Non-finite neuron state or current")
    if bool((a < 0).any()) or bool((b < 0).any()):
        raise ValueError("Heun stability bound requires nonnegative a and b")
    upper_u = torch.maximum(u, 30.0 * b)
    discriminant = torch.clamp(25.0 - 0.16 * (140.0 - upper_u + current), min=0.0)
    lower_v = torch.minimum(v, (-5.0 - torch.sqrt(discriminant)) / 0.08)
    rate = max(7.4, float((-0.08 * lower_v - 5.0).max().detach()), float(a.max().detach()))
    steps = math.ceil(dt / min(max_step, 0.9 / rate))
    if steps > 100_000:
        raise ValueError("Neuron state requires more than 100,000 internal substeps")
    h = dt / steps
    if (integration.CPU_HEUN_ENABLED and v.ndim > 0 and v.dtype in (torch.float32, torch.float64)
            and all(x.device.type == "cpu" and x.dtype == v.dtype and x.shape == v.shape
                    and not x.requires_grad and not x.is_neg() and not x.is_conj()
                    for x in (v, u, current, a, b))):
        return integration._heun_cpu(v, u, current, a, b, steps, h)
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
        fired |= crossed
    return v, u, fired
