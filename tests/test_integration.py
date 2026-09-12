"""Numerical accuracy, inhibition stability and historical checkpoint physics."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import torch

import src.integration as integration
from src.brain import Brain
from src.integration import integrate
from src.neuron import FiringPattern, NeuronType, PATTERN_PARAMS
from src.persistence import load_brain, save_brain
from src.region import Region, RegionType


def reference(v, u, current, a, b, duration=1.0):
    def rhs(t, y):
        return (0.04*y[0]**2 + 5*y[0] + 140 - y[1] + current,
                a*(b*y[0] - y[1]))

    def threshold(t, y):
        return y[0] - 30.0

    threshold.terminal = True
    threshold.direction = 1
    solution = solve_ivp(rhs, (0, duration), (v, u), method="DOP853",
                         events=threshold, rtol=1e-11, atol=1e-11, max_step=0.01)
    assert solution.success
    return solution.y[:, -1], bool(solution.t_events[0].size)


@pytest.mark.parametrize("pattern", list(FiringPattern))
@pytest.mark.parametrize("current", [-10000., -1000., -300., -200., -50., 0., 10., 100., 1000.])
def test_one_step_matches_independent_high_accuracy_solver(pattern, current):
    a, b, *_ = PATTERN_PARAMS[pattern]
    values = (-65., -65.*b, current, a, b)
    args = [torch.tensor([x], dtype=torch.float64) for x in values]
    before = [x.clone() for x in args]
    v, u, fired = integrate(*args, 1.0)
    expected, spike = reference(*values)
    assert bool(fired[0]) == spike
    np.testing.assert_allclose([float(v[0]), float(u[0])], expected, atol=0.22, rtol=0)
    for original, saved in zip(args, before):
        assert torch.equal(original, saved)


@pytest.mark.parametrize("current", [-300., -200., -50., 0., 10.])
def test_substep_refinement_converges_to_independent_solution(current):
    values = (-65., -13., current, 0.02, 0.2)
    expected, spike = reference(*values)
    assert not spike
    errors = []
    for h in (0.1, 0.05, 0.025):
        args = [torch.tensor([x], dtype=torch.float64) for x in values]
        v, u, fired = integrate(*args, 1.0, max_step=h)
        assert not fired.any()
        errors.append(float(np.max(np.abs(np.array([v.item(), u.item()]) - expected))))
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


def test_strong_inhibition_cannot_create_spikes_over_100ms():
    region = Region("inhibition", RegionType.ASSOCIATION, max_neurons=35)
    currents = [-50., -200., -300., -1000., -10000.]
    for pattern in FiringPattern:
        for _ in currents:
            region.add_neuron(NeuronType.EXCITATORY, pattern)
    stimulus = torch.tensor(currents * len(FiringPattern), device=region.v.device)
    for step in range(100):
        region.current[:] = stimulus
        region.step(step + 1., step + 1)
        assert not region.fired.any()
        assert torch.isfinite(region.v).all() and torch.isfinite(region.u).all()
    assert not region.total_spikes.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stability_bound_covers_very_negative_initial_voltage(dtype):
    generator = torch.Generator().manual_seed(17)
    v = torch.rand(128, dtype=dtype, generator=generator) * -950. - 50.
    u = torch.rand(128, dtype=dtype, generator=generator) * 90. - 30.
    current = torch.rand(128, dtype=dtype, generator=generator) * -10000. - 300.
    a = torch.full_like(v, .1)
    b = torch.full_like(v, .26)
    next_v, next_u, fired = integrate(v, u, current, a, b, 1.)
    assert not fired.any()
    assert torch.isfinite(next_v).all() and torch.isfinite(next_u).all()
    assert (next_v < -50.).all()


@pytest.mark.parametrize("dt", [.07, .25, .75, 1.1, 2.2])
def test_fractional_external_steps_integrate_the_full_duration(dt):
    values = (-65., -13., -200., .02, .2)
    args = [torch.tensor([x], dtype=torch.float64) for x in values]
    v, u, fired = integrate(*args, dt, max_step=.005)
    expected, spike = reference(*values, duration=dt)
    assert bool(fired[0]) == spike
    np.testing.assert_allclose([v.item(), u.item()], expected, atol=.003, rtol=0)


def test_first_crossing_is_frozen_and_counted_once_dead_neurons_do_not_emit():
    region = Region("crossing", RegionType.SENSORY, max_neurons=3)
    for _ in range(3):
        region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    region.v[0] = 35.
    initial_u = float(region.u[0])
    region.neuron_alive[2] = False
    region.current[:] = 1e6
    region.step(1., 1)
    assert region.fired.tolist() == [True, True, False]
    assert region.total_spikes.tolist() == [1, 1, 0]
    assert float(region.u[0]) == pytest.approx(initial_u + float(region.d[0]))
    assert torch.isfinite(region.v).all() and torch.isfinite(region.u).all()
    assert torch.equal(region.v[:2], region.c[:2])


def test_legacy_mode_preserves_original_arithmetic_exactly():
    generator = torch.Generator().manual_seed(7)
    v = torch.randn(40, generator=generator) * 30 - 65
    u = torch.randn(40, generator=generator) * 10 - 13
    current = torch.randn(40, generator=generator) * 300
    a, b = torch.full_like(v, .02), torch.full_like(v, .2)
    for dt in (.05, .9, 1., 2.):
        expected_v, expected_u = v.clone(), u.clone()
        h = dt / max(1, int(dt / .5))
        for _ in range(max(1, int(dt / .5))):
            dv = (.04*expected_v*expected_v + 5.*expected_v + 140. - expected_u + current) * h
            du = a * (b*expected_v - expected_u) * h
            expected_v += dv
            expected_u += du
        actual_v, actual_u, fired = integrate(v, u, current, a, b, dt, method="legacy_euler")
        assert torch.equal(actual_v, expected_v)
        assert torch.equal(actual_u, expected_u)
        assert torch.equal(fired, expected_v >= 30.)


@pytest.mark.parametrize("kwargs", [{"dt": 0.}, {"dt": float("nan")},
                                    {"integration_method": "unknown"},
                                    {"integration_max_step": 0.}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Brain(**kwargs)
    with pytest.raises(ValueError):
        Region("invalid", RegionType.SENSORY, **kwargs)


def test_nonfinite_inputs_and_unsupported_parameters_fail_explicitly():
    args = [torch.tensor([x]) for x in (-65., -13., 10., .02, .2)]
    for index, value in ((0, float("nan")), (2, float("inf")), (3, -.1), (4, -.2)):
        invalid = [x.clone() for x in args]
        invalid[index][0] = value
        with pytest.raises(ValueError):
            integrate(*invalid, 1.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [0, 1, 16, 800])
@pytest.mark.parametrize("dt", [.05, .75, 1.])
def test_cpu_loop_is_bitwise_equal_to_torch_including_strided_inputs(dtype, n, dt):
    generator = torch.Generator().manual_seed(31)
    def draw(scale, offset):
        return (torch.rand(n * 2, generator=generator, dtype=dtype) * scale + offset)[::2]
    args = [draw(400., -350.), draw(90., -40.), draw(20000., -10000.),
            draw(.08, .02), draw(.1, .2)]
    before = [x.clone() for x in args]
    with patch.object(integration, "CPU_HEUN_ENABLED", False):
        expected = integrate(*args, dt)
    with patch.object(integration, "CPU_HEUN_ENABLED", True), torch.inference_mode():
        actual = integrate(*args, dt)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)
        assert left.dtype == right.dtype
        assert left.numpy().tobytes() == right.numpy().tobytes()
    for left, right in zip(args, before):
        assert torch.equal(left, right)


def test_numpy_kernel_is_not_used_with_autograd_or_mixed_dtype():
    args = [torch.tensor([x]) for x in (-65., -13., 10., .02, .2)]
    with patch.object(integration, "_heun_cpu", side_effect=AssertionError("must use torch")):
        args[0].requires_grad_()
        integrate(*args, 1.)
        args[0] = args[0].detach().double()
        integrate(*args, 1.)


def test_scalar_tensors_retain_the_torch_fallback():
    args = [torch.tensor(x) for x in (-65., -13., -300., .02, .2)]
    with patch.object(integration, "_heun_cpu", side_effect=AssertionError("must use torch")):
        v, u, fired = integrate(*args, 1.)
    assert v.ndim == 0 and u.ndim == 0 and not fired


@pytest.mark.parametrize("starting_voltage,current", [(35., -1000.), (-65., -1000.),
                                                      (-65., 1000.), (-65., 10.)])
def test_threshold_shortcuts_match_unconditional_loop(starting_voltage, current):
    from examples.heun_sparse_check import dense_heun
    args = [torch.full((16,), x) for x in (starting_voltage, -13., current, .02, .2)]
    expected = dense_heun(*args, 10, .1)
    actual = integration._heun_cpu(*args, 10, .1)
    for left, right in zip(expected, actual):
        assert left.numpy().tobytes() == right.numpy().tobytes()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_non_cpu_fallback_does_not_convert_to_numpy(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable")
    args = [torch.tensor([x], device=device) for x in (-65., -13., -300., .02, .2)]
    with patch.object(integration, "_heun_cpu", side_effect=AssertionError("must use torch")):
        v, u, fired = integrate(*args, 1.)
    assert v.device.type == device and u.device.type == device
    assert not fired.any()


def test_checkpoint_retains_region_overrides_and_exact_continuation(tmp_path):
    brain = Brain(seed=4, integration_max_step=.05)
    region = brain.add_region("input", RegionType.SENSORY, 4, connectivity=0.)
    legacy = brain.add_region("legacy", RegionType.SENSORY, 2, connectivity=0.)
    legacy.integration_method = "legacy_euler"
    region.current[:4] = torch.tensor([-1000., -300., 10., 100.], device=region.v.device)
    brain.step()
    save_brain(brain, tmp_path / "brain")
    loaded = load_brain(tmp_path / "brain")
    assert loaded.integration_method == "heun"
    assert loaded.integration_max_step == .05
    assert loaded.regions["legacy"].integration_method == "legacy_euler"
    assert loaded.regions["input"].integration_max_step == .05
    for _ in range(5):
        brain.step()
        loaded.step()
        for name in brain.regions:
            for key, value in vars(brain.regions[name]).items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, getattr(loaded.regions[name], key)), (name, key)


@pytest.mark.parametrize("where", ["meta", "region"])
def test_current_format_cannot_silently_fall_back_to_legacy_when_corrupt(tmp_path, where):
    brain = Brain(seed=4)
    brain.add_region("input", RegionType.SENSORY, 1)
    save_brain(brain, tmp_path / "brain")
    if where == "meta":
        path = tmp_path / "brain/meta.json"
        data = json.loads(path.read_text())
        data.pop("integration_method")
        path.write_text(json.dumps(data))
    else:
        path = tmp_path / "brain/regions/region_0.pt"
        data = torch.load(path, weights_only=True)
        data.pop("integration_method")
        torch.save(data, path)
    with pytest.raises(KeyError, match="integration_method"):
        load_brain(tmp_path / "brain")


@pytest.mark.parametrize("version", [4, 5])
def test_old_checkpoint_keeps_legacy_physics(tmp_path, version):
    brain = Brain(seed=4, integration_method="legacy_euler")
    brain.add_region("input", RegionType.SENSORY, 4, connectivity=0.)
    save_brain(brain, tmp_path / "brain")
    meta_path = tmp_path / "brain/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = version
    for key in ("integration_method", "integration_max_step"):
        meta.pop(key)
    meta_path.write_text(json.dumps(meta))
    region_path = tmp_path / "brain/regions/region_0.pt"
    state = torch.load(region_path, weights_only=True)
    for key in ("integration_method", "integration_max_step"):
        state.pop(key)
    torch.save(state, region_path)
    loaded = load_brain(tmp_path / "brain")
    assert loaded.integration_method == "legacy_euler"
    assert loaded.regions["input"].integration_method == "legacy_euler"
    for _ in range(4):
        brain.regions["input"].current[:4] = -300.
        loaded.regions["input"].current[:4] = -300.
        brain.step()
        loaded.step()
        assert torch.equal(brain.regions["input"].v, loaded.regions["input"].v)
        assert torch.equal(brain.regions["input"].u, loaded.regions["input"].u)
    assert loaded.add_region("new", RegionType.SENSORY, 1).integration_method == "legacy_euler"
