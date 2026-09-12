"""CPU validation shortcuts keep errors, output ownership and numerical results."""

from unittest.mock import patch

import pytest
import torch

from examples.heun_preflight_reference import integrate as reference
import src.integration as integration


def tensors(n=32, dtype=torch.float32):
    generator = torch.Generator().manual_seed(45)
    return [(torch.rand(n * 2, dtype=dtype, generator=generator) * scale + offset)[::2]
            for scale, offset in ((120., -100.), (40., -20.), (2000., -1500.), (.08, .02), (.1, .2))]


def same_outputs(left, right):
    for x, y in zip(left, right):
        assert x.dtype == y.dtype and x.shape == y.shape
        if x.numel():
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [0, 1, 800])
@pytest.mark.parametrize("dt", [.07, 1., 2.2])
def test_preflight_matches_reference_and_preserves_input_storage(dtype, n, dt):
    args = tensors(n, dtype)
    snapshots = [x.clone() for x in args]
    versions = [x._version for x in args]
    expected = reference(*args, dt)
    with torch.inference_mode():
        actual = integration.integrate(*args, dt)
    same_outputs(expected, actual)
    assert [x._version for x in args] == versions
    for original, saved in zip(args, snapshots):
        assert torch.equal(original, saved)
    if n:
        assert actual[0].data_ptr() != args[0].data_ptr()
        assert actual[1].data_ptr() != args[1].data_ptr()


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_have_the_same_error_without_changing_inputs(index, value):
    args = tensors()
    args[index][0] = value
    saved = [x.contiguous().view(torch.uint8).clone() for x in args]
    for fn in (reference, integration.integrate):
        with pytest.raises(ValueError, match="Non-finite neuron state or current"):
            fn(*args, 1.)
    for original, before in zip(args, saved):
        assert torch.equal(original.contiguous().view(torch.uint8), before)


@pytest.mark.parametrize("index", [3, 4])
def test_negative_stability_parameters_keep_their_error(index):
    args = tensors()
    args[index][0] = -.1
    for fn in (reference, integration.integrate):
        with pytest.raises(ValueError, match="nonnegative a and b"):
            fn(*args, 1.)


def test_large_step_count_is_still_rejected():
    args = tensors()
    args[2][0] = -1e14
    for fn in (reference, integration.integrate):
        with pytest.raises(ValueError, match="100,000 internal substeps"):
            fn(*args, 1.)


@pytest.mark.parametrize("kind", ["scalar", "mixed", "negative_view", "broadcast", "matrix", "legacy", "torch_cpu"])
def test_other_dispatch_cases_keep_original_behavior(kind):
    args, kwargs = tensors(4), {}
    if kind == "scalar":
        args = [x[0] for x in args]
    elif kind == "mixed":
        args[0] = args[0].double()
    elif kind == "negative_view":
        args[0] = torch._neg_view(args[0])
    elif kind == "broadcast":
        args[2] = args[2][:1]
    elif kind == "matrix":
        args = [x.reshape(2, 2).T for x in args]
    elif kind == "legacy":
        kwargs["method"] = "legacy_euler"
    with patch.object(integration, "CPU_HEUN_ENABLED", kind != "torch_cpu"):
        expected = reference(*args, .25, **kwargs)
        actual = integration.integrate(*args, .25, **kwargs)
    # Byte views of scalar tensors need an explicit length-one dimension.
    same_outputs([x.reshape(-1) for x in expected], [x.reshape(-1) for x in actual])


@pytest.mark.parametrize("voltage,current", [(-65., 10.), (-65., 1000.), (35., -1000.)])
def test_autograd_keeps_forward_results_and_supports_backward(voltage, current):
    args = [torch.tensor([x], dtype=torch.float64, requires_grad=True)
            for x in (voltage, -13., current, .02, .2)]
    expected = reference(*args, 1.)
    actual = integration.integrate(*args, 1.)
    same_outputs([x.detach() for x in expected], [x.detach() for x in actual])
    assert torch.autograd.gradcheck(lambda *inputs: integration.integrate(*inputs, 1.)[:2], args)
