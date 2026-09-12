"""Recovery shortcuts preserve active slices, aliases, rounding and fallback state."""

import copy

import pytest
import torch

from examples.synapse_housekeeping_check import inplace_candidate, make_state, reference


def assert_same(left, right):
    for key in vars(left):
        a, b = getattr(left, key), getattr(right, key)
        assert torch.equal(a.detach().contiguous().view(torch.uint8),
                           b.detach().contiguous().view(torch.uint8)), key


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 17, 4700])
@pytest.mark.parametrize("strided", [False, True])
def test_recovery_matches_original_values_and_unused_capacity(dtype, count, strided):
    initial = make_state(count * 2 + 7 if strided else count, dtype=dtype)
    if strided:
        for key, value in vars(initial).items():
            setattr(initial, key, value[::2])
    old, new = copy.deepcopy(initial), copy.deepcopy(initial)
    # deepcopy preserves a noncontiguous view's values, not necessarily stride;
    # explicitly exercise the strided path using interleaved backing arrays.
    if strided:
        for target in (old, new):
            for key, value in vars(target).items():
                backing = torch.zeros(len(value) * 2, dtype=value.dtype)
                backing[::2] = value
                setattr(target, key, backing[::2])
    if count:
        old.syn_resource[0] = new.syn_resource[0] = .995
        old.syn_age[0] = new.syn_age[0] = torch.iinfo(torch.int32).max
    for _ in range(30):
        reference(old, count)
        inplace_candidate(new, count)
        assert_same(old, new)
    for key, value in vars(initial).items():
        assert torch.equal(getattr(new, key)[count:], value[count:])
        if strided:
            assert torch.count_nonzero(getattr(new, key)._base[1::2]) == 0


def test_overlapping_public_arrays_keep_the_original_operation_order():
    initial = make_state(20)
    initial.syn_facilitation = initial.syn_resource
    initial.syn_recent = initial.syn_resource
    old, new = copy.deepcopy(initial), copy.deepcopy(initial)
    for _ in range(30):
        reference(old, 20)
        inplace_candidate(new, 20)
        assert_same(old, new)


def test_integer_resource_fallback_keeps_assignment_casting():
    initial = make_state(20)
    initial.syn_resource = torch.arange(27, dtype=torch.int32)
    old, new = copy.deepcopy(initial), copy.deepcopy(initial)
    reference(old, 20)
    inplace_candidate(new, 20)
    assert_same(old, new)


def test_nonfinite_values_have_the_same_bit_patterns():
    initial = make_state(20)
    initial.syn_resource[:4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.])
    old, new = copy.deepcopy(initial), copy.deepcopy(initial)
    reference(old, 20)
    inplace_candidate(new, 20)
    assert_same(old, new)


@pytest.mark.parametrize("attribute", ["syn_resource", "syn_facilitation", "syn_recent"])
def test_requires_grad_state_uses_original_graph_operations(attribute):
    leaves, outputs = [], []
    for function in (reference, inplace_candidate):
        model = make_state(20)
        leaf = getattr(model, attribute).clone().requires_grad_()
        setattr(model, attribute, leaf + 0.)
        function(model, 20)
        getattr(model, attribute).sum().backward()
        leaves.append(leaf.grad)
        outputs.append(getattr(model, attribute).detach())
    assert torch.equal(leaves[0], leaves[1])
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("recovery,decay", [(0., 1.), (.02, .9), (-.001, 1.01)])
def test_nondefault_rates_preserve_the_original_update(recovery, decay):
    initial = make_state(20)
    old, new = copy.deepcopy(initial), copy.deepcopy(initial)
    for _ in range(10):
        reference(old, 20, recovery, decay)
        inplace_candidate(new, 20, recovery, decay)
        assert_same(old, new)
