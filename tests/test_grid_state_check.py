"""Grid interventions isolate saved weights, motor thresholds and tie handling."""

import numpy as np
import pytest
import torch

import examples.grid_nav_benchmark as grid
import examples.grid_state_check as check
from examples.mnist_optimization_check import simulation_digest


@pytest.mark.parametrize("case", check.CASES)
def test_interventions_do_not_mutate_donor_or_unrelated_state(case):
    brain = grid.build_brain()
    brain.get_projection("input", "motor").syn_weight[:100] = .2
    brain.regions["motor"].theta[:4] = torch.tensor([.1, .2, .3, .4], device=grid.DEVICE)
    before = simulation_digest(brain)
    model = check.variant(brain, case)
    assert simulation_digest(brain) == before
    weights = model.get_projection("input", "motor").syn_weight[:100]
    expected = .05 if case == "initial_readout" else 3.2 if "gain16" in case else .2
    torch.testing.assert_close(weights, torch.full_like(weights, expected))
    if case == "equal_motor_theta" or "equal_theta" in case:
        torch.testing.assert_close(model.regions["motor"].theta[:4], torch.full_like(model.regions["motor"].theta[:4], .25))
    # Put only the intervened tensors back and compare the complete state.
    model.get_projection("input", "motor").syn_weight.copy_(brain.get_projection("input", "motor").syn_weight)
    modulation = model.get_projection("input", "motor").syn_modulation
    original_modulation = brain.get_projection("input", "motor").syn_modulation
    torch.testing.assert_close(modulation[:100], original_modulation[:100] * check.TRANSMISSION_CASES.get(case, 1.))
    modulation.copy_(original_modulation)
    model.regions["motor"].theta.copy_(brain.regions["motor"].theta)
    assert simulation_digest(model) == before


def test_gain_rejects_out_of_bounds_instead_of_silently_clipping():
    brain = grid.build_brain()
    brain.get_projection("input", "motor").syn_weight[:100] = 1.
    before = simulation_digest(brain)
    with pytest.raises(ValueError, match="bounds"):
        check.variant(brain, "readout_gain16")
    assert simulation_digest(brain) == before


@pytest.mark.parametrize("counts, voltage, epsilon", [([2, 2, 1, 0], [0, 1, 10, 20], 0.),
    ([0, 3, 1, 0], [20, 0, 0, 0], 0.), ([0, 0, 0, 0], [0, 1, 2, 3], 0.),
    ([2, 2, 1, 0], [0, 1, 10, 20], 1.)])
def test_voltage_tie_policy_keeps_exploration_rng_and_non_tied_choices(counts, voltage, epsilon):
    counts, voltage = np.array(counts), np.array(voltage)
    rng1, rng2 = np.random.default_rng(8), np.random.default_rng(8)
    original = grid.choose_action(counts, voltage, epsilon, rng1)
    changed = check.voltage_tie_policy(grid.choose_action)(counts, voltage, epsilon, rng2)
    assert rng1.bit_generator.state == rng2.bit_generator.state
    if original[1] == "policy" and sum(counts == counts.max()) > 1:
        assert changed == (1, "policy")
    else:
        assert changed == original
