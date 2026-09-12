"""Empty synaptic paths must still advance neurons, delays and live synapse state."""

import copy
from unittest.mock import patch

import pytest
import torch

from examples.mnist_optimization_check import simulation_digest
from examples.region_step_reference import dense_step
from src.brain import Brain
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


@pytest.mark.parametrize("mode", ["heun", "legacy_euler"])
@pytest.mark.parametrize("connections", [0, 4, 3000])
@pytest.mark.parametrize("drive", [0.0, 25.0, -200.0])
def test_empty_and_nonempty_steps_match_reference(mode, connections, drive):
    brain = Brain(seed=11, integration_method=mode)
    region = brain.add_region("test", RegionType.SENSORY, 8, connectivity=0.0, max_neurons=10)
    if connections:
        region.add_synapses(
            torch.arange(connections, dtype=torch.int32) % 8,
            (torch.arange(connections, dtype=torch.int32) + 1) % 8,
            torch.ones(connections), torch.full((connections,), 2.0),
            torch.full((connections,), NeurotransmitterType.GLUTAMATE.value, dtype=torch.int32),
        )
        region.syn_alive[1::3] = False
        region.syn_resource[:connections] = 0.4
        region.syn_facilitation[:connections] = 0.2
        region.syn_recent[:connections] = 3.0
    region.neuron_alive[7] = False
    region.spike_buffer[3, 0] = 25.0
    region.spike_buffer[19, 1] = 30.0
    expected = copy.deepcopy(brain)
    for step in range(30):
        for model in (brain, expected):
            model.regions["test"].current[:8] += drive
        # Brain.step also checks homeostasis, projection delivery and RNG state.
        brain.step()
        with patch.object(Region, "step", dense_step):
            expected.step()
        assert simulation_digest(brain) == simulation_digest(expected), step


def test_quiet_region_still_checks_invalid_dead_presynaptic_endpoints():
    region = Region("test", RegionType.SENSORY, max_neurons=4)
    region.populate(4, connectivity=0.0)
    region.add_one_synapse(0, 1, 1.0)
    region.syn_pre[0], region.syn_alive[0] = 4, False
    error = IndexError if region.v.device.type == "cpu" else RuntimeError
    with pytest.raises(error, match="out of bounds"):
        region.step(1.0, 1)


def test_no_neurons_matches_reference_without_consuming_buffer():
    region = Region("empty", RegionType.SENSORY, max_neurons=4)
    region.spike_buffer[1, 0] = 12.0
    reference = copy.deepcopy(region)
    assert torch.equal(region.step(1.0, 1), dense_step(reference, 1.0, 1))
    assert torch.equal(region.spike_buffer, reference.spike_buffer)
