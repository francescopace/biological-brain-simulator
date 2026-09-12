"""CPU inputs and diagnostic snapshots must work with the selected backend."""

import torch

from examples.mnist_learning_check import weight_delta_metrics, weight_summary
from src.brain import Brain
from src.growth import GrowthController
from src.memory import MemoryTrace
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


def test_batch_synapses_accept_cpu_float64_inputs_without_mutating_them():
    region = Region("test", RegionType.SENSORY, max_neurons=2)
    region.populate(2, connectivity=0.0)
    inputs = (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1, 0], dtype=torch.int64),
        torch.tensor([1.25, 2.5], dtype=torch.float64),
        torch.tensor([1.25, 2.75], dtype=torch.float64),
        torch.tensor([NeurotransmitterType.GLUTAMATE.value,
                      NeurotransmitterType.GABA.value], dtype=torch.int64),
    )
    original = tuple(value.clone() for value in inputs)
    region.add_synapses(*inputs)

    assert region.n_synapses == 2
    assert region.syn_weight[:2].tolist() == [1.25, -2.5]
    assert region.syn_delay[:2].tolist() == [1, 3]
    assert region.syn_pre[:2].tolist() == [0, 1]
    assert region.syn_post[:2].tolist() == [1, 0]
    for current, before in zip(inputs, original):
        assert current.device.type == "cpu"
        assert current.dtype == before.dtype
        assert torch.equal(current, before)


def test_cpu_trace_refresh_replay_and_apoptosis_preserve_aligned_activity():
    brain = Brain(seed=7)
    region = brain.add_region("test", RegionType.SENSORY, 4, connectivity=0.0, max_neurons=4)
    trace = MemoryTrace(torch.arange(4), torch.zeros(4), region.name)
    brain.memory.traces = [trace]
    region.activity[:4] = torch.tensor([0.2, 0.3, 0.4, 0.5], device=region.activity.device)

    assert brain.memory.capture_trace(region, 1.0) is trace
    assert len(brain.memory.traces) == 1
    assert trace.neuron_indices.device.type == "cpu"
    assert trace.activity_snapshot.device == region.activity.device
    assert torch.equal(trace.activity_snapshot, region.activity[:4])

    brain.memory.consolidation_interval = 1
    assert brain.memory.consolidate(brain.regions, 1.0) == 1
    assert torch.all(region.current[:4] == brain.memory.replay_strength)

    snapshot = trace.activity_snapshot.clone()
    region.neuron_age[0] = 20_000
    region.activity[0] = 0.0
    assert GrowthController()._apoptosis(region, memory=brain.memory) == 1
    assert brain.memory.traces == [trace]
    assert trace.neuron_indices.tolist() == [1, 2, 3]
    assert torch.equal(trace.activity_snapshot, snapshot[1:])
    region.current.zero_()
    assert brain.memory.consolidate(brain.regions, 2.0) == 1
    assert region.current[0] == 0.0
    assert torch.all(region.current[1:4] == brain.memory.replay_strength)


def test_weight_diagnostics_match_cpu_float64_and_preserve_source_tensors():
    region = Region("test", RegionType.SENSORY, max_neurons=2)
    before = torch.tensor([0.1, 0.3, 0.7])
    after = torch.tensor([0.15, 0.25, 0.7])
    normalized = torch.tensor([0.12, 0.27, 0.72])
    values = tuple(value.to(region.v.device) for value in (before, after, normalized))
    snapshots = tuple(value.clone() for value in values)
    expected = weight_delta_metrics(before, after, normalized)
    assert weight_delta_metrics(*values) == expected
    for value, snapshot in zip(values, snapshots):
        assert torch.equal(value, snapshot)

    region.populate(2, connectivity=0.0)
    region.add_synapses(torch.tensor([0, 1]), torch.tensor([1, 0]), before[:2],
                        torch.ones(2), torch.full((2,), NeurotransmitterType.GLUTAMATE.value))
    weights = region.syn_weight[:2].clone()
    result = weight_summary(region)
    reference = weights.cpu().double()
    assert result["mean"] == float(reference.mean())
    assert result["std"] == float(reference.std(unbiased=False))
    assert torch.equal(region.syn_weight[:2], weights)
