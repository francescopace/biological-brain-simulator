"""Tests for growth controller: pruning, synaptogenesis, neurogenesis, apoptosis."""

import torch
import pytest

from src.growth import GrowthController, GrowthStats
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


class TestSynapsePruning:
    def test_old_weak_synapses_pruned(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(10, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=0.001, nt=NeurotransmitterType.GLUTAMATE)
        r.syn_age[0] = 5000
        r.syn_recent[0] = 0.0

        gc = GrowthController(growth_interval=1, synapse_prune_threshold=0.02, synapse_age_threshold=3000)
        gc.step([r])
        assert r.syn_alive[0].item() is False

    def test_strong_synapses_survive(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(10, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=5.0, nt=NeurotransmitterType.GLUTAMATE)
        r.syn_age[0] = 5000
        r.syn_recent[0] = 0.0

        gc = GrowthController(growth_interval=1, synapse_prune_threshold=0.02)
        gc.step([r])
        assert r.syn_alive[0].item() is True

    def test_young_synapses_survive(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(10, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=0.001, nt=NeurotransmitterType.GLUTAMATE)
        r.syn_age[0] = 100  # young

        gc = GrowthController(growth_interval=1, synapse_age_threshold=3000)
        gc.step([r])
        assert r.syn_alive[0].item() is True


class TestApoptosis:
    def test_inactive_old_neurons_die(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        r.neuron_age[0] = 20000
        r.activity[0] = 0.0
        r.total_spikes[0] = 0

        gc = GrowthController(growth_interval=1, apoptosis_age=10000, apoptosis_threshold=0.001)
        gc.step([r])
        assert r.neuron_alive[0].item() is False

    def test_active_neurons_survive(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        r.neuron_age[0] = 20000
        r.activity[0] = 0.5  # active
        r.total_spikes[0] = 100

        gc = GrowthController(growth_interval=1, apoptosis_age=10000)
        gc.step([r])
        assert r.neuron_alive[0].item() is True

    def test_apoptosis_kills_connected_synapses(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=1.0, nt=NeurotransmitterType.GLUTAMATE)
        r.neuron_age[0] = 20000
        r.activity[0] = 0.0
        r.total_spikes[0] = 0

        gc = GrowthController(growth_interval=1, apoptosis_age=10000, apoptosis_threshold=0.001)
        gc.step([r])
        assert r.syn_alive[0].item() is False


class TestSynaptogenesis:
    def test_coactive_neurons_get_connected(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        r.activity[0] = 0.3
        r.activity[1] = 0.3

        gc = GrowthController(growth_interval=1, max_new_synapses_per_cycle=5)
        gc._rng.manual_seed(42)
        gc.step([r])
        assert r.n_synapses > 0

    def test_no_synaptogenesis_when_inactive(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        # all activity at 0

        gc = GrowthController(growth_interval=1)
        gc.step([r])
        assert r.n_synapses == 0


class TestNeurogenesis:
    def test_high_activity_triggers_neurogenesis(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        r.activity[:5] = 0.5  # high activity everywhere

        gc = GrowthController(growth_interval=1, neurogenesis_threshold=0.3)
        gc._rng.manual_seed(42)
        gc.step([r])
        assert r.n_neurons > 5

    def test_no_neurogenesis_at_capacity(self):
        r = Region("test", RegionType.SENSORY, max_neurons=5)
        r.populate(5, connectivity=0.0)
        r.activity[:5] = 0.5

        gc = GrowthController(growth_interval=1, neurogenesis_threshold=0.3)
        gc.step([r])
        assert r.n_neurons == 5

    def test_low_activity_no_neurogenesis(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        # all activity at 0

        gc = GrowthController(growth_interval=1, neurogenesis_threshold=0.3)
        gc.step([r])
        assert r.n_neurons == 5


class TestGrowthInterval:
    def test_only_runs_at_interval(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        gc = GrowthController(growth_interval=10)
        for i in range(9):
            result = gc.step([r])
            assert result is None
        result = gc.step([r])
        assert result is not None

    def test_history_tracked(self):
        r = Region("test", RegionType.SENSORY, max_neurons=20)
        r.populate(5, connectivity=0.0)
        gc = GrowthController(growth_interval=1)
        gc.step([r])
        gc.step([r])
        assert len(gc.history) == 2
        assert isinstance(gc.history[0], GrowthStats)
