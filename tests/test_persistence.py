"""Tests for save/load round-trip."""

import torch
import pytest

from src.brain import Brain
from src.region import RegionType
from src.persistence import save_brain, load_brain


@pytest.fixture
def trained_brain(tmp_path):
    """A brain that has been run for a few steps with stimulation."""
    b = Brain(seed=42)
    b.add_region("input", RegionType.SENSORY, n_neurons=10, connectivity=0.0)
    b.add_region("cortex", RegionType.ASSOCIATION, n_neurons=20, connectivity=0.1)
    b.add_region("motor", RegionType.MOTOR, n_neurons=5, connectivity=0.1)
    b.connect_regions("input", "cortex", density=0.3)
    b.connect_regions("cortex", "motor", density=0.3)

    for i in range(50):
        b.stimulate("input", [0.5] * 10)
        b.step()
        if i == 25:
            b.reward(1.0, Brain.projection_target("input", "cortex"))

    return b


class TestSaveLoad:
    def test_round_trip_preserves_structure(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert set(loaded.regions.keys()) == set(trained_brain.regions.keys())
        assert len(loaded.projections) == len(trained_brain.projections)

    def test_round_trip_preserves_time(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert loaded.time == pytest.approx(trained_brain.time)
        assert loaded.step_count == trained_brain.step_count

    def test_round_trip_preserves_neuron_state(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        for name in trained_brain.regions:
            orig = trained_brain.regions[name]
            rest = loaded.regions[name]
            n = orig.n_neurons
            assert rest.n_neurons == n
            assert torch.allclose(rest.v[:n], orig.v[:n])
            assert torch.allclose(rest.u[:n], orig.u[:n])
            assert torch.equal(rest.neuron_alive[:n], orig.neuron_alive[:n])
            assert torch.equal(rest.total_spikes[:n], orig.total_spikes[:n])

    def test_round_trip_preserves_synapse_state(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        for name in trained_brain.regions:
            orig = trained_brain.regions[name]
            rest = loaded.regions[name]
            ns = orig.n_synapses
            assert rest.n_synapses == ns
            if ns > 0:
                assert torch.allclose(rest.syn_weight[:ns], orig.syn_weight[:ns])
                assert torch.equal(rest.syn_pre[:ns], orig.syn_pre[:ns])
                assert torch.equal(rest.syn_post[:ns], orig.syn_post[:ns])

    def test_round_trip_preserves_projections(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        for orig_proj, rest_proj in zip(trained_brain.projections, loaded.projections):
            assert rest_proj.source_name == orig_proj.source_name
            assert rest_proj.target_name == orig_proj.target_name
            ns = orig_proj.n_synapses
            assert rest_proj.n_synapses == ns
            if ns > 0:
                assert torch.allclose(rest_proj.syn_weight[:ns], orig_proj.syn_weight[:ns])

    def test_round_trip_preserves_memory(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert len(loaded.memory.traces) == len(trained_brain.memory.traces)
        assert loaded.memory.consolidation_interval == trained_brain.memory.consolidation_interval

    def test_round_trip_preserves_subsystems(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert loaded.homeostasis.theta_plus == trained_brain.homeostasis.theta_plus
        assert loaded.homeostasis.theta_leak == trained_brain.homeostasis.theta_leak
        assert loaded.growth.growth_interval == trained_brain.growth.growth_interval
        assert loaded.encoder.strategy == trained_brain.encoder.strategy

    def test_loaded_brain_can_continue_running(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        # Should be able to keep stepping without errors
        for _ in range(20):
            loaded.stimulate("input", [0.3] * 10)
            loaded.step()
        assert loaded.step_count == trained_brain.step_count + 20

    def test_overwrite_existing_save(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        save_brain(trained_brain, path)  # should not raise
        loaded = load_brain(path)
        assert loaded.step_count == trained_brain.step_count
