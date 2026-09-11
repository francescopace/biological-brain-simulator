"""Tests for save/load round-trip."""

from pathlib import Path

import torch
import pytest

import src.persistence as persistence
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
    def test_refuses_to_delete_unrelated_directory(self, trained_brain, tmp_path):
        path = tmp_path / "important"
        path.mkdir()
        sentinel = path / "keep.txt"
        sentinel.write_text("do not delete")

        with pytest.raises(ValueError, match="not a brain save directory"):
            save_brain(trained_brain, path)

        assert sentinel.read_text() == "do not delete"

    def test_save_rejects_transient_reward_restriction_before_writing(self, tmp_path):
        brain = Brain(seed=42)
        path = tmp_path / "not_created" / "brain_save"
        with brain.reward_stdp.restrict_to_posts("cortex", [0]):
            brain.reward(1.0, "cortex")
            with pytest.raises(ValueError, match="active reward restriction"):
                save_brain(brain, path)
        assert not path.parent.exists()
        save_brain(brain, path)
        assert load_brain(path).reward_stdp._post_restrictions == {}

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

    def test_round_trip_preserves_selective_plasticity(self, trained_brain, tmp_path):
        trained_brain.freeze_plasticity()
        trained_brain.enable_projection_plasticity(
            "input", "cortex", A_plus=0.04, A_minus=0.05,
        )
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert all(not region.plasticity_enabled for region in loaded.regions.values())
        assert loaded.get_projection("input", "cortex").plasticity_enabled is True
        assert loaded.get_projection("cortex", "motor").plasticity_enabled is False
        active = loaded.get_projection("input", "cortex")
        assert torch.all(active.syn_A_plus[:active.n_synapses] == 0.04)
        assert torch.all(active.syn_A_minus[:active.n_synapses] == 0.05)
        # Synapses created after loading inherit the region's frozen state.
        loaded.regions["cortex"].add_one_synapse(0, 1)
        loaded.freeze_homeostatic_scaling()
        loaded.step()
        for region in loaded.regions.values():
            assert torch.count_nonzero(region.syn_A_plus[:region.n_synapses]) == 0
            assert torch.count_nonzero(region.syn_A_minus[:region.n_synapses]) == 0

    def test_legacy_checkpoints_default_to_enabled_plasticity(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        for subdirectory in ("regions", "projections"):
            for checkpoint in (path / subdirectory).glob("*.pt"):
                state = torch.load(checkpoint, weights_only=True)
                state.pop("plasticity_enabled")
                torch.save(state, checkpoint)

        loaded = load_brain(path)
        assert all(region.plasticity_enabled for region in loaded.regions.values())
        assert all(projection.plasticity_enabled for projection in loaded.projections)

    def test_round_trip_preserves_memory(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert len(loaded.memory.traces) == len(trained_brain.memory.traces)
        assert loaded.memory.consolidation_interval == trained_brain.memory.consolidation_interval

    def test_round_trip_preserves_subsystems(self, trained_brain, tmp_path):
        trained_brain.freeze_homeostatic_scaling()
        trained_brain.freeze_adaptive_thresholds()
        trained_brain.disable_memory()
        trained_brain.disable_oscillations()
        trained_brain.disable_reward_modulated_plasticity()
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert loaded.homeostasis.theta_plus == trained_brain.homeostasis.theta_plus
        assert loaded.homeostasis.theta_leak == trained_brain.homeostasis.theta_leak
        assert loaded.homeostasis.activity_decay == trained_brain.homeostasis.activity_decay
        assert loaded.homeostasis.scaling_enabled is False
        assert loaded.homeostasis.theta_enabled is False
        assert loaded.memory.enabled is False
        assert loaded.oscillators.enabled is False
        assert loaded.reward_stdp.enabled is False
        assert loaded.growth.growth_interval == trained_brain.growth.growth_interval
        assert loaded.encoder.strategy == trained_brain.encoder.strategy

    def test_round_trip_preserves_seed_flags_and_morphology(self, trained_brain, tmp_path):
        trained_brain.freeze_structural_plasticity()
        trained_brain.regions["cortex"].enable_morphology()
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        assert loaded.seed == trained_brain.seed
        assert loaded._region_seed_counter == trained_brain._region_seed_counter
        assert loaded.metaplasticity_enabled is False
        assert loaded.regions["cortex"].morphology_manager is not None

    def test_round_trip_continuation_is_equivalent(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        for _ in range(10):
            trained_brain.stimulate("input", [0.3] * 10)
            loaded.stimulate("input", [0.3] * 10)
            trained_brain.step()
            loaded.step()

        for name in trained_brain.regions:
            original = trained_brain.regions[name]
            restored = loaded.regions[name]
            n = original.n_neurons
            assert torch.equal(restored.v[:n], original.v[:n])
            assert torch.equal(restored.u[:n], original.u[:n])

    def test_round_trip_preserves_future_region_rng(self, trained_brain, tmp_path):
        path = tmp_path / "brain_save"
        save_brain(trained_brain, path)
        loaded = load_brain(path)

        original_region = trained_brain.add_region(
            "future", RegionType.MEMORY, n_neurons=8, connectivity=0.2,
        )
        loaded_region = loaded.add_region(
            "future", RegionType.MEMORY, n_neurons=8, connectivity=0.2,
        )

        assert torch.equal(loaded_region.a[:8], original_region.a[:8])
        assert torch.equal(
            loaded_region.syn_weight[:loaded_region.n_synapses],
            original_region.syn_weight[:original_region.n_synapses],
        )
        assert [o.phase for o in loaded.oscillators.oscillators["future"]] == [
            o.phase for o in trained_brain.oscillators.oscillators["future"]
        ]

    def test_region_names_are_not_used_as_paths(self, tmp_path):
        brain = Brain(seed=42)
        brain.add_region("../outside", RegionType.SENSORY, n_neurons=2)
        path = tmp_path / "brain_save"
        save_brain(brain, path)
        loaded = load_brain(path)
        assert "../outside" in loaded.regions

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

    @pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
    @pytest.mark.parametrize("rename_number", [1, 2])
    @pytest.mark.parametrize("after_move", [False, True])
    def test_failed_replacement_leaves_a_loadable_checkpoint(
        self, tmp_path, monkeypatch, error_type, rename_number, after_move,
    ):
        brain = Brain(seed=42)
        path = tmp_path / "brain_save"
        save_brain(brain, path)
        brain.step_count = 7
        original_replace = persistence.os.replace
        calls = 0

        def interrupted_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == rename_number:
                if after_move:
                    original_replace(source, destination)
                raise error_type("injected replacement failure")
            return original_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", interrupted_replace)
        with pytest.raises(error_type, match="injected replacement failure"):
            save_brain(brain, path)

        # Once the second move has completed the new complete checkpoint is
        # available; before that point the previous checkpoint is restored.
        expected_steps = 7 if rename_number == 2 and after_move else 0
        assert load_brain(path).step_count == expected_steps
        assert not list(tmp_path.glob(".brain_save.tmp-*"))

    def test_failed_rollback_preserves_backup(self, tmp_path, monkeypatch):
        brain = Brain(seed=42)
        path = tmp_path / "brain_save"
        save_brain(brain, path)
        original_replace = persistence.os.replace

        def fail_install_and_rollback(source, destination):
            if Path(destination) == path:
                raise OSError("destination unavailable")
            return original_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", fail_install_and_rollback)
        with pytest.raises(OSError, match="destination unavailable"):
            save_brain(brain, path)

        backups = list(tmp_path.glob(".brain_save.backup-*"))
        assert len(backups) == 1
        assert load_brain(backups[0]).step_count == 0
