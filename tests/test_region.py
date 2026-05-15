"""Tests for Region: neuron population, synapse management, simulation step."""

import torch
import pytest

from src.region import Region, RegionType, MAX_DELAY_STEPS
from src.neuron import NeuronType, FiringPattern, PATTERN_PARAMS
from src.synapse import NeurotransmitterType


class TestPopulation:
    def test_populate_creates_neurons(self, region):
        assert region.n_neurons == 20
        assert region.n_alive_neurons == 20

    def test_exc_inh_ratio(self, region):
        types = region.neuron_type[:region.n_neurons]
        n_exc = int((types == NeuronType.EXCITATORY.value).sum())
        n_inh = int((types == NeuronType.INHIBITORY.value).sum())
        assert n_exc + n_inh == 20
        assert n_exc > n_inh  # 80/20 ratio on average

    def test_neuron_alive_flags(self, region):
        assert torch.all(region.neuron_alive[:region.n_neurons])
        assert not torch.any(region.neuron_alive[region.n_neurons:])

    def test_initial_membrane_potential(self, region):
        assert torch.allclose(region.v[:region.n_neurons], torch.full((20,), -65.0), atol=1.0)

    def test_populate_respects_max_neurons(self):
        r = Region("small", RegionType.SENSORY, max_neurons=5)
        r.populate(100, connectivity=0.0)
        assert r.n_neurons == 5

    def test_add_neuron_returns_index(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        idx = r.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
        assert idx == 0
        idx2 = r.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
        assert idx2 == 1
        assert r.n_neurons == 2

    def test_add_neuron_at_capacity(self):
        r = Region("test", RegionType.SENSORY, max_neurons=1)
        r.add_neuron()
        idx = r.add_neuron()
        assert idx == -1

    def test_izhikevich_params_match_pattern(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.add_neuron(NeuronType.EXCITATORY, FiringPattern.FAST_SPIKING)
        a, b, c, d = PATTERN_PARAMS[FiringPattern.FAST_SPIKING]
        assert r.a[0].item() == pytest.approx(a)
        assert r.b[0].item() == pytest.approx(b)


class TestSynapses:
    def test_populate_creates_synapses(self, region):
        assert region.n_synapses > 0
        assert region.n_alive_synapses > 0

    def test_add_one_synapse(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=1.5, delay_ms=3.0, nt=NeurotransmitterType.GLUTAMATE)
        assert r.n_synapses == 1
        assert r.syn_pre[0] == 0
        assert r.syn_post[0] == 1
        assert r.syn_weight[0].item() == pytest.approx(1.5)
        assert r.syn_alive[0].item() is True

    def test_inhibitory_synapse_weight_negative(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=2.0, nt=NeurotransmitterType.GABA)
        assert r.syn_weight[0].item() < 0

    def test_synapse_capacity_doubling(self):
        r = Region("test", RegionType.SENSORY, max_neurons=100)
        r.populate(50, connectivity=0.0)
        initial_cap = r._syn_capacity
        n = initial_cap + 10
        pre = torch.zeros(n, dtype=torch.int32)
        post = torch.ones(n, dtype=torch.int32)
        weights = torch.ones(n, dtype=torch.float32)
        delays = torch.full((n,), 2.0, dtype=torch.float32)
        nt_vals = torch.full((n,), NeurotransmitterType.GLUTAMATE.value, dtype=torch.int32)
        r.add_synapses(pre, post, weights, delays, nt_vals)
        assert r._syn_capacity >= n
        assert r.n_synapses == n

    def test_add_lateral_inhibition(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(4, connectivity=0.0)
        count = r.add_lateral_inhibition(weight=-5.0, neuron_indices=[0, 1, 2])
        assert count == 6  # 3 * (3-1) = 6 all-to-all pairs
        assert r.n_synapses == 6
        assert torch.all(r.syn_weight[:6] < 0)

    def test_delay_clamped_to_valid_range(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(3, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=1.0, delay_ms=0.01)  # very small
        r.add_one_synapse(0, 2, weight=1.0, delay_ms=999.0)  # very large
        assert r.syn_delay[0].item() >= 1
        assert r.syn_delay[1].item() <= MAX_DELAY_STEPS - 1


class TestStep:
    def test_step_returns_tensor(self, region):
        fired = region.step(1.0, 1)
        assert isinstance(fired, torch.Tensor)
        assert fired.dtype == torch.int32 or fired.dtype == torch.int64

    def test_step_with_strong_current_produces_spikes(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        # Izhikevich neurons need sustained current over multiple steps to reach threshold
        for t in range(10):
            r.current[:5] = 40.0
            fired = r.step(float(t), t)
            if len(fired) > 0:
                break
        assert len(fired) > 0

    def test_fired_neurons_get_reset(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.current[:5] = 40.0
        r.step(1.0, 1)
        # Fired neurons should have v reset to c (around -65)
        assert torch.all(r.v[:5] < 30.0)

    def test_spike_tracking_increments(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.current[:5] = 40.0
        fired = r.step(1.0, 1)
        for idx in fired:
            assert r.total_spikes[idx].item() >= 1
            assert r.last_spike_time[idx].item() == 1.0

    def test_current_cleared_after_step(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        r.current[:5] = 10.0
        r.step(1.0, 1)
        assert torch.all(r.current[:5] == 0.0)

    def test_ring_buffer_delivers_delayed_current(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(3, connectivity=0.0)
        # Manually deposit into ring buffer at slot for step 5
        slot = 5 % MAX_DELAY_STEPS
        r.spike_buffer[slot, 0] = 20.0
        # Advance to step 5 — should read from that slot
        fired = r.step(5.0, 5)
        # Buffer should be cleared after read
        assert r.spike_buffer[slot, 0].item() == 0.0

    def test_empty_region_step(self):
        r = Region("empty", RegionType.SENSORY, max_neurons=10)
        fired = r.step(1.0, 1)
        assert len(fired) == 0

    def test_activity_trace_increases_on_spike(self):
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(5, connectivity=0.0)
        assert r.activity[0].item() == 0.0
        r.current[:5] = 40.0
        r.step(1.0, 1)
        fired = r.fired[:5]
        for i in range(5):
            if fired[i]:
                assert r.activity[i].item() > 0.0

    def test_spike_propagation_through_synapse(self):
        """A spike from neuron 0 should eventually deliver current to neuron 1 via the ring buffer."""
        r = Region("test", RegionType.SENSORY, max_neurons=10)
        r.populate(3, connectivity=0.0)
        r.add_one_synapse(0, 1, weight=5.0, delay_ms=1.0, nt=NeurotransmitterType.GLUTAMATE)

        # Drive neuron 0 until it fires, then check all buffer slots
        neuron0_fired = False
        for t in range(20):
            r.current[0] = 50.0
            r.step(float(t), t)
            if r.fired[0].item():
                neuron0_fired = True
                # Check the entire spike buffer for neuron 1 across all delay slots
                buffer_for_n1 = r.spike_buffer[:, 1].sum().item()
                if buffer_for_n1 > 0:
                    break
        assert neuron0_fired, "Neuron 0 never fired"
        # After the fire step, neuron 1 should eventually receive the current
        # via the ring buffer (may take delay_steps more iterations)
        received = False
        for t2 in range(t + 1, t + MAX_DELAY_STEPS + 1):
            pre_current = r.spike_buffer[t2 % MAX_DELAY_STEPS, 1].item()
            if pre_current > 0:
                received = True
                break
            r.step(float(t2), t2)
        assert received, "Neuron 1 never received synaptic current from neuron 0"


class TestProperties:
    def test_mean_activity_zero_initially(self, region):
        assert region.mean_activity == 0.0

    def test_n_alive_neurons(self, region):
        assert region.n_alive_neurons == 20

    def test_n_alive_synapses(self, region):
        assert region.n_alive_synapses == region.n_synapses

    def test_repr(self, region):
        s = repr(region)
        assert "test" in s
        assert "sensory" in s
