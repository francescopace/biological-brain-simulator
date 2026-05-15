"""Tests for oscillator bank: frequency, phase, coupling."""

import math
import pytest

from src.oscillator import Oscillator, OscillatorBank, FrequencyBand, BAND_FREQUENCIES
from src.region import RegionType


class TestOscillator:
    def test_step_advances_phase(self):
        osc = Oscillator(frequency=10.0, amplitude=1.0, phase=0.0)
        osc.step(1.0)  # 1 ms
        assert osc.phase > 0.0

    def test_output_bounded_by_amplitude(self):
        osc = Oscillator(frequency=10.0, amplitude=2.5, phase=0.0)
        values = [osc.step(1.0) for _ in range(1000)]
        assert max(values) <= 2.5 + 1e-6
        assert min(values) >= -2.5 - 1e-6

    def test_phase_wraps_around(self):
        osc = Oscillator(frequency=1000.0, amplitude=1.0, phase=0.0)
        for _ in range(100):
            osc.step(1.0)
        assert 0 <= osc.phase < 2 * math.pi

    def test_normalized_phase_in_0_1(self):
        osc = Oscillator(frequency=10.0, amplitude=1.0)
        for _ in range(50):
            osc.step(1.0)
        assert 0.0 <= osc.current_phase_normalized <= 1.0

    def test_coupling_affects_phase(self):
        theta = Oscillator(frequency=6.0, amplitude=1.0, phase=0.0)
        gamma = Oscillator(frequency=40.0, amplitude=0.5, phase=math.pi,
                          coupled_to=theta, coupling_strength=5.0)
        uncoupled = Oscillator(frequency=40.0, amplitude=0.5, phase=math.pi)

        for _ in range(100):
            theta.step(1.0)
            gamma.step(1.0)
            uncoupled.step(1.0)

        # Coupled gamma phase should differ from uncoupled
        assert abs(gamma.phase - uncoupled.phase) > 0.01


class TestOscillatorBank:
    def test_add_region_creates_oscillators(self):
        bank = OscillatorBank()
        bank.add_region("visual", RegionType.SENSORY)
        assert "visual" in bank.oscillators
        assert len(bank.oscillators["visual"]) > 0

    def test_step_returns_currents(self):
        bank = OscillatorBank()
        bank.add_region("visual", RegionType.SENSORY)
        bank.add_region("memory", RegionType.MEMORY)
        currents = bank.step(1.0)
        assert "visual" in currents
        assert "memory" in currents
        assert isinstance(currents["visual"], float)

    def test_memory_region_has_theta_gamma(self):
        bank = OscillatorBank()
        bank.add_region("hippo", RegionType.MEMORY)
        bands = {osc.band for osc in bank.oscillators["hippo"]}
        assert FrequencyBand.THETA in bands
        assert FrequencyBand.GAMMA in bands

    def test_theta_gamma_coupling_set(self):
        bank = OscillatorBank()
        bank.add_region("hippo", RegionType.MEMORY)
        gamma_oscs = [o for o in bank.oscillators["hippo"] if o.band == FrequencyBand.GAMMA]
        assert len(gamma_oscs) > 0
        assert gamma_oscs[0].coupled_to is not None
        assert gamma_oscs[0].coupling_strength > 0

    def test_theta_phase_available(self):
        bank = OscillatorBank()
        bank.add_region("hippo", RegionType.MEMORY)
        bank.step(1.0)
        phase = bank.get_theta_phase("hippo")
        assert phase is not None
        assert 0.0 <= phase <= 1.0

    def test_no_theta_returns_none(self):
        bank = OscillatorBank()
        bank.add_region("motor", RegionType.MOTOR)
        assert bank.get_theta_phase("motor") is None

    def test_theta_modulation_range(self):
        bank = OscillatorBank()
        bank.add_region("hippo", RegionType.MEMORY)
        values = []
        for _ in range(1000):
            bank.step(1.0)
            values.append(bank.theta_modulation("hippo"))
        assert min(values) >= 0.0
        assert max(values) <= 1.0

    def test_no_theta_modulation_returns_1(self):
        bank = OscillatorBank()
        bank.add_region("motor", RegionType.MOTOR)
        assert bank.theta_modulation("motor") == 1.0
