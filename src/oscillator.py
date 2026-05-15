"""
Brain oscillations: rhythmic neural activity at specific frequency bands.

Each brain region can have oscillators at different frequencies that
inject background current modulation and influence plasticity.

Frequency bands:
- Delta  (0.5-4 Hz):  deep sleep
- Theta  (4-8 Hz):    memory encoding, spatial navigation (hippocampus)
- Alpha  (8-13 Hz):   relaxed wakefulness, attentional suppression
- Beta   (13-30 Hz):  motor control, active thinking
- Gamma  (30-100 Hz): perceptual binding, focused attention

Cross-frequency coupling (theta-gamma) is implemented: gamma oscillations
are nested within theta cycles, with gamma amplitude modulated by theta
phase. This is critical for multi-item working memory.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field


class FrequencyBand(enum.Enum):
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


# Default center frequencies (Hz) for each band
BAND_FREQUENCIES: dict[FrequencyBand, float] = {
    FrequencyBand.DELTA: 2.0,
    FrequencyBand.THETA: 6.0,
    FrequencyBand.ALPHA: 10.0,
    FrequencyBand.BETA: 20.0,
    FrequencyBand.GAMMA: 40.0,
}

# Default region → dominant oscillation band
from .region import RegionType  # noqa: E402

REGION_DEFAULT_BANDS: dict[RegionType, list[FrequencyBand]] = {
    RegionType.SENSORY:     [FrequencyBand.GAMMA],
    RegionType.ASSOCIATION: [FrequencyBand.BETA, FrequencyBand.GAMMA],
    RegionType.MEMORY:      [FrequencyBand.THETA, FrequencyBand.GAMMA],
    RegionType.MOTOR:       [FrequencyBand.BETA],
    RegionType.REWARD:      [FrequencyBand.ALPHA],
}


@dataclass
class Oscillator:
    """A single oscillator at a specific frequency."""
    frequency: float          # Hz
    amplitude: float = 1.0    # current injection amplitude
    phase: float = 0.0        # current phase (radians)
    band: FrequencyBand = FrequencyBand.THETA

    # Phase coupling to another oscillator (for cross-frequency coupling)
    coupled_to: Oscillator | None = field(default=None, repr=False)
    coupling_strength: float = 0.0

    def step(self, dt_ms: float) -> float:
        """
        Advance phase by one timestep.
        Returns the current oscillatory value in [-amplitude, +amplitude].
        """
        dt_s = dt_ms / 1000.0
        self.phase += 2.0 * math.pi * self.frequency * dt_s

        # Phase coupling: pull phase toward coupled oscillator
        if self.coupled_to is not None and self.coupling_strength > 0:
            phase_diff = self.coupled_to.phase - self.phase
            self.phase += self.coupling_strength * math.sin(phase_diff) * dt_s

        # Keep phase in [0, 2π)
        self.phase = self.phase % (2.0 * math.pi)

        return self.amplitude * math.sin(self.phase)

    @property
    def current_phase_normalized(self) -> float:
        """Phase normalized to [0, 1] — 0=trough, 0.25=rising, 0.5=peak, 0.75=falling."""
        return self.phase / (2.0 * math.pi)


class OscillatorBank:
    """
    Manages oscillators for all brain regions.

    Each region gets oscillators based on its type.
    Theta-gamma coupling is automatically set up for MEMORY regions.
    """

    def __init__(
        self,
        base_amplitude: float = 0.5,
        gamma_theta_coupling: float = 5.0,
    ):
        self.base_amplitude = base_amplitude
        self.gamma_theta_coupling = gamma_theta_coupling
        self.oscillators: dict[str, list[Oscillator]] = {}

    def add_region(self, region_name: str, region_type: RegionType) -> None:
        """Create oscillators for a region based on its type."""
        bands = REGION_DEFAULT_BANDS.get(region_type, [FrequencyBand.ALPHA])
        oscs = []

        theta_osc = None
        for band in bands:
            freq = BAND_FREQUENCIES[band]
            # Gamma is faster but weaker; theta is stronger
            amp = self.base_amplitude * (0.5 if band == FrequencyBand.GAMMA else 1.0)
            osc = Oscillator(
                frequency=freq,
                amplitude=amp,
                phase=random.uniform(0, 2 * math.pi),
                band=band,
            )
            oscs.append(osc)
            if band == FrequencyBand.THETA:
                theta_osc = osc

        # Set up theta-gamma coupling: gamma amplitude modulated by theta phase
        if theta_osc is not None:
            for osc in oscs:
                if osc.band == FrequencyBand.GAMMA:
                    osc.coupled_to = theta_osc
                    osc.coupling_strength = self.gamma_theta_coupling

        self.oscillators[region_name] = oscs

    def step(self, dt_ms: float) -> dict[str, float]:
        """
        Advance all oscillators and return per-region background current.
        """
        currents = {}
        for name, oscs in self.oscillators.items():
            total = 0.0
            for osc in oscs:
                total += osc.step(dt_ms)
            currents[name] = total
        return currents

    def get_theta_phase(self, region_name: str) -> float | None:
        """
        Get the current theta phase for a region (if it has a theta oscillator).
        Used to modulate STDP: learning is stronger at theta peak.
        """
        oscs = self.oscillators.get(region_name, [])
        for osc in oscs:
            if osc.band == FrequencyBand.THETA:
                return osc.current_phase_normalized
        return None

    def theta_modulation(self, region_name: str) -> float:
        """
        Returns a value [0, 1] indicating how favorable the current
        theta phase is for learning. Peak theta → 1.0, trough → 0.0.
        Regions without theta return 1.0 (no modulation).
        """
        phase = self.get_theta_phase(region_name)
        if phase is None:
            return 1.0
        # Peak at phase=0.5 (sin=1), trough at phase=0 (sin=0)
        return 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
