"""
Neuron morphology: multi-compartment model with cable equation.

Real neurons are not points — they have a dendritic tree (input),
a soma (cell body), and an axon (output). The position of a synapse
along the dendrite affects how much its current reaches the soma.

This module implements a simplified 3-compartment model:
- Distal dendrite: far from soma, input is attenuated
- Proximal dendrite: near soma, less attenuation
- Soma: integration point (Izhikevich dynamics)

The cable equation governs how voltage propagates between compartments:
    C * dV/dt = (V_parent - V) / R_axial + I_syn

Predefined morphology templates:
- Pyramidal: long apical dendrite (5 compartments) — cortical principal cells
- Interneuron: compact dendritic tree (2 compartments) — fast local inhibition
- Stellate: symmetric short dendrites (3 compartments) — sensory cortex
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import torch

from .device import DEVICE


class CompartmentType(enum.Enum):
    SOMA = "soma"
    PROXIMAL_DENDRITE = "proximal"
    DISTAL_DENDRITE = "distal"
    APICAL_DENDRITE = "apical"
    AXON_HILLOCK = "hillock"


class MorphologyTemplate(enum.Enum):
    POINT = "point"           # no morphology, classic single-compartment
    PYRAMIDAL = "pyramidal"   # long apical dendrite, basal dendrites
    INTERNEURON = "interneuron"  # compact, fast
    STELLATE = "stellate"     # symmetric short dendrites


@dataclass
class Compartment:
    """A single segment of a neuron."""
    ctype: CompartmentType
    length: float = 100.0       # micrometers
    diameter: float = 2.0       # micrometers
    parent_idx: int = -1        # index of parent compartment (-1 = root/soma)

    # Electrical properties
    R_axial: float = 150.0      # axial resistance (Ohm·cm)
    C_membrane: float = 1.0     # membrane capacitance (µF/cm²)

    @property
    def attenuation(self) -> float:
        """
        How much a signal is attenuated when traveling through this compartment.
        Longer, thinner compartments attenuate more.
        """
        length_constant = math.sqrt(self.diameter / (4.0 * self.R_axial * 0.001))
        return math.exp(-self.length / (length_constant * 1000.0))


# Pre-built morphology templates
def _build_pyramidal() -> list[Compartment]:
    return [
        Compartment(CompartmentType.SOMA, length=20.0, diameter=20.0),
        Compartment(CompartmentType.PROXIMAL_DENDRITE, length=100.0, diameter=3.0, parent_idx=0),
        Compartment(CompartmentType.DISTAL_DENDRITE, length=200.0, diameter=1.5, parent_idx=1),
        Compartment(CompartmentType.APICAL_DENDRITE, length=300.0, diameter=2.0, parent_idx=0),
        Compartment(CompartmentType.AXON_HILLOCK, length=30.0, diameter=1.0, parent_idx=0),
    ]


def _build_interneuron() -> list[Compartment]:
    return [
        Compartment(CompartmentType.SOMA, length=15.0, diameter=15.0),
        Compartment(CompartmentType.PROXIMAL_DENDRITE, length=80.0, diameter=2.5, parent_idx=0),
        Compartment(CompartmentType.AXON_HILLOCK, length=20.0, diameter=1.0, parent_idx=0),
    ]


def _build_stellate() -> list[Compartment]:
    return [
        Compartment(CompartmentType.SOMA, length=18.0, diameter=18.0),
        Compartment(CompartmentType.PROXIMAL_DENDRITE, length=80.0, diameter=2.0, parent_idx=0),
        Compartment(CompartmentType.PROXIMAL_DENDRITE, length=80.0, diameter=2.0, parent_idx=0),
        Compartment(CompartmentType.DISTAL_DENDRITE, length=120.0, diameter=1.2, parent_idx=1),
        Compartment(CompartmentType.AXON_HILLOCK, length=25.0, diameter=1.0, parent_idx=0),
    ]


MORPHOLOGY_TEMPLATES: dict[MorphologyTemplate, callable] = {
    MorphologyTemplate.PYRAMIDAL: _build_pyramidal,
    MorphologyTemplate.INTERNEURON: _build_interneuron,
    MorphologyTemplate.STELLATE: _build_stellate,
}


class NeuronMorphology:
    """
    Multi-compartment morphology for a neuron.

    Computes the total attenuation from each compartment to the soma,
    which determines how much synaptic input at that compartment
    affects the soma's membrane potential.
    """

    def __init__(self, template: MorphologyTemplate = MorphologyTemplate.POINT):
        self.template = template
        self.compartments: list[Compartment] = []
        self._attenuation_to_soma: list[float] = [1.0]

        if template != MorphologyTemplate.POINT:
            builder = MORPHOLOGY_TEMPLATES.get(template)
            if builder:
                self.compartments = builder()
                self._compute_attenuation()

    def _compute_attenuation(self) -> None:
        """Compute cumulative attenuation from each compartment to soma."""
        n = len(self.compartments)
        self._attenuation_to_soma = [1.0] * n

        for i in range(n):
            if self.compartments[i].ctype == CompartmentType.SOMA:
                self._attenuation_to_soma[i] = 1.0
                continue

            atten = 1.0
            idx = i
            while idx >= 0 and self.compartments[idx].ctype != CompartmentType.SOMA:
                atten *= self.compartments[idx].attenuation
                idx = self.compartments[idx].parent_idx

            self._attenuation_to_soma[i] = atten

    @property
    def n_compartments(self) -> int:
        return max(len(self.compartments), 1)

    def soma_attenuation(self, compartment_idx: int) -> float:
        """How much a synaptic input at this compartment is attenuated at the soma."""
        if compartment_idx >= len(self._attenuation_to_soma):
            return 1.0
        return self._attenuation_to_soma[compartment_idx]

    @property
    def dendritic_compartments(self) -> list[int]:
        """Indices of compartments where synapses can attach."""
        return [
            i for i, c in enumerate(self.compartments)
            if c.ctype in (
                CompartmentType.PROXIMAL_DENDRITE,
                CompartmentType.DISTAL_DENDRITE,
                CompartmentType.APICAL_DENDRITE,
            )
        ]

    @property
    def soma_idx(self) -> int:
        for i, c in enumerate(self.compartments):
            if c.ctype == CompartmentType.SOMA:
                return i
        return 0


class MorphologyManager:
    """
    Manages morphologies for all neurons in a Region.

    Stores per-synapse attenuation factors based on which compartment
    each synapse targets. This integrates with the Region's synapse arrays.
    """

    def __init__(self):
        self.neuron_morphologies: dict[int, NeuronMorphology] = {}

    def assign_morphology(
        self,
        neuron_idx: int,
        template: MorphologyTemplate,
    ) -> NeuronMorphology:
        """Assign a morphology template to a neuron."""
        morph = NeuronMorphology(template)
        self.neuron_morphologies[neuron_idx] = morph
        return morph

    def assign_default_morphologies(
        self,
        n_neurons: int,
        neuron_types: torch.Tensor,
    ) -> None:
        """Assign default morphologies based on neuron type."""
        from .neuron import NeuronType
        for i in range(n_neurons):
            if neuron_types[i] == NeuronType.EXCITATORY.value:
                self.assign_morphology(i, MorphologyTemplate.PYRAMIDAL)
            else:
                self.assign_morphology(i, MorphologyTemplate.INTERNEURON)

    def compute_synapse_attenuation(
        self,
        n_synapses: int,
        syn_post: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """
        Compute attenuation factor for each synapse based on the
        post-synaptic neuron's morphology and a random compartment assignment.

        Returns tensor of attenuation values [0, 1] per synapse.
        """
        attenuation = torch.ones(n_synapses, dtype=torch.float32, device=DEVICE)

        for i in range(n_synapses):
            post_idx = int(syn_post[i].item())
            morph = self.neuron_morphologies.get(post_idx)
            if morph is None or morph.template == MorphologyTemplate.POINT:
                continue

            dendrites = morph.dendritic_compartments
            if not dendrites:
                continue

            comp_idx = dendrites[int(torch.randint(len(dendrites), (1,), generator=rng, device=DEVICE).item())]
            attenuation[i] = morph.soma_attenuation(comp_idx)

        return attenuation
