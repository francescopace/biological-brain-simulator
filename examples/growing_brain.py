"""
Example: A brain that grows from scratch.

Starts with minimal neurons and demonstrates how the brain
organically grows new neurons and connections as it receives
continuous stimulation — just like a developing brain.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain import Brain
from src.region import RegionType


def main():
    print("=" * 60)
    print("  BRAIN SIMULATION: Growing Brain")
    print("  (Neurogenesis & synaptogenesis from minimal start)")
    print("=" * 60)

    brain = Brain(dt=1.0, seed=123)

    # Start very small — like an embryonic brain
    brain.add_region("sense", RegionType.SENSORY, n_neurons=10, connectivity=0.15)
    brain.add_region("process", RegionType.ASSOCIATION, n_neurons=15, connectivity=0.1)
    brain.add_region("store", RegionType.MEMORY, n_neurons=10, connectivity=0.1)

    brain.connect_regions("sense", "process", density=0.1)
    brain.connect_regions("process", "store", density=0.08, bidirectional=True)

    # Make growth more aggressive for this demo
    brain.growth.growth_interval = 200
    brain.growth.neurogenesis_threshold = 0.1
    brain.growth.max_new_neurons_per_cycle = 5
    brain.growth.max_new_synapses_per_cycle = 15

    print(f"\nStarting state:")
    print(brain.summary())

    rng = np.random.default_rng(42)

    # Simulate continuous varied stimulation
    def varied_stimulus(brain_ref, step):
        # Different patterns at different times
        phase = (step // 500) % 4
        if phase == 0:
            values = [0.9, 0.1, 0.5]
        elif phase == 1:
            values = [0.1, 0.9, 0.3]
        elif phase == 2:
            values = [0.5, 0.5, 0.9]
        else:
            values = [rng.random() for _ in range(3)]

        brain_ref.stimulate("sense", values)

    print("\nRunning simulation with varied stimulation...\n")
    brain.run(
        steps=10000,
        stimulus_fn=varied_stimulus,
        progress_interval=2000,
    )

    print(f"\nFinal state:")
    print(brain.summary())

    total_born = sum(g.neurons_born for g in brain.growth.history)
    total_pruned = sum(g.synapses_pruned for g in brain.growth.history)
    total_syn = sum(g.synapses_created for g in brain.growth.history)

    print(f"\n  Total neurons born over simulation: {total_born}")
    print(f"  Total synapses created: {total_syn}")
    print(f"  Total synapses pruned: {total_pruned}")
    print(f"  Net growth: +{total_born} neurons, +{total_syn - total_pruned} synapses")


if __name__ == "__main__":
    main()
