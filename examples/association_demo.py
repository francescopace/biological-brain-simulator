"""
Example: A brain that learns associations.

This demonstrates how the simulated brain can learn to associate two
different stimuli — similar to Pavlov's classical conditioning.

Scenario:
- A "visual" region receives stimulus A (e.g. seeing food)
- An "auditory" region receives stimulus B (e.g. hearing a bell)
- An "association" region learns to connect them
- After training, presenting only stimulus B activates the
  association cortex in a pattern similar to stimulus A

The brain starts with ~200 neurons and grows as it learns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain import Brain
from src.region import RegionType


def main():
    print("=" * 60)
    print("  BRAIN SIMULATION: Learning Associations")
    print("  (Pavlovian conditioning with spiking neurons)")
    print("=" * 60)

    # Create a small brain
    brain = Brain(dt=1.0, seed=42)

    # Add regions
    brain.add_region("visual", RegionType.SENSORY, n_neurons=30, connectivity=0.1)
    brain.add_region("auditory", RegionType.SENSORY, n_neurons=30, connectivity=0.1)
    brain.add_region("cortex", RegionType.ASSOCIATION, n_neurons=60, connectivity=0.08)
    brain.add_region("memory", RegionType.MEMORY, n_neurons=40, connectivity=0.08)

    # Connect regions (like white matter tracts)
    brain.connect_regions("visual", "cortex", density=0.08)
    brain.connect_regions("auditory", "cortex", density=0.08)
    brain.connect_regions("cortex", "memory", density=0.06, bidirectional=True)

    print(f"\nInitial state:")
    print(brain.summary())

    # ── Phase 1: Baseline (no learning yet) ───────────────────────
    print("\n" + "─" * 60)
    print("Phase 1: Baseline — presenting stimuli separately")
    print("─" * 60)

    # Stimulus A: a specific visual pattern
    stimulus_a = [0.9, 0.1, 0.8, 0.2, 0.7]
    # Stimulus B: a specific auditory pattern
    stimulus_b = [0.2, 0.8, 0.3, 0.9, 0.1]

    # Present stimulus A alone
    for _ in range(500):
        brain.stimulate("visual", stimulus_a)
        brain.step()

    cortex_activity_a = brain.regions["cortex"].mean_activity
    print(f"  Cortex activity with visual stimulus A: {cortex_activity_a:.4f}")

    # Present stimulus B alone
    for _ in range(500):
        brain.stimulate("auditory", stimulus_b)
        brain.step()

    cortex_activity_b = brain.regions["cortex"].mean_activity
    print(f"  Cortex activity with auditory stimulus B: {cortex_activity_b:.4f}")

    # ── Phase 2: Paired training (learning the association) ───────
    print("\n" + "─" * 60)
    print("Phase 2: Paired training — presenting A and B together")
    print("─" * 60)

    for epoch in range(5):
        # Present A + B simultaneously (like bell + food)
        for _ in range(1000):
            brain.stimulate("visual", stimulus_a)
            brain.stimulate("auditory", stimulus_b)
            brain.step()

        # Rest period (consolidation)
        for _ in range(500):
            brain.step()

        print(f"  Epoch {epoch + 1}/5 — {brain.summary().split(chr(10))[0]}")

    print(f"\nAfter training:")
    print(brain.summary())

    # ── Phase 3: Test — does B alone activate A's pattern? ────────
    print("\n" + "─" * 60)
    print("Phase 3: Testing — presenting only stimulus B (auditory)")
    print("─" * 60)

    # Reset activity
    for _ in range(200):
        brain.step()

    # Now present only B
    for _ in range(500):
        brain.stimulate("auditory", stimulus_b)
        brain.step()

    cortex_activity_test = brain.regions["cortex"].mean_activity
    memory_activity = brain.regions["memory"].mean_activity

    print(f"  Cortex activity with B alone (after training): {cortex_activity_test:.4f}")
    print(f"  Memory activity: {memory_activity:.4f}")
    print(f"  Cortex activity with B alone (before training): {cortex_activity_b:.4f}")

    if cortex_activity_test > cortex_activity_b * 1.1:
        print("\n  ✓ The brain has learned the association!")
        print("    Stimulus B now activates a richer cortex pattern,")
        print("    incorporating connections strengthened by co-presentation with A.")
    else:
        print("\n  The association is subtle — try more training epochs")
        print("  or increase the number of neurons.")

    # ── Phase 4: Growth report ────────────────────────────────────
    print("\n" + "─" * 60)
    print("Growth History")
    print("─" * 60)

    total_born = sum(g.neurons_born for g in brain.growth.history)
    total_syn_created = sum(g.synapses_created for g in brain.growth.history)
    total_pruned = sum(g.synapses_pruned for g in brain.growth.history)
    total_died = sum(g.neurons_died for g in brain.growth.history)

    print(f"  Growth cycles:       {len(brain.growth.history)}")
    print(f"  Neurons born:        {total_born}")
    print(f"  Neurons died:        {total_died}")
    print(f"  Synapses created:    {total_syn_created}")
    print(f"  Synapses pruned:     {total_pruned}")
    print(f"  Memory traces:       {len(brain.memory.traces)}")

    # Save state
    brain.save_state("brain_state.json")
    print(f"\n  Brain state saved to brain_state.json")

    print("\n" + "=" * 60)
    print("  Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
