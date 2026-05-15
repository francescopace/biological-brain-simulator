# Biological Brain Simulator

A simulator that reproduces the fundamental mechanisms of the human brain: spiking neurons, synapses that strengthen or weaken with use, new connections forming and old ones dying. This is not machine learning — it is a biological simulation.

The brain starts small and grows organically as it receives input, forming new neurons and connections based on activity patterns, just like a developing biological brain.

## What it simulates

| Biological mechanism | Implementation |
|---|---|
| **Neurons** | Izhikevich model — reproduces 7 real firing patterns (regular spiking, bursting, chattering, fast spiking, ...) |
| **Synapses** | Variable-weight connections with axonal delay, neurotransmitters (glutamate, GABA, dopamine, serotonin, acetylcholine), short-term plasticity (vesicle depletion + facilitation) |
| **STDP** | Event-driven Spike-Timing Dependent Plasticity — neurons that fire together wire together, computed only at spike events |
| **R-STDP** | Reward-Modulated STDP — three-factor learning rule (STDP × dopamine) with per-synapse eligibility traces and per-target dopamine |
| **Homeostatic plasticity** | Synaptic scaling to keep firing rates in a healthy range |
| **Metaplasticity** | The plasticity of plasticity itself adapts (BCM theory) |
| **Neurogenesis** | Birth of new neurons in active regions |
| **Synaptogenesis** | New connections between co-active neurons |
| **Pruning** | Elimination of weak/unused synapses |
| **Apoptosis** | Programmed death of chronically inactive neurons |
| **Memory** | Activity traces, consolidation via replay (like during sleep), pattern completion |
| **Oscillations** | Theta/alpha/beta/gamma rhythms with theta-gamma cross-frequency coupling |
| **Morphology** | Multi-compartment neurons (dendrites/soma/axon) with distance-based synaptic attenuation |
| **Persistence** | Full save/restore of the entire brain state to disk |

## How it differs from ML

| Traditional ML | This simulator |
|---|---|
| Training phase, then the model is fixed | Continuous learning, never stops |
| Fixed size | Grows organically (neurogenesis) |
| Backpropagation | STDP + dopamine (biologically plausible) |
| Neurons = mathematical functions | Neurons with real membrane dynamics |
| No internal structure | Regions, connectivity, neurotransmitters, oscillations |

## Quickstart

**Requirements:** Python >= 3.10, plus numpy, matplotlib, networkx, scikit-learn (all pinned in `requirements.txt`).

```bash
pip install -r requirements.txt

# Run from the project root
python examples/learn_association.py
python examples/growing_brain.py
python examples/iris_benchmark.py      # classification benchmark
python examples/grid_nav_benchmark.py  # grid navigation benchmark
python examples/mnist_prototype.py     # MNIST benchmark (10-class, 784+400+400)
```

## Project structure

```
src/
├── __init__.py      # Public API re-exports
├── neuron.py        # Neuron types and Izhikevich parameters
├── synapse.py       # Neurotransmitter types and properties
├── region.py        # Brain region (vectorized NumPy arrays)
├── brain.py         # Top-level orchestrator + inter-region projections
├── plasticity.py    # STDP, R-STDP, homeostatic, metaplasticity (BCM)
├── growth.py        # Neurogenesis, synaptogenesis, pruning, apoptosis
├── memory.py        # Memory traces, consolidation, pattern completion
├── stimulus.py      # Sensory encoding (rate/temporal/population coding)
├── oscillator.py    # Brain oscillations (theta/alpha/beta/gamma)
├── morphology.py    # Multi-compartment neuron morphology
└── persistence.py   # Full brain serialization and deserialization

examples/
├── learn_association.py   # Pavlovian conditioning with spiking neurons
├── growing_brain.py       # A brain that grows from scratch
├── iris_benchmark.py      # Iris classification with R-STDP (86.7% accuracy)
├── grid_nav_benchmark.py  # Grid navigation with R-STDP (100% success, 4.38 steps)
└── mnist_prototype.py     # MNIST benchmark: 10-class unsupervised STDP (56% accuracy)

BENCHMARKS.md              # Detailed benchmark results and methodology
```

## Usage

```python
from src.brain import Brain
from src.region import RegionType
from src.persistence import save_brain, load_brain

brain = Brain(dt=1.0, seed=42)

# Available region types: SENSORY, ASSOCIATION, MEMORY, MOTOR, REWARD
brain.add_region("input", RegionType.SENSORY, n_neurons=50)
brain.add_region("cortex", RegionType.ASSOCIATION, n_neurons=200)
brain.add_region("output", RegionType.MOTOR, n_neurons=10)
brain.connect_regions("input", "cortex", density=0.1)
brain.connect_regions("cortex", "output", density=0.1)

# Enable multi-compartment neuron morphology (optional)
brain.regions["cortex"].enable_morphology()

# Stimulate and let it learn
for step in range(10000):
    brain.stimulate("input", [0.8, 0.3, 0.5])
    if step == 5000:
        brain.reward(2.0, target="proj:input->cortex")
    if step == 7000:
        brain.punish(1.0, target="proj:cortex->output")
    brain.step()

# The brain has grown
print(brain.summary())

# Save to disk — full state, down to individual synapse weights
save_brain(brain, "my_brain")

# Load and continue from where it left off
brain = load_brain("my_brain")
for step in range(5000):
    brain.stimulate("input", [0.2, 0.9, 0.4])
    brain.step()
```

## Iris Benchmark

`examples/iris_benchmark.py` is the most complete supervised benchmark in the repo.
The current configuration uses:

- place-field encoding with `4 × 20 = 80` sensory neurons
- a direct `input->motor` readout trained with reward-modulated STDP
- anti-Hebbian punishment on the strongest wrong motor neuron
- repeated test-time presentations to reduce spiking noise

With the current settings, the benchmark reaches **86.7% test accuracy**
on Iris, with a best checkpoint of **90.0%** during training.
See [BENCHMARKS.md](BENCHMARKS.md) for detailed results and methodology.

## Grid Navigation Benchmark

`examples/grid_nav_benchmark.py` is the current reinforcement-learning benchmark.
It uses:

- a `5x5` grid encoded with `25` sensory neurons and 2D Gaussian place fields
- a `MEMORY` region (`place_cells`) with theta-gamma coupling
- reward-modulated STDP on `input->motor` and `place_cells->motor`
- inter-episode rest periods to exercise replay / consolidation infrastructure

With the current settings, the benchmark reaches **100.0% success rate**
with **4.38 mean steps-to-goal** over held-out evaluation rollouts
(random baseline: **29.0%** success, **17.18** mean steps).

One important caveat: the synapses learn the policy through fully spiking
R-STDP, but action selection is decoded from the learned `input->motor`
weights because raw motor spike argmax was too noisy for stable control.
See [BENCHMARKS.md](BENCHMARKS.md) for detailed results and methodology.

## MNIST Benchmark

`examples/mnist_prototype.py` is the Step 3 benchmark following the Diehl &
Cook 2015 architecture:

- real MNIST loaded from OpenML, all **10 digit classes**
- `784` input neurons (full `28x28`, rate coded) with L1 intensity equalization
- `400` excitatory + `400` inhibitory cortex neurons (`190k` total synapses)
- unsupervised STDP on `input->cortex` with per-neuron weight normalization
- adaptive excitability threshold during training
- winner-take-all exc/inh microcircuit for competitive specialization
- blended readout over cortex spike and voltage templates

The current configuration reaches **56%** test accuracy on 10-class MNIST
(chance: 10%), with all 10 classes receiving dedicated excitatory neurons
and zero no-response samples. The full run completes in under **4 minutes**
(training: 128s, readout: 72s, evaluation: 27s) with `1,584` neurons and
`190k` synapses.

The gap to the Diehl & Cook target (87% with 400 exc neurons) is mainly due
to using only `50` training images per class (vs 6,000 in the paper) and
shorter presentation windows (`25` steps vs `700`). These can be increased
at the cost of longer wall time.

## Performance

All neuron and synapse state lives in dense NumPy arrays (structure-of-arrays layout). A single `Region.step()` call advances thousands of neurons in parallel rather than looping over individual Python objects.

Key techniques:

- **Vectorized Izhikevich integration** with configurable sub-stepping (`dt / 0.5`)
- **Ring buffer** for spike delays — `np.add.at` deposits postsynaptic currents into future slots; each timestep reads and clears the current slot
- **Event-driven STDP** — nearest-neighbor pairing updates only synapses whose pre or post neuron fired this step, not the entire weight matrix every timestep
- **Sparse-like propagation** — synapses are stored as COO-style parallel arrays; spike delivery filters on `fired[syn_pre]` so only active synapses are touched
- **Pre-allocated neuron arrays** up to `max_neurons`; synapse arrays grow via capacity-doubling when needed
- **Dead flags** (`syn_alive`, `neuron_alive`) for pruning and apoptosis — no costly array compaction

Current limitations: the per-neuron scans in homeostatic plasticity and metaplasticity have already been vectorized, and weight normalization uses fully vectorized `np.add.at` rather than per-neuron loops. The main remaining large-scale bottlenecks are synaptogenesis pair enumeration, pattern completion, and the dense per-step memory/bandwidth cost of very large connectivity. The full MNIST benchmark (`784+400+400`, `190k` synapses) runs at ~85ms per training sample, demonstrating feasible simulation speed at this scale.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Francesco Pace
