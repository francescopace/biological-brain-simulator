# Biological Brain Simulator

An open-source, research-oriented simulator for studying biologically inspired learning in spiking neural systems. 
The project combines spiking neurons, synaptic plasticity, structural growth, memory / replay, oscillations, and full-state persistence in a single experimental framework.

The goal is not to claim a faithful whole-brain model. 
The goal is to provide a practical sandbox for testing whether local learning rules and biologically motivated dynamics can support non-trivial behavior on controlled AI tasks.

## Modeling scope

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

## Research positioning

| Mainstream ML pipeline | This simulator |
|---|---|
| Distinct training and inference phases | Continuous adaptation during interaction |
| Fixed size | Grows organically (neurogenesis) |
| Backpropagation | Local plasticity rules (STDP, R-STDP, homeostasis) |
| Abstract activations | Spiking membrane dynamics |
| Minimal internal structure | Regions, projections, neurotransmitters, oscillations |

## Status

This project is best understood as an experimental research codebase rather than a polished general-purpose framework. The current implementation is useful for exploring biologically inspired learning mechanisms, running controlled benchmarks, and testing architectural ideas, but the benchmark scripts remain the canonical reference for exact experimental settings.

The strongest current evidence in the repo is: supervised R-STDP on Iris, reward-modulated control on grid navigation, and unsupervised STDP on a scaled MNIST setup. The main research gap is not whether the simulator can learn at all, but how far these mechanisms can be pushed while remaining stable, interpretable, and biologically motivated.

## Quickstart

**Requirements:** Python >= 3.10, plus PyTorch, matplotlib, networkx, scikit-learn (all pinned in `requirements.txt`).

```bash
pip install -r requirements.txt

# Run from the project root
python examples/association_demo.py
python examples/growth_demo.py
python examples/iris_benchmark.py      # classification benchmark
python examples/grid_nav_benchmark.py  # grid navigation benchmark
python examples/mnist_benchmark.py     # MNIST benchmark (10-class, 784+1600+1600)
python examples/mnist_diagnosis.py     # short MNIST diagnosis sweeps
```

## Project structure

```
src/
├── __init__.py      # Public API re-exports
├── neuron.py        # Neuron types and Izhikevich parameters
├── synapse.py       # Neurotransmitter types and properties
├── device.py        # Compute device selection (CPU/MPS/CUDA)
├── region.py        # Brain region (vectorized PyTorch tensors)
├── brain.py         # Top-level orchestrator + inter-region projections
├── plasticity.py    # STDP, R-STDP, homeostatic, metaplasticity (BCM)
├── growth.py        # Neurogenesis, synaptogenesis, pruning, apoptosis
├── memory.py        # Memory traces, consolidation, pattern completion
├── stimulus.py      # Sensory encoding (rate/temporal/population coding)
├── oscillator.py    # Brain oscillations (theta/alpha/beta/gamma)
├── morphology.py    # Multi-compartment neuron morphology
└── persistence.py   # Full brain serialization and deserialization

examples/
├── association_demo.py       # Pavlovian conditioning with spiking neurons
├── growth_demo.py            # A brain that grows from scratch
├── iris_benchmark.py         # Iris classification with R-STDP
├── grid_nav_benchmark.py     # Grid navigation with R-STDP
├── mnist_benchmark.py        # MNIST benchmark: 10-class unsupervised STDP
├── mnist_diagnosis.py        # MNIST diagnosis: readout / inhibition / STDP / theta sweeps
├── fewshot_benchmark.py      # Few-shot learning: SNN vs MLP data efficiency
├── forgetting_benchmark.py   # Catastrophic forgetting: sequential task retention
├── degradation_benchmark.py  # Graceful degradation: robustness to neuron damage
├── sleep_benchmark.py        # Sleep/replay: consolidation benefit measurement
└── growth_benchmark.py       # Growth vs fixed: structural plasticity ablation

BENCHMARKS.md              # Concise benchmark notes, results, and caveats
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

### Convenience helpers

The core simulator exposes a few helpers that are useful when building benchmarks or structured training loops:

- `brain.reset_traces()` clears all synaptic eligibility traces and per-target dopamine
- `brain.freeze_structural_plasticity()` disables growth and metaplasticity for stable experiments
- `brain.freeze_plasticity()` disables STDP everywhere
- `brain.enable_projection_plasticity("input", "output")` re-enables STDP on one projection
- `brain.get_projection("input", "output")` fetches an inter-region projection safely
- `brain.regions["output"].add_lateral_inhibition(weight=5.0)` creates all-to-all lateral inhibition

Example:

```python
brain.freeze_plasticity()
brain.enable_projection_plasticity("input", "output", A_plus=0.01, A_minus=0.012)
brain.freeze_structural_plasticity()
brain.regions["output"].add_lateral_inhibition(weight=5.0)
```

## Experimental Results

The repository currently includes three validated reference benchmarks:

| Benchmark | Script | Main question | Latest validated result |
|---|---|---|---|
| Iris classification | `examples/iris_benchmark.py` | Can reward-modulated local plasticity solve a standard supervised classification task? | **86.7%** test accuracy, **90.0%** best checkpoint |
| Grid navigation | `examples/grid_nav_benchmark.py` | Can the simulator learn a usable control policy with reward-modulated spiking dynamics? | **100.0%** success, **4.38** mean steps-to-goal |
| MNIST | `examples/mnist_benchmark.py` | Can the simulator scale to a non-trivial unsupervised vision benchmark? | **64.6%** test accuracy (400 exc, full MNIST 6000/class, 200ms, 15M timesteps). Ongoing diagnosis suggests the main bottleneck is training dynamics (competition / STDP / theta); the best validated next-run configuration so far is `INH_LATERAL_WEIGHT=12.0` with the default theta regime — see BENCHMARKS.md |

`BENCHMARKS.md` contains the detailed benchmark notes, including research setup, caveats, runtime observations, and interpretation. Use the benchmark scripts themselves as the source of truth for exact hyperparameters.

## Computational Performance

The simulator is implemented as a vectorized research codebase rather than a neuron-by-neuron object model. All neuron and synapse state lives in dense PyTorch tensors (structure-of-arrays layout), so a single `Region.step()` call can advance thousands of neurons in parallel.

Key techniques:

- **PyTorch tensors** on CPU — extensive profiling showed CPU outperforms MPS (Apple GPU) at all tested scales (up to 3200 neurons / 2.9M synapses). GPU kernel launch overhead dominates small conditional ops; `torch.compile` fails due to dynamic shapes; masked full-tensor approaches waste compute on the ~95-99% of inactive synapses. See [BENCHMARKS.md](BENCHMARKS.md) for details. Override with `BRAIN_DEVICE=mps` for very large dense networks where GPU occupancy may eventually win.
- **Vectorized Izhikevich integration** with configurable sub-stepping (`dt / 0.5`)
- **Ring buffer** for spike delays — `index_add_` deposits postsynaptic currents into future slots; each timestep reads and clears the current slot
- **Event-driven STDP** — nearest-neighbor pairing updates only synapses whose pre or post neuron fired this step, not the entire weight matrix every timestep. Sync-free implementation avoids GPU stalls.
- **Sparse scatter/gather propagation** — synapses are stored as COO-style parallel arrays; spike delivery filters on `fired[syn_pre]` so only active synapses are touched (O(active) not O(all)). This is the key SNN advantage: with 1-5% neuron activity per step, scatter/gather touches ~30-150k elements vs 2.9M for any dense approach.
- **Vectorized stimulus encoding** — rate/temporal/population coding computed in a single tensor operation instead of per-neuron Python loops
- **Vectorized synaptogenesis** — co-activity scoring via `torch.outer` instead of nested Python loops
- **Pre-allocated neuron arrays** up to `max_neurons`; synapse arrays grow via capacity-doubling when needed
- **Dead flags** (`syn_alive`, `neuron_alive`) for pruning and apoptosis — no costly array compaction

Run-specific timing numbers live in [BENCHMARKS.md](BENCHMARKS.md) alongside the corresponding benchmark results.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Francesco Pace**  
Email: [francesco.pace@gmail.com](mailto:francesco.pace@gmail.com)
LinkedIn: [linkedin.com/in/francescopace](https://www.linkedin.com/in/francescopace/)
