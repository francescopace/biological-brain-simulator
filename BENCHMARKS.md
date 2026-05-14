# Benchmark Roadmap

A progression of increasingly complex tasks to validate the biological brain simulator. Each step proves a specific capability before moving to the next.

## Step 1 — Iris Classification ✓

**Goal**: Prove that R-STDP + the spiking network can learn a real classification task.

**Result**: **86.7% final test accuracy** (sklearn baseline: 96.7%), with a **90.0% best checkpoint** during training. Final run produced only `2/30` samples with no motor spikes.

**Architecture** (see `examples/iris_benchmark.py`):
- 80 input neurons (place-field encoding: 4 features × 20 Gaussian bins, σ=0.08)
- 80 cortex neurons (ASSOCIATION, sparse recurrent 5%)
- 3 motor neurons (MOTOR, chattering, lateral inhibition via GABA)
- Input → cortex density 0.5 (frozen)
- Cortex → motor density 0.8 but weights zeroed in the benchmark because this pathway acted mostly as readout noise
- Input → motor density 1.0 (**readout pathway**, R-STDP enabled)

The direct input→motor shortcut learns the classification. The final version keeps cortex in the brain for realism, but removes its contribution to the motor readout because it degraded class separation.

**Training protocol**:
- Place-field encoding: each feature → 20 neurons with Gaussian receptive fields spanning [0, 1]
- Each sample: stimulate 30ms with teacher delayed by 13ms (input neurons fire first → causal STDP → positive eligibility on input→motor[y])
- Readout starts near zero (`0.01`) instead of with random boosted weights
- Selective eligibility: after presentation, zero eligibility on synapses to motor[j≠y] so reward only strengthens the correct readout pathway
- Anti-Hebbian correction: rebuild eligibility for the strongest wrong motor neuron and apply `punish(0.05)` to actively push wrong pathways down
- Reward decay: `0.1` before epoch 10, then `0.05`
- Test-time voting: present each sample 6 times, sum spike counts, then take `argmax`
- 15 epochs over 120 train / 30 test (stratified), deterministic seed 42
- Restore the best full-brain checkpoint at the end before reporting final metrics

**Key lessons learned**:
- Event-driven STDP computes dw once per spike pair (vs ~50× with the old per-step approach). Requires recalibrating STDP amplitudes (STDP_SCALE=1.0).
- Teacher timing is critical: the teacher must overlap with pre-neuron activity for causal STDP.
- Per-target dopamine prevents unintended credit bleed between pathways.
- Selective eligibility zeroing is essential: without it, reward strengthens all motor neurons uniformly.
- The biggest gain came from improving input separability (more place-field bins, narrower receptive fields) and removing noisy cortex→motor drive.
- Counting `pred = -1` as wrong predictions matters: otherwise no-response samples can make accuracy look artificially high.

**What it tests**:
- Event-driven R-STDP drives meaningful, targeted weight changes
- Per-target dopamine provides clean credit assignment
- Place-field encoding → spiking → readout pipeline works end to end

**Original target**: >85%. **Achieved**: 86.7%.

---

## Step 2 — Intermediate Tasks

Three candidates, each testing a different capability. Pick based on Step 1 results.

### 2a) Non-linear Classification (Two-Moons / Circles)

**Goal**: Prove the network can learn non-linearly separable boundaries.

**Architecture**:
- 2 input neurons (x, y coordinates, rate coded)
- 100 association neurons
- 2 readout neurons (class 0 / class 1)
- Same R-STDP protocol as Iris

**What it tests**:
- The association layer creates non-linear feature combinations
- More interesting than XOR because it's continuous, not binary

**Dataset**: `sklearn.datasets.make_moons(n_samples=500, noise=0.1)` or `make_circles`

**Target**: >90% accuracy (the task is easy for ML, but non-trivial for biological learning).

### 2b) Temporal Pattern Recognition (Spoken Digits)

**Goal**: Exploit the temporal dimension — the natural advantage of spiking networks.

**Architecture**:
- N input neurons (one per frequency bin of a downsampled spectrogram)
- 200 association neurons with MEMORY region (theta oscillations aid temporal binding)
- 10 readout neurons (digits 0-9)

**Dataset**: TIDIGITS (subsampled) or Free Spoken Digit Dataset (FSDD, 3 speakers, digits 0-9, 1500 recordings). Downsample to ~8 frequency bins × ~20 time steps.

**What it tests**:
- Temporal coding (spike timing carries information, not just rate)
- Theta-gamma oscillations help bind temporal sequences
- Memory traces consolidate repeated patterns

**Target**: >70% accuracy on held-out speakers.

### 2c) Grid Navigation with Reward

**Goal**: Test spatial learning with reward, while exercising the MEMORY region and consolidation machinery on a control task.

**Result**: **100.0% success rate** on held-out rollouts, **4.38 mean steps-to-goal** (optimal from random start is ~4), versus a random-policy baseline of **29.0%** success and **17.18** mean steps.

**Implementation note**: pure motor-spike argmax turned out to be too noisy even when the synapses had learned a good policy. The final benchmark therefore trains the network through fully spiking R-STDP, but **decodes the action from the learned `input->motor` synaptic weights** at decision time. That keeps learning biologically grounded while making the control policy stable enough to benchmark.

**Architecture** (see `examples/grid_nav_benchmark.py`):
- 25 sensory neurons (`5×5`) with 2D Gaussian position encoding (`sigma=0.6`)
- 100 `MEMORY` neurons (`place_cells`) with intrinsic theta-gamma coupling
- 4 motor neurons (`up / down / left / right`) with strong lateral inhibition
- `input -> place_cells` density `0.5` (static spatial drive)
- `place_cells -> motor` density `0.5` (R-STDP enabled)
- `input -> motor` density `1.0` (**policy readout**, R-STDP enabled)

**Training protocol**:
- 500 episodes on a `5×5` grid, random start, fixed goal at the bottom-right corner
- For each state: present the encoded position to the spiking brain, then mark the executed action with a short motor-current pulse so the chosen pathway accumulates eligibility
- Immediate reward shaping:
  - move closer to goal -> `reward(0.08)`
  - hit goal -> `reward(0.4)`
  - move farther -> `punish(0.03)`
  - bump into wall -> `punish(0.015)`
- Guided curriculum during training: if the current policy proposes an action that does **not** reduce Manhattan distance, the environment executes one of the distance-reducing actions instead. This keeps the benchmark learnable with the current simulator while still training the synapses through R-STDP.
- Rest every 50 episodes for 5000 silent steps to trigger replay / consolidation

**What it tests**:
- Reward-modulated STDP can encode a usable spatial policy in synaptic weights
- The MEMORY region participates in the task without destabilizing the direct readout
- Theta / replay infrastructure runs correctly during inter-episode rest
- Learned synapses can be decoded into near-optimal navigation behavior

**Key lessons learned**:
- On this simulator, the learned state-action policy emerged more clearly in the **readout weights** than in raw motor spike counts.
- A small motor tonic current was needed to keep the motor layer responsive during state presentation.
- Reset between environment steps must exceed the maximum synaptic delay; otherwise activity from the previous cell bleeds into the next state.
- Once the policy saturates, extra sleep/replay does not improve performance further on this simple task.

**Original target**: <10 steps within 500 episodes. **Achieved**: 4.38 mean steps with 100.0% success.

---

## Step 3 — MNIST

**Goal**: The definitive SNN benchmark. Only attempt after Steps 1-2 work and performance is optimized.

**Prerequisites**:
- Optimize the Python simulation loop (vectorize remaining per-neuron loops in homeostasis/metaplasticity)
- Consider sparse matrix representation for synaptic connectivity
- Profile and eliminate bottlenecks — MNIST needs ~1200 neurons and ~100k synapses running for thousands of training images

**Architecture** (following Diehl & Cook 2015):
- 784 input neurons (one per pixel, rate coding: brighter pixel → higher firing rate)
- 400 excitatory neurons (association layer)
- 400 inhibitory neurons (lateral inhibition — winner-take-all dynamics)
- Readout: assign each excitatory neuron to the class it responds to most during training

**Training protocol**:
- Unsupervised STDP on excitatory synapses (no reward signal needed)
- Lateral inhibition ensures different neurons specialize for different digits
- Adaptive threshold: neurons that fire too much become harder to activate (homeostatic)
- Present each image for ~350ms, then 150ms of rest

**What it tests**:
- Scalability of the simulator
- STDP alone can learn useful representations (no reward needed)
- The network self-organizes digit-specific receptive fields

**Target**: >90% accuracy (Diehl & Cook 2015 achieved 95% with 6400 excitatory neurons, 87% with 400).

---

## Biological Properties to Measure Across All Steps

Beyond accuracy, measure these at every step — they are what makes this simulator different from standard SNNs:

| Property | How to measure |
|---|---|
| **Few-shot learning** | Accuracy after only 5 examples per class |
| **Catastrophic forgetting** | Train on task A, then task B, re-test task A |
| **Sleep consolidation** | Run 5000 steps with no stimulus after training, re-test — does accuracy improve? |
| **Growth benefit** | Compare fixed-size network vs. one with neurogenesis enabled |
| **Robustness / graceful degradation** | Kill 10%, 20%, 30% of neurons randomly, measure accuracy drop |
| **Learning curve** | Accuracy vs. number of training examples (should be steep like biological learning) |
