# Benchmark Notes

This file summarizes the latest validated benchmark outcomes in the repository and the modeling choices needed to interpret them. 
The aim is not leaderboard performance; the aim is to test whether biologically motivated local learning rules can support useful behavior on controlled AI tasks.

For exact hyperparameters, read the benchmark scripts:

- `examples/iris_benchmark.py`
- `examples/grid_nav_benchmark.py`
- `examples/mnist_benchmark.py`

## Iris Classification

**Latest validated result**: **86.7%** final test accuracy, with a **90.0%** best checkpoint during training.

**Research setup**:
- Place-field encoding expands the 4 Iris features into `80` sensory neurons.
- The useful readout is the direct `input->motor` pathway trained with R-STDP.
- Anti-Hebbian punishment on the strongest wrong class improves separation.
- Test-time repeated presentation plus voting reduces spiking noise.

**What this benchmark tests**:
- Event-driven R-STDP can drive targeted learning on a real classification task.
- Per-target dopamine and selective eligibility give clean credit assignment.
- The end-to-end encoding -> spiking -> readout pipeline is working.

**Interpretation notes**:
- Teacher timing is critical: reward only helps when pre-synaptic activity overlaps the delayed teacher signal.
- Counting no-response predictions as wrong is important; otherwise accuracy is easy to overstate.

## Grid Navigation

**Latest validated result**: **100.0%** success rate and **4.38** mean steps-to-goal on held-out evaluation rollouts, versus a random-policy baseline of **29.0%** success and **17.18** mean steps.

**Research setup**:
- The task is a `5x5` grid with Gaussian place-field encoding.
- A `MEMORY` region (`place_cells`) participates together with direct `input->motor` readout connections.
- Learning is fully spiking R-STDP with shaped reward/punishment.
- Inter-episode rest is used to exercise replay / consolidation machinery.

**Important caveat**:
- The synapses learn the policy biologically, but action selection is decoded from learned `input->motor` weights because raw motor spike argmax was too noisy for stable control.

**What this benchmark tests**:
- Reward-modulated STDP can encode a usable state-action policy.
- Replay / consolidation infrastructure runs without destabilizing the task.
- The learned policy is near-optimal on this simple environment.

## MNIST

**Latest validated results**:
- **400 exc neurons**: **62.2%** test accuracy on 10-class MNIST (chance: `10%`), with dedicated neurons for all classes and zero no-response samples.
- **1600 exc neurons**: **51.8%** test accuracy — *worse* than 400 neurons, due to training regime undersaturation (see analysis below).

**Architecture** (Diehl & Cook 2015):
- `784` input neurons (one per pixel, rate coded)
- `1600` excitatory + `1600` inhibitory cortex neurons with 1:1 matched WTA microcircuit (~2.9M synapses)
- Unsupervised STDP on feedforward `input->cortex` pathway
- Adaptive thresholds (leaky theta), per-neuron weight normalization, blended spike/voltage readout

**Observed runtimes** (MacBook Air M2, CPU):
- **400 exc**: `3000` train samples, `3` epochs, ~70 ms/sample, **17 min** total
- **1600 exc**: `3000` train samples, `3` epochs, ~250 ms/sample, **~38 min** training + **~21 min** readout + **~4 min** eval = **63 min** total

**GPU and compilation experiments**:

Several approaches to GPU acceleration were tested and none improved over CPU scatter/gather:

| Approach | Result | Why |
|----------|--------|-----|
| MPS direct (400 exc) | **16x slower** than CPU | Kernel launch overhead dominates small tensor ops |
| MPS direct (1600 exc) | **2.4x slower** than CPU | Gap narrows with scale but still loses |
| `torch.compile` | No speedup, falls back to eager | `torch.where` returns variable-length outputs (graph breaks); ring buffer slot indexing triggers recompilation every step |
| Masked full-tensor (no `torch.where`) | **6.7x slower** on CPU, **2.1x slower** on MPS | Multiplying all 2.9M synapses when only 1-5% are active wastes compute; the "zero work" is not free |

The fundamental issue is that SNN activity is **sparse by nature**: only a small fraction of neurons fire each step, so only a small fraction of synapses transmit. The current sparse scatter/gather pattern (`torch.where` → gather active → `index_add_`) touches O(active) elements per step (~30-150k out of 2.9M), which is much faster than any approach that touches all elements.

**Approaches not yet tested** (and their trade-offs):
- **Sparse matrix-vector multiply** (`W_sparse @ fired`): would replace the entire propagation block with one op, but STP (vesicle depletion, facilitation) modifies effective weights every step, requiring per-step matrix reconstruction — likely negating the SpMV advantage.
- **Batched sample presentation**: process B images in parallel on (B, N) state tensors. Would improve GPU utilization but changes STDP semantics (mini-batch vs sequential updates) and requires batch-aware versions of every subsystem (homeostasis, theta, STDP, memory). A 2-3 day refactoring effort best justified for hyperparameter sweeps rather than single-run accuracy.

CPU with sparse scatter/gather remains the correct default at this scale.

**Current caveats**:
- The main gap vs Diehl & Cook is experimental regime, not basic functionality: the validated run uses 300 images/class (vs 6000 in the paper) and 25 ms presentation (vs 350 ms).
- The leaky theta update is a stabilizer, not just a detail; it is what makes the scaled training regime stay usable.

**Observations from the 1600 exc run**:
- Feedforward weight statistics are identical after epoch 1, 2, and 3 (`mean=1.20, max=10.0`), confirming STDP saturates within a single pass over 300 images/class at 25 ms presentation.
- **1600 neurons scored 51.8% vs 62.2% with 400 neurons.** The label distribution is heavily skewed: 480/1157 labelled neurons (41%) assigned to digit 1, while classes 4 and 8 got only 52 neurons each. With 4x more neurons competing for the same 300 images/class, many converge on the simplest features (digit 1) instead of specializing.
- This confirms the bottleneck is training regime, not network capacity. More neurons actually *hurt* when the stimulus regime is too small to drive diverse specialization.

**Next steps for accuracy improvement** (ordered by expected impact):

| Priority | Change | Rationale | Estimated time |
|----------|--------|-----------|----------------|
| 1 | `TRAIN_PRESENT_STEPS=100`, `REST_STEPS=50` | Most impactful single change. At 25 ms many spike pairs never form — STDP needs enough time within each presentation for pre-post coincidences to accumulate. Paper uses 350 ms. Combines with step 2 for maximum effect. | ~4h |
| 2 | `TRAIN_PER_CLASS=1000`, `EPOCHS=2` | The 1600-neuron run shows weights saturate within 1 epoch over 300 images. More unique images per class (not more epochs) is what drives further neuron specialization. Reduce epochs to 2 since the network learns in one pass. Paper uses 6000 images/class. | ~2h |
| 3 | `A_minus/A_plus` ratio 1.2 → 1.05 | Current ratio over-prunes weak classes (observed with digit 8 at 400 exc). 1.05 is the paper value. Zero-cost change, apply together with steps 1-2. | same run |
| 4 | Sweep `THETA_PLUS`, `THETA_LEAK` | Theta regime calibrated for 400 neurons and 25 ms presentation may not be optimal for the new regime. The fast saturation suggests theta may need to be more aggressive to keep competition alive across epochs. | 3-4 runs |
| 5 | Weight normalization target tuning | The norm target may need to scale with neuron count. With 4x more exc neurons competing, each neuron receives 4x fewer input spikes on average. | 2-3 runs |

**Target**: 85-90% with 1600 exc neurons + tuned regime (paper: 87% with 400, 95% with 6400).

## Next Research Measurements

If the goal is to make the project more informative as an AI research artifact, the next measurements worth adding are:

### Accuracy improvements

- **MNIST accuracy push** — the immediate priority (see roadmap above)

### Qualitative advantages over conventional ML

These tests target capabilities where SNN architectures have a structural advantage over standard backprop-trained models. Positive results here would demonstrate value that accuracy-on-benchmarks alone cannot capture.

| Test | What it measures | Protocol sketch | What a positive result looks like |
|------|-----------------|-----------------|-----------------------------------|
| **Few-shot learning curves** | How quickly the network learns from limited data | Train on 1, 5, 10, 50 samples/class; compare accuracy vs a simple MLP baseline with the same data budget | SNN reaches usable accuracy with fewer samples than the MLP, suggesting local plasticity extracts more from each example |
| **Catastrophic forgetting** | Whether sequential task learning destroys previous knowledge | Train on MNIST digits 0-4, then train on 5-9, then re-test on 0-4 | SNN retains significant accuracy on 0-4 after learning 5-9; MLP baseline drops to chance |
| **Sleep / replay benefit** | Whether offline consolidation improves retention | Compare test accuracy with and without a silent replay phase between training blocks | Post-replay accuracy is measurably higher than without replay, validating the consolidation machinery |
| **Growth vs fixed-size** | Whether neurogenesis and synaptogenesis improve learning | Run identical tasks with growth enabled vs disabled (fixed topology) | Growth-enabled networks reach higher accuracy or learn faster, justifying the structural plasticity overhead |
| **Graceful degradation** | Whether the network tolerates damage better than conventional models | After training, randomly kill 5%, 10%, 20% of neurons; measure accuracy drop vs an MLP with the same fraction of weights zeroed | SNN degrades more gracefully (smaller accuracy drop per % of damage) |
| **Temporal pattern recognition** | Whether native spike timing gives an advantage on time-domain tasks | Classify simple temporal patterns (e.g. spike sequences, rhythmic signals) where input order matters, not just content | SNN outperforms a rate-based MLP that receives the same inputs as static vectors |
| **Online adaptation** | Whether the network adapts to distribution shift without retraining | Train on one distribution, then shift (e.g. rotated MNIST digits); measure how quickly accuracy recovers with continued exposure | SNN recovers accuracy through ongoing plasticity while a frozen MLP cannot adapt |
