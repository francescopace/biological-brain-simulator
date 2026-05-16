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

**Validated results**:

| Run | Neurons | Train/class | Present (ms) | Epochs | Total timesteps | Accuracy | Notes |
|-----|---------|-------------|-------------|--------|-----------------|----------|-------|
| v1 | 400 exc | 300 | 25 | 3 | 225k | **62.2%** | Baseline |
| v2 | 1600 exc | 300 | 25 | 3 | 225k | **51.8%** | More neurons hurt with insufficient data |
| v3 | 400 exc | 6000 | 200 | 1 | 15M | **64.6%** | Full MNIST, near-paper regime |
| Paper | 400 exc | 6000 | 350 | 1 | 21M | **87%** | Diehl & Cook 2015 reference |

**Key finding: the bottleneck is total learning exposure, not network capacity.** The v1→v2 regression (62%→52%) proved that scaling neurons without scaling data is counterproductive. Weights saturate within 1 epoch — additional epochs over the same data add nothing. The v3 run confirmed that scaling data from 300→6000/class with longer presentation (200ms) yields a modest improvement (62.2%→64.6%), but the gain is smaller than expected — suggesting that presentation duration and STDP calibration matter more than raw data volume alone.

**Learning efficiency comparison**: v1 achieved 62.2% with only 225k timesteps — a rate of **276% accuracy per million timesteps** vs the paper's **4.1%/M**. v3 achieved 64.6% with 15M timesteps — **4.3%/M**, almost identical to the paper's efficiency at equivalent scale. This confirms our architectural additions (L1 equalization, blended spike/voltage readout, leaky theta) give a large efficiency advantage at small data budgets, but at scale the learning rate converges toward the paper's baseline.

**v3 neuron label distribution**: 384/400 neurons labelled. Heavy skew toward digit 1 (179 neurons, 47% of labelled) while harder classes (3, 5, 8) got fewer than 10 neurons each. This label imbalance is the primary accuracy limiter — the network has strong per-class accuracy on easy digits but near-chance on hard ones.

**Architecture** (Diehl & Cook 2015):
- `784` input neurons (one per pixel, rate coded)
- excitatory + inhibitory cortex neurons with 1:1 matched WTA microcircuit
- Unsupervised STDP on feedforward `input->cortex` pathway
- Adaptive thresholds (leaky theta), per-neuron weight normalization, blended spike/voltage readout

**Observed runtimes** (MacBook Air M2, CPU):
- **400 exc, 25ms**: `3000` train, `3` epochs, ~70 ms/sample, **17 min** total
- **1600 exc, 25ms**: `3000` train, `3` epochs, ~250 ms/sample, **63 min** total
- **400 exc, 200ms** (v3): `60000` train, `1` epoch, ~357 ms/sample, **5.95h** total (incl. readout + eval)

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

**Observations from completed runs**:
- Feedforward weights saturate within 1 epoch (`mean=1.20, max=10.0` after every epoch). More epochs over the same data add nothing — confirmed on both 400 and 1600 neurons.
- 1600 neurons scored 51.8% vs 62.2% with 400 neurons. The label distribution was heavily skewed: 480/1157 labelled neurons (41%) assigned to digit 1, while classes 4 and 8 got only 52 neurons each. With 4x neurons competing for the same 300 images/class, many converge on the simplest features instead of specializing.
- The leaky theta update is a critical stabilizer; it is what makes the adaptive threshold regime stay usable.

**Changes applied for v3 run** (all simultaneously):

| Change | From | To | Rationale |
|--------|------|----|-----------|
| `TRAIN_PER_CLASS` | 300 | 6000 (full MNIST) | Data diversity is the primary learning driver |
| `TRAIN_PRESENT_STEPS` | 25 | 200 | More time for spike-pair coincidences to accumulate |
| `EPOCHS` | 3 | 1 | Extra epochs add nothing (weights saturate in one pass) |
| `REST_STEPS` | 5 | 50 | Longer rest between samples lets theta decay properly |
| `A_minus/A_plus` | 1.20 | 1.05 | Paper value; 1.2 over-prunes weak classes |
| `STDP_SCALE` | 0.8 | 0.2 | Compensates 8x more STDP events per sample at 200ms |

**Remaining tuning** (for follow-up runs if v3 underperforms):

| Priority | Change | Rationale |
|----------|--------|-----------|
| 1 | `TRAIN_PRESENT_STEPS` 200 → 350 | Match the paper exactly; requires ~11h at 400 exc |
| 2 | `STDP_SCALE` sweep (0.1 – 0.5) | May be over- or under-scaled for the new regime |
| 3 | `THETA_PLUS` / `THETA_LEAK` sweep | Theta regime was tuned for 25ms; may need recalibration |
| 4 | Weight normalization target tuning | Norm target is auto-computed; could benefit from explicit tuning |
| 5 | Scale to 1600 exc | Only *after* the training regime is validated at 400 |

**Target**: 75-85% with 400 exc neurons at 200ms (paper: 87% at 350ms). Current best: 64.6%.

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
