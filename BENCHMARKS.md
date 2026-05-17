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

**Partial diagnosis: readout-only stress test** (reduced protocol: `100/class` train, `20/class` test, `100ms` train, `25ms` rest):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `baseline_50x2_blend` | **48.0%** | `0.972` | `0.192` | Baseline diagnostic run |
| `matched_100x2_blend` | **47.0%** | `0.963` | `0.199` | Matching readout window to training does not help |
| `matched_100x5_blend` | **48.5%** | `0.963` | `0.199` | More test repeats add little |

**Interpretation**: the first diagnosis pass does **not** support the idea that the MNIST gap is mainly caused by an underpowered decoder. Readout changes moved accuracy by only ±1.5 points on the reduced protocol, while neuron-label balance stayed healthy. This shifts the focus from `build_response_templates()` / `predict_sample()` toward training dynamics: competition strength, STDP calibration, and adaptive-threshold behavior.

**Partial diagnosis: competition sweep** (same reduced protocol):

| `INH_LATERAL_WEIGHT` | Accuracy | Label entropy | Max class share | Takeaway |
|----------------------|----------|---------------|-----------------|----------|
| `6.0` | **54.5%** | `0.957` | `0.185` | Clear gain over the readout baseline |
| `8.0` | **52.0%** | `0.958` | `0.225` | Worse than both `6.0` and `10.0+` |
| `10.0` | **53.0%** | `0.982` | `0.155` | Best balance, moderate accuracy |
| `12.0` | **59.0%** | `0.971` | `0.157` | Best overall short-run result so far |

**Interpretation**: the inhibition sweep produced the strongest signal seen so far. The best point (`12.0`) improved the reduced-protocol accuracy from **48.0%** (readout baseline) to **59.0%** without changing the decoder or the STDP rule. This strongly suggests the MNIST gap is driven more by **competition dynamics** than by readout quality.

**Follow-up validation: larger reduced protocol** (`300/class` train, `30/class` test, `100ms` train, `25ms` rest):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `inh_lateral=10.0_bigger` | **63.0%** | `0.888` | `0.280` | Better than the small-protocol baseline, but class balance degrades |
| `inh_lateral=12.0_bigger` | **64.3%** | `0.928` | `0.213` | Best larger-protocol point so far; wins on both accuracy and balance |

**Interpretation**: `INH_LATERAL_WEIGHT=12.0` held up when the reduced protocol was made more demanding. It improved not only the short-run accuracy but also the balance of neuron specialization relative to `10.0`. This is now the strongest candidate for the next full-MNIST run.

**Partial diagnosis: STDP scale sweep** (interrupted after first point):

| `STDP_SCALE` | Accuracy | Label entropy | Max class share | Takeaway |
|--------------|----------|---------------|-----------------|----------|
| `0.10` | **47.5%** | `0.959` | `0.206` | Lower than the readout baseline; not promising |

**Interpretation**: at least in the first tested point, reducing `STDP_SCALE` did **not** recover accuracy. That does not rule out a better point at `0.2-0.4`, but it makes inhibition retuning look like the higher-value lever.

**Partial diagnosis: theta / homeostasis sweep** (on top of `INH_LATERAL_WEIGHT=12.0`, larger reduced protocol):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `theta_default_12` | **67.0%** | `0.882` | `0.269` | Best accuracy overall |
| `theta_low_plus_12` | **65.7%** | `0.927` | `0.221` | Better balance, but lower accuracy |
| `theta_high_plus_12` | **56.0%** | `0.924` | `0.251` | Clearly harmful |
| `theta_fast_leak_12` | **57.3%** | `0.974` | `0.171` | Very balanced, but too much accuracy loss |

**Interpretation**: once inhibition is retuned to `12.0`, the default theta regime is still the best **accuracy-first** choice. Some theta variants improve class balance, but none beat the default on test accuracy. This means the best validated configuration so far is:

- `INH_LATERAL_WEIGHT = 12.0`
- `THETA_PLUS = 0.10`
- `THETA_LEAK = 0.005`
- `STDP_SCALE = 0.2` (still provisional; only `0.10` has been tested and it underperformed)

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

**Diagnosis-driven next steps**:

| Priority | Change | Why it moved up |
|----------|--------|-----------------|
| 1 | Launch the next full-MNIST run with `INH_LATERAL_WEIGHT=12.0` and default theta | This is the best validated configuration so far (`67.0%` on the larger reduced protocol) |
| 2 | Resume `STDP_SCALE` sweep from `0.2` upward only if more confidence is needed before the long run | `0.10` underperformed; if STDP matters, the useful region is more likely near or above the current baseline |
| 3 | Consider richer plasticity than nearest-neighbor pair STDP | Recent literature increasingly favors adaptive or triplet-style rules when pair-based STDP converges too early |
| 4 | Consider a short post-training adaptation phase (e.g. STP / replay) | Newer work suggests frozen-weight evaluation can leave performance on the table |
| 5 | Only then rerun `TRAIN_PRESENT_STEPS` 200 → 350 or other long-run changes | Another 8–11h run is only justified after isolating which stabilizer actually helps |

**Target**: 75-85% with 400 exc neurons at 200ms (paper: 87% at 350ms). Current best: 64.6%.

## Literature Pointers

Recent papers and reviews that appear directly relevant to the current MNIST gap:

- **Zhuang et al., 2023** — [An unsupervised STDP-based spiking neural network inspired by biologically plausible learning rules and connections](https://www.sciencedirect.com/science/article/pii/S0893608023003301): combines adaptive synaptic filtering, adaptive threshold balancing, adaptive lateral inhibition, and temporal-batch STDP. The article text reports **97.9% on MNIST** and **87.0% on CIFAR-10**. The important lesson for this repo is not the exact number, but that **static inhibition + fixed thresholding + simple STDP is often not enough**.
- **Wu et al., 2024** — [Inhibition SNN: unveiling the efficacy of various lateral inhibition learning in image pattern recognition](https://link.springer.com/article/10.1007/s42452-024-06332-z): shows that **inhibition architecture itself** matters, not just inhibition strength. Their simplified inhibition design reports **86% on MNIST** with an unsupervised `784-100` SNN.
- **Arefnadia et al., 2025** — [Unsupervised post-training learning in spiking neural networks](https://www.nature.com/articles/s41598-025-01749-x): argues that a trained SNN should not necessarily be frozen after the main STDP phase. They combine **triplet STDP** during training with **post-training STP** to improve recognition without changing long-term weights.
- **Khan et al., 2025 review** — [Modulated spike-time dependent plasticity (STDP)-based learning for spiking neural network (SNN): A review](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019416): emphasizes recurring practical bottlenecks in SNN classification, especially **threshold regulation, competition control, parameter optimization, and scalability**.
- **Biologically plausible unsupervised learning for self-organizing spiking neural networks with dendritic computation** (2025 article text surfaced via web search): proposes **adaptive self-organizing inhibition** to keep neurons organized into richer feature groups instead of letting a few easy features dominate.

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
