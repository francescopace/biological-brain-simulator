# Benchmark Notes

These diagnostic benchmarks test whether biologically motivated local learning rules support useful behavior on controlled AI tasks. They are not leaderboard submissions.

The numerical results below are historical pre-fix measurements. They must be rerun before being promoted again. No benchmark was launched as part of the latest correction pass.

The stopped diagnostic session that motivated the latest fixes found:

- Iris completed at **80.0% spike-only held-out accuracy**, with **4/30** silent samples. The voltage fallback is now reported separately instead of being folded into the primary metric.
- Grid was stopped at episode **275/500**. Deterministic evaluations fluctuated between roughly **12.5% and 40%** success and replay sometimes reduced performance; this did not establish learning above the measured **29% random baseline**.
- The reduced MNIST baseline completed at **23.0%**, well below the historical **48.0%** reduced result. Inspection found all `159,600` lateral GABA synapses at `-10`, while excitatory internal weights had collapsed from about `8` to a mean near `1.83`.

Those observations led to the following protocol/dynamics corrections:

1. Homeostatic scaling now converts the raw decayed activity count to Hz before comparing it with `target_rate`; the previous comparison was off by about `200x` at `dt=1ms` and decay `0.995`.
2. Synaptic scaling and adaptive theta can be enabled independently. MNIST freezes only scaling, preserving theta, while Grid freezes scaling for a stable minimal baseline.
3. Initial synaptic weights are clamped to their declared bounds. The current MNIST WTA therefore uses inhibition `10.0`, not the out-of-range historical `12.0`.
4. MNIST uses the canonical `60,000/10,000` split, fits intensity equalization on training data only, and requests `5,000` training examples per class. Grid now isolates direct one-hot state-to-action R-STDP and disables replay/oscillations until the minimal policy learns.
5. Validation/readout/test run on copies with adaptive thresholds frozen and encoder noise disabled. Membrane state and short-term synaptic dynamics still evolve within each evaluation sequence.

The follow-up review added these corrections, verified with unit tests and synthetic networks only:

- Grid now uses 25 excitatory sensory neurons, providing all 100 state/action connections. Grid and Iris restrict credit for the full reward or punishment pulse, including newly generated spikes, and clear residual dopamine at the end.
- MNIST diagnosis resolves rest duration at call time: the reduced protocol now actually uses its configured 25 ms. The competition sweep uses only 6, 8 and 10, avoiding a duplicate run caused by clipping 12 to 10. Historical reduced results should not be assumed to have used the declared rest duration.
- MNIST readout and evaluation freeze topology and synaptic adaptation on copies, keeping neuron indices consistent even for growth experiments. The degradation benchmark's missing import has been restored. The forgetting test reuses task A's original decoder after training B; both SNN and MLP are evaluated with the task's candidate classes.
- STDP/R-STDP freeze state survives simulation steps and save/load. Eligibility decay uses elapsed milliseconds via `exp(-dt/tau)`; this does not make every simulator mechanism invariant to the integration timestep.
- Memory admits new patterns by evicting the oldest stored trace when full, including consolidated traces. Consecutive identical patterns within a region refresh the activity snapshot instead of occupying additional slots. Strength still weights replay sampling.

These changes have not been benchmarked. They establish the intended protocol; they do not yet demonstrate recovery of the historical accuracy.

For exact hyperparameters, read the benchmark scripts:

- `examples/iris_benchmark.py`
- `examples/grid_nav_benchmark.py`
- `examples/mnist_benchmark.py`

## Iris classification

Historical pre-fix result: **86.7%** final test accuracy, with a **90.0%** checkpoint selected using the test set. This number is retained for provenance, not as a valid held-out estimate.

Research setup:
- Place-field encoding expands the 4 Iris features into `80` sensory neurons.
- The useful readout is the direct `input->motor` pathway trained with R-STDP.
- Anti-Hebbian punishment on the strongest wrong class improves separation.
- Test-time repeated presentation plus voting reduces spiking noise.
- The corrected protocol fits normalization on training data, selects checkpoints on validation data, and evaluates the test set once.

Questions for the corrected benchmark:
- Whether event-driven R-STDP can drive targeted learning on the Iris dataset.
- Whether per-target dopamine and selective eligibility improve credit assignment.
- Whether the complete encoding -> spiking -> readout pipeline learns on held-out data.

Interpretation notes:
- Teacher timing is critical: reward only helps when pre-synaptic activity overlaps the delayed teacher signal.
- Counting no-response predictions as wrong is important; otherwise accuracy is easy to overstate.

## Grid navigation

Historical pre-fix result: **100.0%** success rate and **4.38** mean steps-to-goal. The old training loop replaced non-optimal actions with an environment-derived teacher action, and evaluation preferred a direct weight decoder; the result therefore did not measure autonomous spike-based reinforcement learning.

Research setup:
- The task is a `5x5` grid with one-hot state encoding.
- A single plastic `input->motor` projection maps `25` states to `4` actions.
- Learning uses spiking R-STDP with shaped reward/punishment, a per-state reward baseline, and no optimal-action teacher.
- Replay and oscillations are disabled in the minimal baseline and belong in later controlled ablations.

Corrected action selection:
- Exploration samples a random action.
- Exploitation uses motor spike counts, with membrane voltage only as a fallback when no motor neuron spikes.

Questions for the corrected benchmark:
- Whether reward-modulated STDP can encode a usable state-action policy.
- Whether the learned policy improves over a random baseline when actions are selected from motor activity.
- Whether eligibility, weight-change, saturation, and action-margin diagnostics show meaningful credit assignment before adding architectural complexity.

## MNIST

Current corrected protocol (results pending):

- `5,000` canonical training samples per class and `50` canonical test samples per class
- `400` excitatory + `400` inhibitory cortex neurons
- lateral inhibition `10.0`, matching the declared `[-10, 0]` inhibitory weight bounds
- homeostatic synaptic scaling frozen; adaptive theta retained
- memory and oscillations disabled to isolate feedforward STDP + WTA
- L1 equalization target fitted on the selected training set only

Historical results pending revalidation:

| Run | Neurons | Train/class | Present (ms) | Epochs | Total timesteps | Accuracy | Notes |
|-----|---------|-------------|-------------|--------|-----------------|----------|-------|
| v1 | 400 exc | 300 | 25 | 3 | 225k | **62.2%** | Baseline |
| v2 | 1600 exc | 300 | 25 | 3 | 225k | **51.8%** | More neurons hurt with insufficient data |
| v3 | 400 exc | 6000 | 200 | 1 | 15M | **64.6%** | Full MNIST, near-paper regime |
| v4 | 400 exc | 6000 | 200 | 1 | 15M | **66.0%** | Same as v3 + `INH_LATERAL_WEIGHT=12.0` |
| Paper | 400 exc | 6000 | 350 | 1 | 21M | **87%** | Diehl & Cook 2015 reference |

The v1→v2 regression (62%→52%) occurred when the network grew from 400 to 1600 excitatory neurons without additional training data. That comparison does not isolate network size from data exposure per neuron. The v3 run also changed several variables at once, including data volume and presentation time, and reached 64.6%. Relative to v3, the v4 inhibition change raised accuracy to 66.0% and produced a more balanced neuron-label distribution. These runs identify competition and plasticity calibration as variables for controlled follow-up experiments; they do not isolate the remaining error source.

Timesteps alone do not provide a meaningful learning-efficiency comparison across these runs because network size, training data, and presentation length also changed. Accuracy, total timesteps, and wall time are therefore reported separately.

v3 neuron label distribution: 384/400 neurons labeled. Digit 1 accounts for 179 neurons, or 47% of the labeled population, while classes 3, 5, and 8 each have fewer than 10 neurons. This imbalance coincides with strong per-class accuracy on easy digits and near-chance accuracy on harder ones, but the run does not establish it as the sole cause.

v4 neuron label distribution: 398/400 neurons labeled, with a more even class spread:

- `0:58`, `1:41`, `2:39`, `3:41`, `4:32`
- `5:25`, `6:34`, `7:56`, `8:32`, `9:40`

Compared with v3, v4 reduced the concentration on digit 1 and improved accuracy by 1.4 percentage points. The result supports keeping the revised inhibition setting while testing plasticity calibration, but it does not identify a single residual bottleneck.

Partial diagnosis: readout-only stress test (reduced protocol: `100/class` train, `20/class` test, `100ms` train, `25ms` rest):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `baseline_50x2_blend` | **48.0%** | `0.972` | `0.192` | Baseline diagnostic run |
| `matched_100x2_blend` | **47.0%** | `0.963` | `0.199` | Matching readout window to training does not help |
| `matched_100x5_blend` | **48.5%** | `0.963` | `0.199` | More test repeats add little |

Interpretation: readout changes moved accuracy by only ±1.5 points on the reduced protocol, while neuron-label balance stayed similar. Within this protocol, the result gives more reason to test competition strength, STDP calibration, and adaptive-threshold behavior than to expand `build_response_templates()` or `predict_sample()`.

Partial diagnosis: competition sweep (same reduced protocol):

| `INH_LATERAL_WEIGHT` | Accuracy | Label entropy | Max class share | Takeaway |
|----------------------|----------|---------------|-----------------|----------|
| `6.0` | **54.5%** | `0.957` | `0.185` | Clear gain over the readout baseline |
| `8.0` | **52.0%** | `0.958` | `0.225` | Worse than both `6.0` and `10.0+` |
| `10.0` | **53.0%** | `0.982` | `0.155` | Best balance, moderate accuracy |
| `12.0` | **59.0%** | `0.971` | `0.157` | Best overall short-run result so far |

Interpretation: `INH_LATERAL_WEIGHT=12.0` improved reduced-protocol accuracy from **48.0%** to **59.0%** without changing the decoder or STDP rule. In this experiment, competition dynamics had a larger measured effect than the tested readout variants.

Follow-up validation: larger reduced protocol (`300/class` train, `30/class` test, `100ms` train, `25ms` rest):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `inh_lateral=10.0_bigger` | **63.0%** | `0.888` | `0.280` | Better than the small-protocol baseline, but class balance degrades |
| `inh_lateral=12.0_bigger` | **64.3%** | `0.928` | `0.213` | Best larger-protocol point so far; wins on both accuracy and balance |

Interpretation: on the larger reduced protocol, `INH_LATERAL_WEIGHT=12.0` improved accuracy and neuron-label balance relative to `10.0`. It was therefore selected for the next full-MNIST run.

Partial diagnosis: STDP scale sweep (interrupted after first point):

| `STDP_SCALE` | Accuracy | Label entropy | Max class share | Takeaway |
|--------------|----------|---------------|-----------------|----------|
| `0.10` | **47.5%** | `0.959` | `0.206` | Lower than the readout baseline; not promising |

Interpretation: the only completed point, `STDP_SCALE=0.10`, reached **47.5%** and did not recover accuracy. No conclusion can yet be drawn about the untested `0.2-0.4` range.

Partial diagnosis: theta / homeostasis sweep (on top of `INH_LATERAL_WEIGHT=12.0`, larger reduced protocol):

| Variant | Accuracy | Label entropy | Max class share | Takeaway |
|---------|----------|---------------|-----------------|----------|
| `theta_default_12` | **67.0%** | `0.882` | `0.269` | Best accuracy overall |
| `theta_low_plus_12` | **65.7%** | `0.927` | `0.221` | Better balance, but lower accuracy |
| `theta_high_plus_12` | **56.0%** | `0.924` | `0.251` | Clearly harmful |
| `theta_fast_leak_12` | **57.3%** | `0.974` | `0.171` | Very balanced, but too much accuracy loss |

Historical interpretation: with inhibition set to `12.0`, the default theta regime produced the highest test accuracy among the tested variants. Some alternatives improved class balance but reduced accuracy. That pre-fix selection was:

- `INH_LATERAL_WEIGHT = 12.0`
- `THETA_PLUS = 0.10`
- `THETA_LEAK = 0.005`
- `STDP_SCALE = 0.2` (still provisional; only `0.10` has been tested and it underperformed)

Full-MNIST confirmation run (same `6000/class`, `200ms`, `1 epoch`, but with `INH_LATERAL_WEIGHT=12.0`):

- **Accuracy:** `66.0%`
- **Training time:** `49346.6s` (~`13.7h`)
- **Readout build:** `1140.2s`
- **Eval time:** `111.9s`
- **Total wall time:** `50607s`

Interpretation: the full benchmark moved from **64.6%** to **66.0%** after changing `INH_LATERAL_WEIGHT` to `12.0`. This 1.4-point gain leaves the result below the 75-85% target. The next experiments keep readout and competition fixed while varying plasticity.

Architecture (Diehl & Cook 2015):
- `784` input neurons (one per pixel, rate coded)
- excitatory + inhibitory cortex neurons with 1:1 matched WTA microcircuit
- Unsupervised STDP on feedforward `input->cortex` pathway
- Adaptive thresholds (leaky theta), per-neuron weight normalization, blended spike/voltage readout

Observed runtimes (MacBook Air M2, CPU):
- **400 exc, 25ms**: `3000` train, `3` epochs, ~70 ms/sample, **17 min** total
- **1600 exc, 25ms**: `3000` train, `3` epochs, ~250 ms/sample, **63 min** total
- **400 exc, 200ms** (v3): `60000` train, `1` epoch, ~357 ms/sample, **5.95h** total (incl. readout + eval)

GPU and compilation experiments:

Several approaches to GPU acceleration were tested and none improved over CPU scatter/gather:

| Approach | Result | Why |
|----------|--------|-----|
| MPS direct (400 exc) | **16x slower** than CPU | Kernel launch overhead dominates small tensor ops |
| MPS direct (1600 exc) | **2.4x slower** than CPU | Gap narrows with scale but still loses |
| `torch.compile` | No speedup, falls back to eager | `torch.where` returns variable-length outputs (graph breaks); ring buffer slot indexing triggers recompilation every step |
| Masked full-tensor (no `torch.where`) | **6.7x slower** on CPU, **2.1x slower** on MPS | Multiplying all 2.9M synapses when only 1-5% are active wastes compute; the "zero work" is not free |

The tested workloads favor CPU execution because activity is sparse. The current scatter/gather path (`torch.where` → gather active → `index_add_`) touches O(active) elements per step, roughly 30-150k out of 2.9M. The full-tensor alternatives tested above process every element and were slower at these scales.

Approaches not yet tested and their trade-offs:
- **Sparse matrix-vector multiply** (`W_sparse @ fired`): this would replace the propagation block with one operation, but STP modifies effective weights every step and may require rebuilding the matrix each time.
- **Batched sample presentation**: processing B images on `(B, N)` state tensors could improve GPU utilization, but it changes STDP semantics and requires batch-aware homeostasis, theta, STDP, and memory implementations. This refactor is more relevant to hyperparameter sweeps than to a single accuracy run.

CPU with sparse scatter/gather remains the default for the tested scales.

Observations from completed runs:
- Feedforward weights saturated within 1 epoch (`mean=1.20, max=10.0` after every epoch) for both 400- and 1600-neuron configurations. Additional epochs over the same data did not improve the result.
- The 1600-neuron run scored 51.8%, compared with 62.2% for 400 neurons. Of the 1157 labeled neurons, 480 (41%) were assigned to digit 1, while classes 4 and 8 received 52 neurons each. The observation is consistent with too little data per neuron, but the run did not isolate that factor.
- The leaky theta update kept the adaptive-threshold regime usable in the completed runs. A dedicated ablation is still needed to measure its effect.

Changes applied for v3 run (all simultaneously):

| Change | From | To | Rationale |
|--------|------|----|-----------|
| `TRAIN_PER_CLASS` | 300 | 6000 (full MNIST) | Data diversity is the primary learning driver |
| `TRAIN_PRESENT_STEPS` | 25 | 200 | More time for spike-pair coincidences to accumulate |
| `EPOCHS` | 3 | 1 | Extra epochs add nothing (weights saturate in one pass) |
| `REST_STEPS` | 5 | 50 | Longer rest between samples lets theta decay properly |
| `A_minus/A_plus` | 1.20 | 1.05 | Paper value; 1.2 over-prunes weak classes |
| `STDP_SCALE` | 0.8 | 0.2 | Compensates 8x more STDP events per sample at 200ms |

Corrections applied before the next benchmark run:

| Priority | Change | Reason |
|----------|--------|--------|
| 1 | Fix activity-rate units in homeostatic scaling | Prevents the WTA circuit from being driven toward weight bounds by a `~200x` unit mismatch |
| 2 | Freeze scaling independently from adaptive theta | Keeps the intended competition circuit fixed without removing neuron-level adaptation |
| 3 | Enforce weight bounds at creation | Removes the invalid `-12` initialization under a `-10` minimum |
| 4 | Restore canonical data isolation | Prevents test leakage and train/test-dependent preprocessing |

After revalidation, any STDP, inhibition, replay, or longer-presentation sweep should change one variable at a time and use the same fixed evaluation set.

Target: 75-85% with 400 exc neurons at 200ms (paper: 87% at 350ms). Historical best: **66.0%**; current corrected result pending.

## Literature pointers

These papers cover mechanisms represented in the current MNIST experiments:

- **Zhuang et al., 2023** — [An unsupervised STDP-based spiking neural network inspired by biologically plausible learning rules and connections](https://www.sciencedirect.com/science/article/pii/S0893608023003301): combines adaptive synaptic filtering, adaptive threshold balancing, adaptive lateral inhibition, and temporal-batch STDP. The article reports **97.9% on MNIST** and **87.0% on CIFAR-10**. For this repository, the relevant design difference is the combination of adaptive inhibition, thresholding, filtering, and STDP rather than any single component.
- **Wu et al., 2024** — [Inhibition SNN: unveiling the efficacy of various lateral inhibition learning in image pattern recognition](https://link.springer.com/article/10.1007/s42452-024-06332-z): shows that **inhibition architecture itself** matters, not just inhibition strength. Their simplified inhibition design reports **86% on MNIST** with an unsupervised `784-100` SNN.
- **Arefnadia et al., 2025** — [Unsupervised post-training learning in spiking neural networks](https://www.nature.com/articles/s41598-025-01749-x): argues that a trained SNN should not necessarily be frozen after the main STDP phase. They combine **triplet STDP** during training with **post-training STP** to improve recognition without changing long-term weights.
- **Khan et al., 2025 review** — [Modulated spike-time dependent plasticity (STDP)-based learning for spiking neural network (SNN): A review](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019416): emphasizes recurring practical bottlenecks in SNN classification, especially **threshold regulation, competition control, parameter optimization, and scalability**.
- **Zhang et al., 2025** — [Biologically plausible unsupervised learning for self-organizing spiking neural networks with dendritic computation](https://www.sciencedirect.com/science/article/pii/S0925231225003790): combines a multi-compartment neuron model, adaptive self-organizing inhibition, and partially shared connections. The inhibition strategy groups neurons that encode similar features.

## Next research measurements

The following measurements would test claims that classification accuracy alone cannot answer:

### Accuracy improvements

- **MNIST accuracy push** — the immediate priority (see roadmap above)

### Qualitative advantages over conventional ML

These tests compare online adaptation, retention, structural plasticity, and damage tolerance against explicit baselines. Their outcomes should be reported as measurements rather than treated as advantages in advance.

| Test | What it measures | Protocol sketch | What a positive result looks like |
|------|-----------------|-----------------|-----------------------------------|
| **Few-shot learning curves** | How quickly the network learns from limited data | Train on 1, 5, 10, 50 samples/class; compare accuracy vs a simple MLP baseline with the same data budget | SNN reaches usable accuracy with fewer samples than the MLP, suggesting local plasticity extracts more from each example |
| **Catastrophic forgetting** | Whether sequential task learning destroys previous knowledge | Train on MNIST digits 0-4, then train on 5-9, then re-test on 0-4 | SNN retains significant accuracy on 0-4 after learning 5-9; MLP baseline drops to chance |
| **Sleep / replay benefit** | Whether offline consolidation improves retention | Compare test accuracy with and without a silent replay phase between training blocks | Post-replay accuracy is measurably higher than without replay, validating the consolidation machinery |
| **Growth vs fixed-size** | Whether neurogenesis and synaptogenesis improve learning | Run identical tasks with growth enabled vs disabled (fixed topology) | Growth-enabled networks reach higher accuracy or learn faster, justifying the structural plasticity overhead |
| **Graceful degradation** | Whether the network tolerates damage better than conventional models | After training, randomly kill 5%, 10%, 20% of neurons; measure accuracy drop vs an MLP with the same fraction of weights zeroed | SNN degrades more gracefully (smaller accuracy drop per % of damage) |
| **Temporal pattern recognition** | Whether native spike timing gives an advantage on time-domain tasks | Classify simple temporal patterns (e.g. spike sequences, rhythmic signals) where input order matters, not just content | SNN outperforms a rate-based MLP that receives the same inputs as static vectors |
| **Online adaptation** | Whether the network adapts to distribution shift without retraining | Train on one distribution, then shift (e.g. rotated MNIST digits); measure how quickly accuracy recovers with continued exposure | SNN recovers accuracy through ongoing plasticity while a frozen MLP cannot adapt |
