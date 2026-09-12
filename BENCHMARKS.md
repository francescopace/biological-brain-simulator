# Benchmark Notes

These diagnostic benchmarks test whether biologically motivated local learning rules support useful behavior on controlled AI tasks. They are not leaderboard submissions.

The measurements below come from successive protocols. Keep their evaluation rules attached to each result: the batch in `results/20260911-190006` predates the independent-image evaluation patch described here.

## Independent-image evaluation and controlled sleep

MNIST readout fitting and evaluation now reset membrane voltage, recovery state, delayed currents, spike history, activity and short-term synaptic resources before every simulated image presentation. The copies retain trained weights and theta values; topology, learning, theta adaptation, memory, encoder noise and oscillations are disabled during inference. Identical test repeats can reuse the first response under these conditions. Training still uses its existing presentation and rest schedule. Few-shot evaluation now uses the same frozen-copy helpers.

On the saved reduced checkpoint, reversing test order changed 129 of 200 predictions under the previous protocol. With independent images, none changed. The accuracy comparison on the same 300 readout and 200 test samples was:

| Evaluation intervention | Accuracy |
|---|---:|
| Previous sequential-state protocol | 28.0% |
| Zero theta only | 24.5% |
| Independent images, retained theta | 25.0% |
| Independent images, zero theta | 24.0% |
| 5,000 quiet steps, frozen theta and weights | 29.5% |
| 5,000 quiet steps, adapting theta and frozen weights | 22.5% |

These are diagnostic comparisons on reused test data, not held-out selection scores. They establish order invariance under the new policy, not recovery of historical accuracy. The six-condition diagnostic retains the old behavior locally through `INDEPENDENT_INFERENCE=False`, so its baseline remains reproducible. Raw results are in `results/20260911-230336-state-diagnosis/summary.json`.

The sleep weight audit found that the old no-replay phase changed 159,979 of 160,000 cortical weights through homeostatic scaling: mean inhibition moved from -10 to about -1.33. Freezing scaling prevented those changes. The updated sleep comparison therefore uses four arms, replay on/off crossed with manual feedforward STDP on/off. Every arm starts from an identical pre-sleep copy and uses one unchanged pre-sleep decoder. Scaling, theta adaptation, growth and R-STDP are disabled during sleep; stored traces can replay, but new traces are not captured. Training settings remain unchanged. The output records actual weight changes and replay counts, and rejects updates outside the allowed feedforward pathway. Single-seed differences do not establish statistical significance.

The `--compare-initial` option of `python -m examples.mnist_state_diagnosis` compares an initialized network with the saved trained checkpoint using the standard independent-image policy and the same labelled readout budget. This measures the training pipeline, including normalization and retained theta, rather than STDP alone. It does not retrain either network.

That comparison returned 22.5% for the initial network and 25.0% for the trained checkpoint: five additional correct predictions out of 200. The trained network again had zero prediction changes under reversed test order. The observed 2.5-point gain is small and comes from one seed on reused diagnostic data; it does not establish a reliable learning advantage. Results are in `results/20260911-230336-state-diagnosis/learning_check.json`. The updated four-arm sleep protocol has been checked with synthetic regression tests, but has not yet been rerun on MNIST.

## Simulation shortcuts with unchanged inference policy

The independent-image path collects readout responses once for both neuron labels and class templates, omits quiet steps before transient resets, and simulates each deterministic test image once. It retains the requested repeat count in score accumulation, including float32 addition order. `FAST_INDEPENDENT_INFERENCE=False` restores the unoptimized independent schedule for comparisons; historical state-diagnosis modes still retain their sequential schedule. Shortcuts require a frozen network with noise, oscillations and memory disabled.

For the reduced protocol's 300 readout and 200 test images, this removes 50,000 of 75,000 simulation steps. With the default 50-step rest, readout and evaluation use 75% fewer steps. Neither percentage applies to training or directly predicts total wall time. Inference-copy clocks, accumulated counters and RNG consumption differ because fewer steps are simulated; source networks and the response/decoder contract are preserved.

Training now converts each input to a float32 tensor once per presentation and skips unused response collection. The encoder still draws fresh noise at every timestep, and training retains normalization, STDP and rest steps. The simulator also avoids the auto-capture activity reduction when memory is disabled. Synthetic tests compare all neuron/synapse tensors and RNG state against the original training loop, and compare inference templates, scores and predictions exactly for 1, 2, 3 and 5 repeats.

`python -m examples.mnist_optimization_check` checks both paths on a saved network without replacing it. It requires `--checkpoint`, `--reference` and a new `--output` directory, and accepts `--learning-reference` to check predictions against an earlier diagnostic. Its output includes source hashes, thread count, simulated steps, separate inference/training timings and a short training profile. Timing runs made alongside a full benchmark share host load and are not isolated throughput measurements.

The saved reduced checkpoint passed exact comparisons of neuron labels, templates, scores and all 200 predictions. Accuracy remained 25%, matching the pre-optimization independent-image diagnostic. Readout plus evaluation fell from 116.5 to 42.1 seconds (2.77x) as simulated steps fell from 75,000 to 25,000. This check ran with four CPU threads alongside full MNIST, at reduced scheduling priority. The short ABBA training comparison returned a 0.99x timing ratio, so it does not demonstrate a training speedup; neuron/synapse state and RNG hashes matched in all four trials. Results and the training profile are in `results/20260912-optimization-check`.

A CPU thread check on 20-sample training copies tested 1, 2 and 4 threads in forward and reverse order, warming each thread pool first. Median times were 5.597, 5.060 and 5.044 seconds, respectively, with identical state/RNG hashes. Under this concurrent load, one thread was slower and two threads offered no useful advantage over four. No thread default was changed. Run this check with `--thread-sweep-only --training-samples 20`; its results are in `results/20260912-thread-check`.

## Checked CPU synapse indices

Propagation and STDP now use a checked adjacency index for CPU pathways with at least 2,048 synapses and sparse neuron activity. The index returns live synapse positions in their original storage order, preserving floating-point accumulation and LTP/LTD update order. Small pathways, dense activity and non-CPU devices retain the PyTorch scan. STDP converts only selected endpoint indices to int64 on the indexed path.

Endpoint contents are compared with the cached snapshot on every lookup. This remains an O(number of synapses) validity check, but avoids the more expensive gather/mask/nonzero sequence over all connections. It catches edits through PyTorch, NumPy views and `Tensor.data`; changes to the alive mask take effect without rebuilding adjacency. A shorter active-neuron view can reuse a table built for the region's full capacity. Derived indices are rebuilt after deepcopy and load, and are not saved as learned state.

On 800 recorded MNIST spike frames, the checked selector was 7.2x faster for cortical propagation, 9.4x for feedforward propagation/LTD and 5.6x for feedforward LTP. Every selected index and its order matched the scan. These are selector timings, not whole-simulator speedups; results are in `results/20260912-checked-synapse-kernels.json`. The tests also cover growth, pruning, direct endpoint edits, negative/invalid indices, weight clipping, reward restrictions and save/load continuation.

Use `python -m examples.mnist_optimization_check --event-index-check` with the checkpoint/reference/output arguments to compare indexed and dense events. Both arms use the same optimized independent-image schedule. The ABBA training trials include lazy index construction and compare all recorded neuron/synapse tensors, subsystem counters and RNG state exactly.

The final checked-index run reduced median time for 30 training samples from 7.461 to 4.425 seconds (1.69x). Readout plus evaluation fell from 40.159 to 29.211 seconds (1.37x), with 25,000 simulated steps in both arms. All four training-state/RNG hashes matched; neuron labels, templates, scores and the 200 predictions matched exactly, including the earlier independent-image diagnostic. Accuracy stayed at 25%. Results are in `results/20260912-event-index-final`; a preceding implementation returned similar gains in `results/20260912-event-index-check`. Both comparisons used four CPU threads alongside the original full benchmark at reduced scheduling priority. These checks establish a runtime improvement on this workload, not an accuracy improvement or a universal speedup across devices and network sizes.

## Train-only learning controls and numerical diagnosis

`python -m examples.mnist_learning_check --output results/NEW_RUN` compares initialization, normalization without STDP, and STDP plus normalization. The default screening protocol uses 200 training images, 100 labelled readout images drawn from that training set, and 200 separate validation images. All rows come from the canonical 60,000-image training partition; intensity equalization is fitted before validation is transformed. Seeds 101, 102 and 103 share the same split. Each paired condition starts from an identical initialized network and uses the same sample order, noise schedule and adaptive-threshold policy. No canonical test accuracy is computed.

The first run, in `results/20260912-learning-controls`, returned these mean validation accuracies:

| Condition | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---:|---:|---:|
| Initial network | 19.50% | 20.83% | 31.00% |
| Normalization only | 19.17% | 19.50% | 30.50% |
| STDP + normalization | 20.17% | 20.67% | 31.50% |

The paired template gains from STDP were +2, +3 and -2 percentage points. Their mean is +1 point, a small and inconsistent gain on this screening split. Ridge uses a fixed alpha of 1.0; its feature scaler and classifier are fitted only on the same labelled readout rows. Ridge on the image pixels reached 64.5% with that label budget. The neural decoder improvement therefore does not demonstrate useful STDP learning, and neither decoder establishes a held-out test result.

The study saves per-image weight deltas before STDP, after STDP and after normalization, plus cached readout/validation responses. Checkpoints contain the brain, initial normalization target, sample cursor, protocol/data hashes and diagnostics. They are published in new directories at completed image boundaries and never replace an earlier checkpoint. `--resume` requires the original output directory and arguments, unchanged source/dependency versions and unchanged data; completed conditions are skipped. The source fingerprint includes the shared quiet-step helper and imported diagnostic helpers, which the earlier explicit file list omitted. A regression test rejects a simulated helper-source change without touching workspace files. Tests compare uninterrupted and resumed training exactly, including an interruption in the middle of a CLI condition. This resumable loop currently belongs to the learning-check script, not the older full MNIST process.

The learning results exposed a numerical problem in the former integrator. At `dt=1ms`, its two-half-step Euler integration produces 100 spikes in 100ms from a regular-spiking neuron under constant input of -300 or -1000. Running the same neuron at `dt=0.05ms` produces none. At input -200, the coarse solution also oscillates away from the finer solution despite emitting no spikes.

`examples.mnist_numerics_check` replayed eight validation images on the saved seed-101 initial network. It observed 7,923 excitatory spikes, of which 6,427 occurred with net input below -200. Reintegrating those individual steps from their recorded starting states with 0.02ms and 0.01ms reference steps failed to reproduce 6,045 spikes in both cases. These are one-step probes along the coarse trajectory, not a full higher-resolution accuracy comparison. The source checkpoint was unchanged. Results are in `results/20260912-inhibitory-numerics.json`.

The same diagnostic found a wiring discrepancy: the seed-101 model has 47,066 active input-to-inhibitory connections out of 94,343 feedforward synapses. Their plasticity is disabled, but they still transmit current. The benchmark describes a Diehl–Cook-style circuit, whose input projects to excitatory neurons and whose inhibition is driven through matched excitatory-to-inhibitory pairs. [Diehl and Cook, section 2.2](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full).

## Stable integration and excitatory-only input

New networks now use Heun integration, with internal steps capped at `0.1ms` and reduced further when the voltage dynamics require it. The bound uses the lower voltage equilibrium under the current input and an upper bound on recovery state. It prevents a strong inhibitory input from driving an unstable Euler excursion. The tests compare all seven firing patterns against an independent, high-accuracy DOP853 solution, check convergence as the step cap decreases, and verify the absence of inhibition-induced spikes across 100ms. The bound requires nonnegative `a` and `b`, as used by all built-in patterns; unsupported parameters and non-finite state fail explicitly.

This correction changes trajectories intentionally. It does not change the external network timestep or make other simulator mechanisms timestep-invariant. Each neuron emits at most one spike per external step. The integrator retains the first threshold crossing, estimates recovery at that crossing, and waits for the end-of-step reset; it does not resolve multiple spikes within a network step. The smaller internal steps add computation. Earlier exact-equivalence runtime measurements describe legacy dynamics, not the cost of this correction.

Version 6 checkpoints save `integration_method` and `integration_max_step` on the brain and each region. Loading version 4/5 files without those fields selects `legacy_euler`, preserving historical continuation. Loading a checkpoint does not migrate its dynamics or remove connections. New MNIST models instead allocate feedforward connections to living excitatory targets only, through `Brain.connect_regions(..., target_neuron_type=NeuronType.EXCITATORY)`. The general connection API retains its all-target default. Explicit `build_brain(integration_method="legacy_euler", feedforward_exc_only=False)` reconstructs the old MNIST circuit.

`python -m examples.mnist_dynamics_check --checkpoint PATH --study STUDY --output NEW_DIR --check-refinement` separates the integration and wiring changes on copies of one saved model. It retains identical excitatory weights, disables existing inhibitory-target connections only in the wiring arm, and fits each decoder on the same labelled rows. It uses the learning study's disjoint, train-only validation split and checks a halved step cap. No network is trained, no test partition is used, and the source checkpoint is hashed before and after. Disabled connections still occupy storage in this intervention, so its timings do not measure the compact new builder.

The seed-101 initial checkpoint produced the following results on the same 100 labelled readout and 200 validation images:

| Integrator | Input targets | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---|---:|---:|---:|
| Legacy Euler | Excitatory and inhibitory | 21.5% | 18.5% | 32.5% |
| Heun, maximum 0.1ms | Excitatory and inhibitory | 44.0% | 25.5% | 42.5% |
| Legacy Euler | Excitatory only | 70.0% | 64.5% | 68.0% |
| Heun, maximum 0.1ms | Excitatory only | 69.0% | 71.5% | 72.5% |
| Heun, maximum 0.05ms | Excitatory only | 70.0% | 71.5% | 72.0% |

The legacy/all-target arm reproduced every earlier decoder prediction. The corrected integration removed the diagnosed inhibitory artifacts on the eight probed images even when the old wiring was retained: none of its excitatory spikes occurred at net input below -200, compared with 6,427 in the legacy arm. The excitatory-only arms also produced none in that current range. These counts address the identified failure mode, not every possible numerical error.

Halving the step cap changed 3 of 200 template predictions, 9 spike-ridge predictions and 10 spike-plus-voltage ridge predictions. Only 62 of 80,000 excitatory spike-count entries changed, each by one spike; the mean absolute change in average-voltage features was 0.027mV. Similar aggregate accuracies therefore do not mean identical trajectories or fully converged predictions. This is a single initialized network, with no STDP training in any arm. It supports correcting the dynamics and wiring, but does not establish a learning gain or held-out test accuracy. The new builder also draws a different compact topology, so these intervention scores must not be presented as its measured accuracy. Results are in `results/20260912-dynamics-check`.

## CPU loop for the corrected integrator

The Heun loop now uses NumPy for compatible float32/float64 CPU arrays, retaining the PyTorch reference for other inputs. Both paths enforce the same validation rules and share the stability-bound and substep-count calculations. The CPU loop keeps the same operation order without fused arithmetic. `src.integration.CPU_HEUN_ENABLED=False` selects the reference for comparisons. This switch changes execution only; it does not select different dynamics or require a checkpoint migration.

`python -m examples.heun_kernel_check --study STUDY --output NEW_DIR` compares the two implementations on newly built, compact excitatory-only MNIST networks. The final run used 20 training images in ABBA order and the study's 100 readout/200 validation images for inference. Median training time fell from 6.271 to 4.364 seconds (1.44x). Inference fell from 40.122 to 25.758 seconds (1.56x). All four complete training-state/RNG hashes matched, and every readout/validation spike count, voltage feature and decoder prediction matched. Source hashes were verified against the files at completion. The measurements ran at reduced priority alongside the original full benchmark, so they remain workload- and host-load-specific. Results are in `results/20260912-heun-kernel-final`; the preliminary run is in `results/20260912-heun-kernel-check`.

The newly constructed seed-101 network scored 71.0% with templates, 64.5% with spike ridge and 67.5% with spike-plus-voltage ridge before STDP training. These are separate from the retained-topology intervention scores above. They use the same small validation split; the paired multi-seed learning check follows below.

The CPU loop also skips threshold interpolation when no new neuron crosses, skips frozen-state masking before the first crossing, and stops internal integration when every neuron is already frozen. The external network step, spike counters, delays and plasticity still run normally. `python -m examples.heun_sparse_check --study STUDY --output NEW_DIR` retains the unconditional NumPy loop as its reference. In the final ABBA check, median time for 20 training images fell from 3.419 to 3.117 seconds (1.10x); response extraction for 40 validation images improved by 1.11x. Training-state/RNG hashes and all response bytes matched. Results are in `results/20260912-heun-sparse-final`. The old full benchmark finished during this stage, so compare the arms within each run rather than comparing their absolute times with earlier measurements.

`Region.step` now skips housekeeping for regions with no internal synapses and skips propagation when no live outgoing synapse is selected. Neuron integration, delayed delivery, counters and existing synapse recovery still advance; endpoint validation runs even when no neuron fires. The same ABBA command with `--region-step` compares this path against the retained pre-shortcut step. Median training time fell from 3.110 to 2.849 seconds (1.09x), with a 1.09x inference gain. All four training-state/RNG hashes and every response byte matched. Results are in `results/20260912-region-empty-check`; these gains are additional to the earlier Heun changes, not a new estimate of their combined speedup.

A repeat after extending normalization reproduced the same previously recorded training-state/RNG hash (`2811e313...adc35`) in all four arms and matched every response byte. Scalar normalization retains Python-scalar/Tensor division explicitly: substituting zero-dimensional Tensor/Tensor division changes rounding. This final equivalence check is in `results/20260912-region-empty-final`; its timings overlapped the test suite, so the preceding run remains the timing reference. The full suite passed with 460 tests and two device-dependent skips.

Synaptic recovery now uses one shared helper in regions and projections. Native PyTorch in-place operations avoid temporary arrays and redundant slice assignments. Its 33 targeted tests cover floating-point dtypes, strided views, shared arrays, unused capacity, integer-age overflow, non-finite values, custom rates and the original integer-resource/autograd fallback. The isolated prototype's ABBA kernel check improved by 1.21x on both 47,000- and 160,000-synapse arrays, with identical final bytes. Those measurements shared the host with the WTA replication and are not whole-simulator speedups; results are in `results/20260912-housekeeping-prototype.json`.

After the replication finished, `python -m examples.heun_sparse_check --housekeeping --study STUDY --output NEW_DIR` compared the integrated helper against the original updates. On the corrected weak-coupling network, median training time for 20 images fell from 2.906 to 2.806 seconds (1.036x), and inference improved by 1.033x. Every response byte and all four training-state/RNG hashes matched, including the previously recorded training hash. Results are in `results/20260912-housekeeping-end-to-end`.

With active inhibitory coupling at 64, the same check reduced median training time from 2.871 to 2.766 seconds (1.038x), with a 1.028x inference gain. All four training-state/RNG hashes and response bytes matched again. Results are in `results/20260912-housekeeping-wta64`. Both end-to-end checks ran after the learning jobs finished; the full test suite passed with 517 tests and two device-dependent skips.

### CPU integration preflight

For compatible nonempty CPU float32/float64 arrays without gradients, integration now checks finiteness and nonnegative `a`/`b` through read-only NumPy views. The CPU kernel already copies voltage and recovery, so the caller no longer makes a second pair of copies. Stability-bound arithmetic and internal steps are unchanged. Empty, scalar, mixed-dtype, autograd and device fallback cases retain the original checks. The retained entry point in `examples/heun_preflight_reference.py` supports an end-to-end comparison through `python -m examples.heun_sparse_check --preflight --study STUDY --output NEW_DIR`.

The new regression tests also exposed an existing backward-pass error: the PyTorch loop updated the spike mask in place after `torch.where` had saved it for differentiation. Assigning a fresh mask preserves forward values and allows backward to run. Finite-difference gradient checks cover a non-crossing case, a threshold-crossing case and a neuron already above threshold. This does not make discrete spike decisions differentiable at their boundaries or establish end-to-end differentiation through `Brain.step`.

The completed ABBA comparison at coupling 64 reduced median training time from 2.809 to 2.678 seconds (1.049x), with a 1.043x inference gain. The weak-coupling check, using the `results/20260912-learning-1000` study, reduced training from 2.827 to 2.694 seconds (1.049x), with a 1.049x inference gain. Both used 20 training images and 40 validation images for response extraction. Within each comparison, all four complete training-state/RNG hashes matched and every response byte was identical. Timing runs followed the learning jobs and test suite, without another benchmark launched by this task concurrently. These are incremental gains over the preceding CPU implementation, not a measured cumulative speedup. Results are in `results/20260912-preflight-wta64` and `results/20260912-preflight-default`.

The final suite passed with 590 tests and two device-dependent skips. It includes 46 preflight/gradient checks and 12 separate-arm audit checks.

## MPS regression checks

On the macOS 26.6.2 arm64 host with PyTorch 2.12.0, `torch.backends.mps.is_available()` returned false inside the sandbox and true outside it. Run GPU checks from a process with access to Metal:

```bash
BRAIN_DEVICE=mps python -m pytest -q -ra
```

The first full MPS-configured suite finished with 655 passed, 97 failed, four setup errors and one CUDA skip. Most failures came from restoring regional RNG state after moving it to MPS, requesting unsupported float64 tensors in diagnostics, or mixing CPU inputs with accelerator state. Some tests also assumed CPU-only event caches, NumPy views or exception classes.

Checkpoint loading restores generator state from a CPU ByteTensor. Diagnostic reductions use CPU float64 while retaining the original order of subtraction and conversion; simulation state stays on its configured device. Batched synapse creation moves input tensors to the region's device, and memory replay and apoptosis align indices and masks with the tensors they access. Tests check CPU event caches and the accelerator fallback separately, retaining exact-state and continuation checks. Three additional regressions cover CPU float64 synapse inputs, mixed-device memory traces and diagnostic precision without source mutation.

After these corrections, the full CPU-configured suite passed 758 tests in 36.57 seconds, with the two accelerator probes skipped inside the sandbox. The full MPS-configured suite passed 759 tests in 668.81 seconds outside it, skipping only CUDA. Neither run had failures or setup errors. Both collected the same 760 cases, including explicit CPU kernel reference tests. The 182 targeted MPS checks also passed. Reports are in `results/mps-validation.ZyNJLm/cpu-after-fixes.xml`, `mps-after-fixes.xml` and `targeted-after-fixes.xml`; `report.xml` records the original failures.

These are software regression checks. They do not establish equal trajectories between CPU and MPS, GPU speedups, or new benchmark accuracy results. Test-suite elapsed times are not paired performance measurements. CPU remains the default backend.

## CPU versus MPS after the fixes

The bounded comparison on an Apple M2 with 16 GiB RAM, macOS 26.6.2 and PyTorch 2.12.0 favors CPU for the current MNIST learning studies. It uses the corrected Heun dynamics from commit `c049515`, 1,584 neurons, 207,254 synapses and excitatory-to-inhibitory coupling 64. Each trial processes ten training images at 100 presentation steps plus 25 rest steps, then ten independent inference images at 50 steps. These are the reduced learning-study settings, not the full benchmark's 200-step training schedule.

All four processes ran outside the sandbox, sequentially in CPU/MPS/MPS/CPU order, with four CPU threads. Each warmed a disposable training and inference copy before timing. MPS was synchronized before and after each measured segment. Setup, model copies and diagnostics were excluded; training includes STDP, normalization and rest, while inference includes response transfer to CPU. Other applications were active, so the host was not isolated.

| Workload, ten images | CPU median (range) | MPS median (range) | MPS / CPU |
|---|---:|---:|---:|
| Training | 1.411 s (1.385–1.437) | 43.419 s (42.663–44.175) | 30.77x |
| Inference | 0.526 s (0.524–0.529) | 15.147 s (14.414–15.880) | 28.79x |

The fixture hashes verify identical initial neuron/synapse tensors and images. Training retains native backend noise at level 0.02, so equal seeds do not imply equal noise draws. Each inference trial instead starts from the common initialized network with learning and noise disabled, not from independently trained weights. All 4,000 excitatory spike-count entries matched across the four inference trials. The largest cross-backend mean-voltage difference was 0.0000458 mV. CPU state and response hashes repeated exactly; MPS hashes did not, including small voltage differences between its two trials. These checks do not establish long-run accuracy or bitwise backend equivalence.

Use `python -m examples.device_timing_check --prepare --output results/NEW_RUN` with `BRAIN_DEVICE=cpu` to create a local fixture. Then run `--trial NAME --output results/NEW_RUN` in separate processes with `BRAIN_DEVICE=cpu` or `BRAIN_DEVICE=mps`, following the same alternating order. Only load fixtures generated locally by this script. Three harness tests passed on each backend. Raw trials, input-row IDs, source hashes and the aggregate report are in `results/20260912-cpu-mps-timing-01/summary.json` and its sibling files.

Keep CPU for the next controlled learning runs. MPS now passes the software checks, but this workload gains no throughput from it. Larger networks, batched execution and trained-network inference would need separate measurements; this short comparison does not justify a full-run duration forecast.

## Learning controls after the dynamics correction

The three-seed study was repeated with Heun integration and newly generated excitatory-only feedforward wiring. It retained 200 training images, 100 labelled readout images and the same 200 train-only validation images. Mean accuracies were:

| Condition | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---:|---:|---:|
| Initial network | 69.83% | 61.17% | 63.83% |
| Normalization only | 69.33% | 60.83% | 64.50% |
| STDP + normalization | 70.33% | 60.33% | 65.00% |

STDP added exactly two correct template predictions out of 200 for each seed: +1 percentage point over normalization alone. Spike-ridge gains were +2.5, -0.5 and -3.5 points; spike-plus-voltage gains were -1.5, +2 and +1 points. The large recovery relative to the old dynamics is already present at initialization. The additional learning contribution remains small and decoder-dependent. Seeds share the same validation images, so these are not 600 independent test examples. Results, image-boundary checkpoints and cached responses are in `results/20260912-corrected-learning-controls`.

An audit of the first 20 training presentations compared each observed STDP update with a one-step counterfactual that excluded timestamps from before the current image. Pre-image history affected 263 of 2,000 presentation steps. The summed L1 difference was 1.998, against 191.626 for the observed updates (about 1.04%). This difference is not an additive attribution or an accuracy effect. It confirms that the current rest/reset schedule permits cross-image pairing, but does not support treating it as the dominant cause of weak learning. The training boundary policy remains unchanged. Results are in `results/20260912-stdp-boundary-audit.json`; `examples.mnist_boundary_audit` reproduces the diagnostic without changing its source checkpoint.

The completed scale check used 1,000 training images, 100 labelled readout images and a new, disjoint 200-image validation set, with seed 101. Increasing the training count changes the split and fitted intensity target; compare its arms within this study, not directly against the percentages above.

| Condition, 1,000-image study | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---:|---:|---:|
| Initial network | 66.5% | 65.5% | 65.5% |
| Normalization only | 61.5% | 62.5% | 60.5% |
| STDP + normalization | 59.0% | 59.5% | 62.5% |

STDP lost 2.5 template points and 3 spike-ridge points against normalization alone, while gaining 2 points with spike-plus-voltage ridge. Every trained condition remained below initialization for each decoder; none had silent validation samples. The labelled pixel-ridge baseline scored 59.5%. These one-seed results do not support scaling the current training rule to a long run. Training took 150.5 seconds for normalization alone and 159.1 seconds for STDP, excluding readout extraction. All source hashes matched at completion. Checkpoints every 100 images and cached responses are in `results/20260912-learning-1000`.

An evaluation-only factorial crossed feedforward weights and regional adaptive thresholds from those three checkpoints. All three diagonal cases reproduced the saved spike and voltage responses byte for byte. With thresholds reset to their initial zero values, templates scored 66.5% on initial weights, 62.0% on normalization-only weights and 60.0% on STDP weights. Keeping initial weights but using either trained threshold state scored 65.5%. This points primarily to weight changes for the template decline. The effect is decoder-dependent: spike ridge on initial weights fell from 65.5% to 60.0% or 59.5% under the two trained threshold states. Zeroing thresholds is therefore not a general cure. The nine cases neither retrained a network nor changed their source checkpoints; results are in `results/20260912-state-factorial-1000`.

The learning-check CLI also supports `--normalization-policy initial_per_neuron`. This experimental policy fits a separate incoming-weight budget from each neuron's initial weights, retaining their initial sums as targets instead of replacing them with the population mean. The existing weight bounds still apply. Budgets are checkpointed and restored without refitting on resume. The default remains `population_mean`.

The completed per-neuron-budget run used the same 1,000/100/200 split, initial state and seed 101; initial decoder predictions matched exactly. Normalization alone scored 65.0% with templates, 59.5% with spike ridge and 61.5% with spike-plus-voltage ridge. Adding STDP scored 63.5%, 61.0% and 60.5%, respectively. This recovers 4.5 template points versus the previous trained model, but STDP still loses 1.5 template points against its new matched control and every score remains below the corresponding initial-network score. The decoder trade-offs and one-seed scope do not justify changing the default. Source hashes matched at completion; results are in `results/20260912-learning-1000-per-neuron`.

### Activating the matched inhibitory circuit

An initial-network screen found that the intended competition circuit was inactive in the sampled presentations. Across the first eight training images of the 1,000-image split, independent 100ms presentations produced 284, 255, 139, 222, 131, 220, 211 and 187 excitatory spikes, but no inhibitory spikes. In an isolated regular-spiking/fast-spiking pair, one excitatory spike transmitted at the current weight of 8 did not trigger its inhibitory partner within 30ms.

`python -m examples.mnist_wta_check --study STUDY --output NEW_FILE` repeats the pulse and image-activity checks without training or accuracy-based selection. The saved seed-101 initial network produced:

| Matched coupling | Partner's spike step after source fires at step 1 | Mean exc spikes/image | Mean inh spikes/image |
|---|---:|---:|---:|
| 8 | No spike within 30 steps | 206.125 | 0 |
| 24 | 4 | 48.125 | 48.125 |
| 32 | 3 | 28.875 | 28.875 |
| 64 | 2 | 18.375 | 18.375 |

The pulse uses the real one-step delay and short-term synapse dynamics. Weight 64 also triggered the partner with half its initial synaptic resource, at step 3. Pulse timings matched with Heun step caps of 0.1ms and 0.05ms. A separate two-pair regression test verifies that recruited lateral inhibition suppresses a competing spike. These checks establish functional recruitment and competition, not a classification benefit. Results are in `results/20260912-wta-calibration.json`.

The builder accepts `exc_to_inh_weight`, and the learning-check CLI exposes `--exc-to-inh-weight`. Values above 10 explicitly raise the matched edges' upper bounds; otherwise the generic glutamate bound would silently clip the requested pulse. Feedforward weights, lateral inhibitory weights, topology, RNG state and other bounds remain unchanged. The default is still 8, and its initial-state/RNG hash matches the preceding study. Coupling 64 was chosen for prompt pulse recruitment before evaluating accuracy. The completed seed-101 run retained the previous 1,000 training images, 100 labelled readout images and 200 validation images:

| Condition, coupling 64 | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---:|---:|---:|
| Initial network | 57.5% | 46.5% | 64.5% |
| Normalization only | 56.0% | 44.5% | 64.0% |
| STDP + normalization | 56.5% | 49.0% | 65.0% |

STDP gained 0.5, 4.5 and 1 percentage points over its matched normalization control. Compared with initialization, the changes were -1, +2.5 and +0.5 points. The active circuit therefore gives a positive but small one-seed STDP/control comparison, not a general accuracy gain: its template accuracy remains below the weak-coupling alternatives. The default has not changed. Source hashes matched at completion, and the saved checkpoints/responses are in `results/20260912-learning-1000-wta64`.

The completed replication on seeds 102 and 103 used identical source hashes, image rows, preprocessing and readout budget. Means over all three seeds were:

| Condition, coupling 64, three seeds | Templates | Ridge on spikes | Ridge on spikes + voltage |
|---|---:|---:|---:|
| Initial network | 53.33% | 40.50% | 62.33% |
| Normalization only | 51.83% | 38.83% | 62.33% |
| STDP + normalization | 53.17% | 44.33% | 63.17% |

Paired STDP/control gains were +0.5/+1/+2.5 template points and +4.5/+6/+6 spike-ridge points. Spike-plus-voltage gains were +1/+2/-0.5 points. The respective mean gains are +1.33, +5.5 and +0.83 points. Templates still do not improve on mean initialization; spike ridge gains 3.83 points over initialization, and spike-plus-voltage ridge gains 0.83 points. These are repeated network initializations on the same 200 validation images, not 600 independent held-out observations. Replication checkpoints are in `results/20260912-learning-1000-wta64-replication`; the combined summary is `results/20260912-wta64-three-seed-summary.json`.

A subsequent seed-101 check used intermediate coupling 32 with the same image split, budgets and STDP scale 0.2. Initial scores were 56.0% / 48.5% / 64.0%; normalization-only scores were 59.0% / 49.5% / 60.5%; STDP scores were 58.0% / 49.0% / 60.0%, in template/spike-ridge/spike-plus-voltage order. STDP lost 1.0, 0.5 and 0.5 points against this matched control. Coupling 32 therefore does not establish a better learning configuration on this seed. No default changed. Source hashes matched at completion; results are in `results/20260912-learning-1000-wta32`.

### STDP amplitude calibration

After 1,000 training images with active coupling 64, the feedforward L1 distance between the STDP and normalization-only weights was 0.167% of the control's L1 weight sum for seed 101 and 0.161% for seed 102. This small change motivates testing a larger update amplitude; it does not predict the resulting accuracy.

The builder and learning-check CLI now accept `stdp_scale` and `--stdp-scale`, respectively. The default remains 0.2. A seed-101 experiment uses scale 2.0, increasing both LTP and LTD tenfold while preserving their ratio, the circuit, training order, image split and readout budget. The initial decoder predictions match the preceding run exactly. Results also record live-weight distributions and exact occupancy of the configured bounds; distances from initialization include normalization and must not be interpreted as pure STDP effects. Resume tests cover both amplitudes, and the default builder reproduces its previously recorded state/RNG hash.

A 20-image boundary audit at scale 2.0 measured a summed L1 update of 204.102. Censoring pre-image timestamps in one-step counterfactuals changed that by 4.600, about 2.25%, affecting 171 of 2,000 presentation steps. This is a difference along the observed trajectory, not an additive attribution or an accuracy effect. The image-boundary policy remains unchanged. Results are in `results/20260912-boundary-wta64-stdp2.json`; the amplitude experiment is in `results/20260912-learning-1000-wta64-stdp2`.

The scale-2.0 run completed with 55.0% template accuracy, 47.0% spike-ridge accuracy and 64.0% spike-plus-voltage accuracy. Its initial and normalization-only predictions reproduced the scale-0.2 controls. Compared with scale 0.2, the trained scores fell by 1.5, 2.0 and 1.0 points. Of 47,254 live feedforward weights, 114 reached zero and none reached the upper bound. This one-seed result does not support increasing the default amplitude; it also does not show widespread bound saturation. Recorded source hashes matched at completion.

`python -m examples.mnist_plasticity_audit --study STUDY --output NEW_JSON` separates the two arms on private weights, preserving the production order and checking the combined update against production at every step. On the first 20 scale-2.0 images, realized LTP and LTD magnitudes totaled 107.768 and 105.894; the signed weight-sum change before normalization was +1.874. The totals include clipping and do not predict which LTP/LTD ratio would classify better. They do not show aggregate depression dominating this short prefix. The audit uses training images only, leaves the saved model unchanged, and records source/checkpoint hashes. Results are in `results/20260912-plasticity-arms-wta64-stdp2.json`.

Extending the audit to 100 images gave LTP 574.245 and LTD 588.131, a signed pre-normalization total of -13.886. This longer prefix has slightly more depression, still not evidence that a changed ratio would improve accuracy. Every separated-arm update matched production byte for byte. The complete final training-state/RNG hash also matched the original study's 100-image checkpoint after the CPU preflight optimization. The original checkpoint remained unchanged; results are in `results/20260912-plasticity-arms-100-wta64-stdp2.json`.

### Post-triggered trace experiment

The learning check now accepts `--learning-rule post_trace`. Its default remains `pair`. The experimental rule keeps a decaying presynaptic arrival trace for each input-to-cortex synapse and updates weights only when the postsynaptic neuron fires. This takes the arrival-trace and post-triggered update ideas from [Diehl and Cook's learning rules](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full). It is an additive variant with the benchmark's existing normalization, not a reproduction of their weight-dependent rules, conductance-based neurons or Poisson inputs.

Each simulation step applies `x *= exp(-dt / trace_tau)` and adds one for each spike arriving after the synapse's actual delay. A postsynaptic spike then applies `dw = A_plus * learning_rate * (x - trace_target)`, clipped to the existing weight bounds. `A_minus` is unused. The initial settings are `trace_tau=20ms`, `trace_target=0.2` and the unchanged STDP scale 0.2. No labels or rewards enter this update.

Trace history starts empty for each image and is discarded before rest; queued neural currents and electrical state are not reset by the rule. The pair rule still retains last-spike timestamps across images. This experiment therefore compares both the update rule and its explicit trace-boundary policy. Fixed topology is required within a presentation; completed-image checkpoints need no additional trace state. Tests cover arrival timing against real delayed delivery, accumulated events, ring-buffer wrap, masking and bounds, exact pair-loop preservation, and exact checkpoint continuation for both rules.

The learning check also archives its source files and verifies their hashes, including the new rule helper, before marking a run complete. The separate-arm and timestamp-boundary audits reject trace-rule studies before loading a checkpoint or dataset: those diagnostics implement pair-STDP counterfactuals only. The seed-101 screening uses the existing 1,000 training / 100 labelled readout / 200 validation split:

```bash
python -m examples.mnist_learning_check \
  --output results/20260912-learning-1000-post-trace --seeds 101 \
  --train-per-class 100 --readout-per-class 10 --validation-per-class 20 \
  --train-steps 100 --inference-steps 50 --rest-steps 25 \
  --checkpoint-every 100 --exc-to-inh-weight 64 --stdp-scale .2 \
  --learning-rule post_trace
```

The completed scale-0.2 candidate scored 53.5% with templates, 41.0% with spike ridge and 63.0% with spike-plus-voltage ridge. These are 3, 8 and 2 percentage points below pair STDP on the same seed, and 2.5, 3.5 and 1 point below normalization alone. Both control conditions reproduced the earlier complete model/RNG state and response bytes exactly. No validation image was silent. Source and archived-source hashes matched; all six compared checkpoints passed integrity checks and remained unchanged. The comparison is in `results/20260912-post-trace-comparison.json`.

The new rule's weight distance from its normalization control was 1.640% of the control's L1 weight sum, versus 0.167% for pair STDP. Thus the same nominal amplitude produced roughly ten times the final displacement; this does not isolate the cause of the accuracy loss. A completed follow-up used `--stdp-scale .02` with the same trace rule and all other settings unchanged, writing to `results/20260912-learning-1000-post-trace-scale002`.

At scale 0.02, the trace rule scored 54.5% / 46.0% / 64.0%, still 2 / 3 / 1 points below pair STDP at scale 0.2. Weight distance fell to 0.162%, close to the pair rule's 0.167%. Reduced amplitude therefore helps this trace candidate but does not recover the pair rule's scores. Relative to normalization alone, it loses 1.5 template points, gains 1.5 spike-ridge points and ties spike-plus-voltage ridge. Neither trace configuration supports a default change on this seed; pair STDP at scale 0.2 remains the default.

The two trace runs used identical source hashes, image rows and preprocessing. Their control response arrays and predictions matched byte for byte. Control model/RNG state also matched after restoring only the unused `A_plus` and `A_minus` arrays on private copies. Source and snapshot hashes matched at completion, and checkpoints passed integrity checks without modification. The follow-up comparison is in `results/20260912-post-trace-scale002-comparison.json`. Both studies reuse the same 200 validation images and do not evaluate the canonical test set.

### Dense input projection

The builder and learning check now accept `input_density` / `--input-density` and `input_weight_boost` / `--input-weight-boost`. Defaults remain 0.15 and 4.0. Values are checked before constructing a network; zero permits an explicit disconnected or zero-weight ablation. New checkpoint signatures include both parameters and reject incompatible resumes.

The dense candidate connects all 784 pixels to each of the 400 excitatory cortex neurons: 313,600 feedforward synapses instead of the sparse seed-101 network's 47,254. Its weight boost is 0.6 and pair-STDP scale is 0.03. Multiplying the old values by 0.15 preserves the expected incoming weight sum and the nominal update size relative to each initial weight. This is a calibrated topology comparison, not a change in density alone: weight distributions, projection RNG consumption, and instantaneous currents still differ. The inhibitory circuit and all neuron parameters are unchanged.

An eight-image training preflight measured mean initial incoming weight sums of 141.058 for sparse input and 140.907 for dense input, a difference of about 0.11%. Their across-neuron standard deviations were 15.896 and 5.115. Mean excitatory/inhibitory presentation counts were 16.375/16.250 spikes for sparse input and 13.5/13.5 for dense input. The source models were unchanged, and the sparse initialization reproduced the earlier complete state/RNG hash. This checks activity and budget calibration, not classification accuracy. Results are in `results/20260912-dense-input-preflight.json`.

The screening uses the same 1,000 training images, 100 labelled readout images and 200 train-only validation images as the preceding studies, with coupling 64:

```bash
python -m examples.mnist_learning_check \
  --output results/20260912-learning-1000-dense --seeds 101 \
  --train-per-class 100 --readout-per-class 10 --validation-per-class 20 \
  --train-steps 100 --inference-steps 50 --rest-steps 25 \
  --checkpoint-every 100 --exc-to-inh-weight 64 --stdp-scale .03 \
  --learning-rule pair --input-density 1 --input-weight-boost .6
```

The completed dense study scored 52.5% / 34.0% / 48.5% at initialization, 58.0% / 41.0% / 50.5% with normalization alone, and 58.0% / 39.5% / 52.0% with STDP, in template/spike-ridge/spike-plus-voltage order. STDP therefore adds no template points, loses 1.5 spike-ridge points and gains 1.5 spike-plus-voltage points against its own control. Compared with the trained sparse network, dense templates gain 1.5 points but the two ridge decoders lose 9.5 and 13 points. The template gain is already present without STDP.

Dense inference also has silent validation images: 22 at initialization and 27 after either training condition, versus zero for all three sparse conditions. Average spike counts alone hid this difference: dense STDP averaged 24.805 spikes per image, versus 15.025 for sparse STDP. The candidate is not promoted. All six compared checkpoints passed integrity/state checks and remained unchanged; dense source and snapshot hashes matched at completion. The comparison is in `results/20260912-dense-input-comparison.json`.

### Decoder-only voltage centering screen

A separate probe reused the saved sparse-circuit responses for seeds 101, 102 and 103, without simulating or training an SNN. It fitted the same standardized ridge classifier on the same 100 labelled readout images. Two per-image transforms were compared with the original concatenated spike/voltage features: subtracting that image's mean voltage across excitatory neurons, and adding normalization of its spike counts by their total. Both transforms use float64 arithmetic and no validation labels or cross-image statistics.

| Mean ridge accuracy, three seeds | Original features | Centered voltage | Centered voltage + spike fractions |
|---|---:|---:|---:|
| Initial network | 62.33% | 64.17% | 66.67% |
| Normalization only | 62.33% | 64.00% | 63.50% |
| STDP + normalization | 63.17% | 64.50% | 63.50% |

Voltage centering gained 1, 2 and 1 points on the three trained networks. Every original-feature prediction reproduced the saved decoder, and all response caches were unchanged. This small gain was exploratory: the same 200 validation images had been reused for several experiments. It motivated the additional-image check below, without changing the SNN or template defaults. Results and input hashes are in `results/20260912-readout-centering-screen.json`.

`examples.mnist_readout_check` verifies this candidate on additional images without retraining the SNN. It fixes the comparison to original features and centered voltages, keeps ridge alpha and the 100 labelled readout samples unchanged, and selects 80 additional canonical-training images per class with seed 20260913. These 800 rows exclude the source studies' 1,000 training and 200 prior validation rows; the fitted intensity target is reused. “Additional” refers to these source studies, not every historical MNIST experiment. The canonical test split remains untouched.

Before scoring new images, the check re-extracts each model's original readout and validation responses and requires identical array bytes and original ridge predictions. It then extracts the new responses on frozen independent copies, scores the two fixed classifiers, and verifies source-model/RNG, checkpoint and cache preservation. Source files are archived and checked again before completion. Tests cover disjoint balanced selection, fixed preprocessing, per-image transforms, label-independent prediction, complete-source preservation and rejection of inconsistent caches or changed code.

```bash
python -m examples.mnist_readout_check \
  --studies results/20260912-learning-1000-wta64 \
            results/20260912-learning-1000-wta64-replication \
  --output results/20260912-readout-centering-fresh \
  --fresh-per-class 80 --fresh-seed 20260913
```

The completed check reproduced all original response bytes before evaluating the additional images:

| Network seed | Original ridge | Centered-voltage ridge | Gain |
|---|---:|---:|---:|
| 101 | 61.500% | 63.250% | +1.750 pp |
| 102 | 63.125% | 63.250% | +0.125 pp |
| 103 | 60.000% | 62.625% | +2.625 pp |

Mean accuracy rose from 61.54% to 63.04%, a gain of 1.50 points. Paired wins/losses were 23/9, 14/13 and 28/7. The same 800 additional images were used for each model, not 2,400 independent images. A 5,000-replicate bootstrap resampled images within each digit class after averaging the three model-specific correctness differences per image. Its 95% interval was +0.71 to +2.33 points, conditional on these three fixed models; it does not estimate variation across a population of network seeds. Results are in `results/20260912-readout-centering-fresh`, with integrity and uncertainty checks in `results/20260912-readout-centering-fresh-verification.json`.

The learning check now reports `ridge_spikes_centered_voltage` alongside its original three decoders. The shared transform in `examples/mnist_readout_features.py` is read-only and uses the same float64 arithmetic as the verified candidate. No SNN training rule, coupling, template classifier or existing decoder was replaced. An integration check preserved every original prediction and accuracy across all nine cached source conditions and reproduced both raw and centered predictions on all additional images. It required no further SNN simulation; results are in `results/20260912-readout-integration-verification.json`.

## Completed pre-correction full MNIST run

The original process in `results/20260911-190006` completed successfully with 48.8% accuracy on 500 canonical test images, after training on 50,000 images and fitting its readout on 5,000 labelled training images. It reported 17,651.1 seconds for training, 1,192.3 seconds for readout construction and 101.4 seconds for evaluation; total wall time was 18,953 seconds, about 5h16m. The process was not interrupted or restarted.

That run used the integrator, wiring and sequential-image evaluation code loaded before the corrections. Its result does not validate the current model and is not directly comparable to the corrected small validation studies. The runner and all jobs in that batch are terminal, with return code zero. The final log is `results/20260911-190006/09_mnist_full.log`.

## Earlier corrections and measurements

The stopped diagnostic session that motivated the latest fixes found:

- Iris completed at **80.0% spike-only held-out accuracy**, with **4/30** silent samples. The voltage fallback is now reported separately instead of being folded into the primary metric.
- Grid was stopped at episode **275/500**. Deterministic evaluations fluctuated between roughly **12.5% and 40%** success and replay sometimes reduced performance; this did not establish learning above the measured **29% random baseline**.
- The reduced MNIST baseline completed at **23.0%**, well below the historical **48.0%** reduced result. Inspection found all `159,600` lateral GABA synapses at `-10`, while excitatory internal weights had collapsed from about `8` to a mean near `1.83`.

Those observations led to the following protocol/dynamics corrections:

1. Homeostatic scaling now converts the raw decayed activity count to Hz before comparing it with `target_rate`; the previous comparison was off by about `200x` at `dt=1ms` and decay `0.995`.
2. Synaptic scaling and adaptive theta can be enabled independently. MNIST freezes only scaling, preserving theta, while Grid freezes scaling for a stable minimal baseline.
3. Initial synaptic weights are clamped to their declared bounds. The current MNIST WTA therefore uses inhibition `10.0`, not the out-of-range historical `12.0`.
4. MNIST uses the canonical `60,000/10,000` split, fits intensity equalization on training data only, and requests `5,000` training examples per class. Grid now isolates direct one-hot state-to-action R-STDP and disables replay/oscillations until the minimal policy learns.
5. Validation/readout/test ran on copies with adaptive thresholds frozen and encoder noise disabled. Membrane state and short-term synaptic dynamics still evolved between images in that version; the independent-image patch above changes MNIST inference only.

The follow-up review added these corrections, verified with unit tests and synthetic networks only:

- Grid now uses 25 excitatory sensory neurons, providing all 100 state/action connections. Grid and Iris restrict credit for the full reward or punishment pulse, including newly generated spikes, and clear residual dopamine at the end.
- MNIST diagnosis resolves rest duration at call time: the reduced protocol now actually uses its configured 25 ms. The competition sweep uses only 6, 8 and 10, avoiding a duplicate run caused by clipping 12 to 10. Historical reduced results should not be assumed to have used the declared rest duration.
- MNIST readout and evaluation freeze topology and synaptic adaptation on copies, keeping neuron indices consistent even for growth experiments. The degradation benchmark's missing import has been restored. The forgetting test reuses task A's original decoder after training B; both SNN and MLP are evaluated with the task's candidate classes.
- STDP/R-STDP freeze state survives simulation steps and save/load. Eligibility decay uses elapsed milliseconds via `exp(-dt/tau)`; this does not make every simulator mechanism invariant to the integration timestep.
- Memory admits new patterns by evicting the oldest stored trace when full, including consolidated traces. Consecutive identical patterns within a region refresh the activity snapshot instead of occupying additional slots. Strength still weights replay sampling.

The subsequent batch and checkpoint diagnosis are recorded in the result directories above. Their measurements do not establish recovery of the historical accuracy.

For exact hyperparameters, read the benchmark scripts:

- `examples/iris_benchmark.py`
- `examples/grid_nav_benchmark.py`
- `examples/mnist_benchmark.py`

## Iris classification

Historical pre-fix result: **86.7%** final test accuracy, with a **90.0%** checkpoint selected using the test set. This number is retained for provenance, not as a valid held-out estimate.

Research setup:


- Place-field encoding expands the 4 Iris features into `80` excitatory sensory neurons.
- The useful readout is the direct `input->motor` pathway trained with R-STDP.
- Training rewards the teacher-selected class and also punishes the strongest wrong pre-teacher response.
- Evaluation uses independent, frozen, noiseless presentations. The six identical responses are computed once and reused in the vote.
- The corrected protocol fits normalization on training data, selects checkpoints on validation data, and evaluates the test set once.

### September 12 revalidation

The builder audit found that the generic sensory region made 16 of its 80 neurons inhibitory at seed 42. Since inter-region connections use excitatory sources, those 16 encoded bins had no outgoing pathway: the direct readout contained 192 edges instead of 240. Iris now creates all 80 sensory neurons explicitly as excitatory. The same audit found 39 input-to-cortex weights above their declared maximum after the benchmark's weight boost; projection weights are now clamped after initialization. Cortex-to-motor remains zero and frozen. These are architecture/bounds corrections, not execution-only speedups.

The previous sequential evaluation retained electrical history and oscillations between samples. In a three-sample check on the untrained model, reversing sample order changed two voltage-fallback predictions; all spike-only predictions were silent in that check. Standard Iris evaluation now freezes plasticity, growth, thresholds, memory and oscillations on copies and resets transient state before each sample. `evaluate(..., independent=False)` retains the former protocol for explicit diagnostics. `evaluate(..., fast=False)` retains all repetitions within the new independent protocol. Training keeps its original rest schedule, memory replay, oscillations and fresh encoder noise at every step. Its deterministic input conversion happens once per sample.

The frozen-copy and reset helpers are shared with MNIST in `examples/_utils.py`. MNIST's wrappers preserve its existing policy switches. Tests check source-state/RNG preservation, sample-order independence, raw repeated responses, and exact fast/slow counts and predictions. A saved-state check also reproduced ten previous MNIST validation responses byte for byte after this refactor.

`python -m examples.iris_benchmark --output NEW_DIR --validation-only` records the configuration, source hashes, exact split IDs, normalization bounds, epoch metrics and immutable initial/per-epoch/selected checkpoints. The checkpoints include the shuffle RNG state after each epoch, but there is no CLI resume support. Omitting `--validation-only` retains final test scoring. Existing output directories are rejected.

The completed run retained the original 96/24/30 train/validation/test split and 15-epoch budget. It selected epoch 13 by spike-only validation accuracy: 24/24 correct, with no silent samples. Initialization was silent on all 24 validation samples. Training plus validation and recording completed in 133.64 seconds. Results and checkpoints are in `results/20260912-iris-corrected-validation`.

`python -m examples.iris_inference_check --study STUDY --output NEW_JSON --evaluate-test` reproduced the selected validation results and evaluated that checkpoint on the held-out test partition without further tuning. It classified 27/30 test samples correctly (90.0%), with no silent samples; voltage fallback therefore also scored 90.0%. Logistic regression, fitted on the same 96 training rows, scored 29/30 (96.67%). This is one seed and a small test set. It revalidates this corrected configuration; it does not isolate the accuracy effect of each correction or establish a general advantage over the baseline.

The same check timed slow/fast/fast/slow evaluation on the 24 validation samples. Median time fell from 10.270 to 1.702 seconds (6.03x), with identical spike counts and both prediction vectors. The saved checkpoint and complete source model/RNG state were unchanged. This speedup compares two implementations of the independent protocol, not independent versus historical sequential evaluation. The timing run followed training with no other benchmark launched concurrently by this task. Results and the one final test evaluation are in `results/20260912-iris-inference-and-test.json`.

The final test suite passed with 601 tests and two device-dependent skips. Source hashes matched at the end of both Iris runs.

Remaining checks:

- Replicate the result across network seeds and data splits.
- Separate the contributions of targeted reward, teacher timing and memory replay with matched controls.

Interpretation notes:

- Reward updates depend on the eligibility present during the pulse, so teacher and input spike timing must be measured together.
- No-response predictions count as wrong in the primary spike-only metric.

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

### Independent-episode revalidation — 2026-09-12

Evaluation now freezes plasticity, thresholds, scaling, reward, memory and oscillations on a noiseless copy. Transient state resets between episodes; neural history remains intact within each path. Starting positions are drawn before any rollout, so path length cannot alter later starts. Every checkpoint uses the same 40 selection starts. Initial, random and selected policies share a separate list of 100 final starts. These lists sample the same finite world: this is not unseen-state generalization, and repeated starts are not independent tests of a deterministic policy.

Silence is divided by the number of actions actually executed, rather than the maximum episode length. Training also counts silent responses during exploration. Training's reward rule, action marking, noise, rest and continuous neural state remain unchanged. `independent=False` is available for explicit diagnostics of the former sequential evaluation.

The 500-episode run selected episode 250 using only the selection starts:

| Policy | Success on 100 final starts | Mean steps, all paths | Mean final distance |
|---|---:|---:|---:|
| Initial | 0% | 20.00 | 5.48 |
| Random | 27% | 16.79 | 2.92 |
| Selected episode 250 | 41% | 15.19 | 2.01 |

The selected policy solved **9 of all 24 non-goal starts (37.5%)**. Successful exhaustive paths averaged 6.89 steps; all paths averaged 15.08. This one-seed result does not establish a reliable controller. Selection success fluctuated from 45% at episode 250 to 10% at 300, 2.5% at 400 and 40% at 450. Continuing training did not yield steady improvement.

The recorded run is `results/20260912-grid-corrected`. It contains immutable initial, 50-episode and selected checkpoints, the exact starting positions, training RNG state at checkpoints and source hashes. Total wall time was 522.36 seconds; this is not a controlled training-speed comparison. CLI resume is not implemented.

Within the corrected deterministic protocol, repeated starts can reuse complete episode metrics. An ABBA comparison on the selected checkpoint and the same 100 final starts measured median evaluation time of **42.067 seconds without reuse and 10.036 seconds with reuse (4.192×)**. Every aggregate metric matched the original final result. The saved model, RNG, checkpoint and source files were unchanged. This measures evaluation reuse only, not an end-to-end training speedup or a comparison with the former sequential protocol.

`results/20260912-grid-policy-check.json` also records every action from all 24 starts. Of 362 decisions, 76 had tied maximum spike counts, 43 hit a wall and 138 increased distance to the goal. The first response was `[0, 0, 0, 4]` motor spikes at every starting cell, always choosing right despite different learned state-action weights. This motivates separating state-dependent input from motor excitability and within-path history before changing the learning rule.

Reproduce with new output paths:

```bash
python -m examples.grid_nav_benchmark --output results/grid-study
python -m examples.grid_policy_check --study results/grid-study --output results/grid-policy.json
```

`examples/grid_state_check.py` provides separate saved-state interventions for readout weights, motor thresholds, tonic drive and spike-tie handling. Its outputs are diagnostic model/protocol changes, not alternative training results or execution-only optimizations.

The saved-state checks reproduced the 9/24 baseline, then tested the following interventions without changing the checkpoint:

| Intervention | Successful starts / 24 | Silent decisions |
|---|---:|---:|
| Baseline | 9 | 1.38% |
| Restore initial readout weights; retain learned theta | 7 | 0% |
| Equalize motor theta at its saved mean | 7 | 9.30% |
| Multiply readout weights by 16, within existing bounds | 8 | 1.89% |
| Gain 16 and equal motor theta | 11 | 0.86% |
| Voltage breaks tied maximum spike counts | 6 | 0.67% |
| Equal theta and 100 ms rest | 3 | 0% |
| Equal theta and 200 ms rest | 4 | 0% |
| Equal theta and no tonic motor current | 24 | 100% |

The last arm solved every start in the minimum total of 100 actions (mean 4.17), but emitted no motor spikes: every decision used the voltage fallback. Gain 16 with no tonic current also solved all 24 starts entirely through fallback, with either saved or equalized motor theta. These results show useful state-dependent information in the learned projection, but do not solve the spike-based readout problem. All nine baseline successes used spikes throughout their paths.

Equalizing motor theta made the first action distance-reducing at all 24 starts, yet most complete paths still failed. Merely extending rest to 100 or 200 ms made performance worse. These interventions do not justify promoting theta equalization, a tie rule or longer rest as a standalone fix. The next learning experiment needs to test a motor readout that preserves this state information across decisions, with spike-driven successes reported separately from fallback-assisted ones.

Artifacts are `results/20260912-grid-state-check.json`, `results/20260912-grid-drive-interactions.json` and `results/20260912-grid-recovery-check.json`. Each check verified unchanged source model/RNG, source files and saved checkpoint at completion. `--cases` can restrict the diagnostic; it always reproduces the baseline first. No training or default-parameter change resulted from these checks.

### Motor transmission calibration

A second saved-state check removed tonic motor current and multiplied `input->motor` transmission modulation by a fixed gain. It retained the learned weights, their bounds, adaptive theta and the spike-count action rule. This changes the effective synaptic current; it is not an execution optimization or a biologically validated conductance calibration.

| Transmission gain, tonic current 0 | Successful starts / 24 | Successes without fallback / 24 | Silent decisions |
|---|---:|---:|---:|
| 32 | 24 | 0 | 82.00% |
| 64 | 22 | 1 | 58.33% |
| 128 | 24 | 14 | 31.29% |
| 256 | 15 | 15 | 34.91% |
| 512 | 21 | 19 | 15.56% |

The results are in `results/20260912-grid-transmission-check.json` and `results/20260912-grid-transmission-high-check.json`. The gain-512 intervention raised fully spike-driven successes from 9 to 19, but still failed three starts and used fallback on two successful paths. This diagnostic selected a candidate for new training; it does not establish that training with gain 512 will learn a better policy.

The benchmark exposes `--transmission-gain`, `--motor-baseline-current`, `--episodes`, `--seed` and `--selection-metric`. `spike_only_success_rate` counts successful episodes with no silent motor decision anywhere in the path. `fallback_assisted_success_rate` counts successful episodes that used voltage at least once. Their sum is the existing `success_rate`; silent or failed episodes are not removed from the denominator. Checkpoint scoring is `100 * selected_rate - mean_steps`.

The gain-one builder reproduced the full state/RNG of the original saved initialization exactly. New runs also save the source bytes used for the study in `source_snapshot`, alongside the hashes. The gain-512 training used the same 500-episode reward protocol, with spike-only checkpoint selection:

```bash
python -m examples.grid_nav_benchmark --output results/grid-gain512 \
  --transmission-gain 512 --motor-baseline-current 0 \
  --selection-metric spike_only_success_rate --episodes 500 --seed 42
```

Three completed runs used identical settings except for the network/training seed:

| Seed | Selected episode | Initial total success on final starts | Final spike-only success | Exhaustive spike-only starts | Exhaustive mean steps |
|---|---:|---:|---:|---:|---:|
| 42 | 400 | 4% | 100% | 24/24 | 4.333 |
| 43 | 500 | 0% | 100% | 24/24 | 4.167 |
| 44 | 450 | 7% | 100% | 24/24 | 4.250 |

All selected policies had zero silent decisions in both final and exhaustive evaluation. Across seeds, exhaustive paths averaged **4.25 steps**; the minimum possible average over these starts is 4.167. These are three trained networks on the same 24-state evaluation set, not 72 independent held-out environments. The separate final-start lists and random baselines differ by seed. Gain 512, zero tonic motor current and spike-only checkpoint selection are now the Grid defaults; this is a benchmark-specific calibration, not a change to core simulator defaults or a claim about larger worlds.

The former baseline remains available explicitly:

```bash
python -m examples.grid_nav_benchmark --output results/grid-former-baseline \
  --transmission-gain 1 --motor-baseline-current 4 --selection-metric success_rate
```

For seed 42, restoring only the selected checkpoint's readout weights to their initial values reduced exhaustive success from 24/24 to 1/24, with learned theta and transmission gain retained. This saved-state control supports a contribution from learned weights; it is not a separate on-policy training ablation. The artifact is `results/20260912-grid-gain512-weight-control.json`.

`results/20260912-grid-gain512-policy-check.json` reproduced every final and exhaustive metric from the saved seed-42 checkpoint. All 104 traced actions used spikes, with no wall moves; two left moves account for four extra steps above the shortest-path total. The independent-protocol ABBA check measured median evaluation time of 11.564 seconds without reuse and 2.920 seconds with reuse, **3.960×**, with all metrics equal and the source model/RNG unchanged. Seed-42 training and evaluation took 209.31 seconds end to end. Its shorter paths reduce simulation work, so this is not a kernel-speed comparison with the earlier 522.36-second run. The other seeds ran concurrently; their wall times are not used for performance claims.

The studies are `results/20260912-grid-gain512-training`, `results/20260912-grid-gain512-seed43` and `results/20260912-grid-gain512-seed44`. `results/20260912-grid-gain512-replication-summary.json` checks matched configurations, archived source bytes, checkpoint integrity and completed episode histories. Current-default verification additionally compares the full saved initialization/RNG and reproduces both final and exhaustive metrics for every seed:

```bash
python -m examples.grid_replication_summary --studies \
  results/20260912-grid-gain512-training results/20260912-grid-gain512-seed43 \
  results/20260912-grid-gain512-seed44 --output results/grid-defaults-check.json \
  --verify-defaults
```

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
