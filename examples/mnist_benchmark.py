"""
MNIST benchmark — unsupervised STDP + template readout.

Full Step 3 benchmark following the Diehl & Cook 2015 architecture:

- true MNIST loaded from OpenML, all 10 digit classes
- 784 input neurons (one per pixel, rate coded)
- 400 excitatory + 400 inhibitory cortex neurons with WTA microcircuit
- unsupervised STDP on input->cortex feedforward synapses
- adaptive excitability thresholds + per-neuron weight normalization
- L1 intensity equalization for balanced cross-class drive
- class readout from cortex response templates (spike + voltage blend)

Target: 85-90% accuracy (Diehl & Cook 2015 achieved 95% with 6400 exc neurons).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import confusion_matrix, quiet_steps
from examples._utils import can_reuse_independent_inference, inference_brain, reset_independent_state
from examples.mnist_trace_stdp import PostTraceSTDP, validate_learning_rule
from src.device import DEVICE

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    print("This prototype requires scikit-learn:")
    print("  pip install scikit-learn")
    raise SystemExit(1)

from src.brain import Brain, DISABLED_GROWTH_INTERVAL
from src.neuron import FiringPattern, NeuronType
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


# --- Dataset scope ----------------------------------------------------------

CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TRAIN_PER_CLASS = 5000
READOUT_PER_CLASS = 500
TEST_PER_CLASS = 50
DOWNSAMPLE = 1
IMAGE_SIDE = 28 // DOWNSAMPLE
N_INPUT = IMAGE_SIDE * IMAGE_SIDE


# --- Network / training hyperparameters ------------------------------------

N_CORTEX_EXC = 400
N_CORTEX_INH = 400
CORTEX_CONNECTIVITY = 0.0
EXC_TO_INH_WEIGHT = 8.0
INH_LATERAL_WEIGHT = 10.0
INPUT_TO_CORTEX_DENSITY = 0.15
INPUT_WEIGHT_BOOST = 4.0

TRAIN_PRESENT_STEPS = 200
ASSIGN_PRESENT_STEPS = 50
TEST_PRESENT_STEPS = 50
REST_STEPS = 50
EPOCHS = 1
ASSIGN_TOP_K = 20
TEST_REPEATS = 2
SPIKE_SCORE_WEIGHT = 0.7
VOLTAGE_SCORE_WEIGHT = 0.3
ENCODER_MAX_CURRENT = 25.0
ENCODER_NOISE = 0.02

STDP_SCALE = 0.2
STDP_A_PLUS = 0.01
STDP_A_MINUS = 0.0105

THETA_PLUS = 0.10
THETA_LEAK = 0.005

SEED = 42

# Historical diagnosis may opt out locally; standard image classification
# always starts each presentation from the same electrical/STP state.
INDEPENDENT_INFERENCE = True
# Disable only for equivalence checks against the unoptimized independent policy.
FAST_INDEPENDENT_INFERENCE = True


# --- Brain construction -----------------------------------------------------


def validate_input_projection(density: float, weight_boost: float) -> None:
    if not math.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("Input density must be finite and between zero and one")
    if not math.isfinite(weight_boost) or weight_boost < 0.0:
        raise ValueError("Input weight boost must be finite and nonnegative")


def build_brain(seed: int = SEED, *, integration_method: str = "heun",
                integration_max_step: float = 0.1, feedforward_exc_only: bool = True,
                exc_to_inh_weight: float | None = None, stdp_scale: float | None = None,
                input_density: float | None = None, input_weight_boost: float | None = None) -> Brain:
    plasticity_scale = STDP_SCALE if stdp_scale is None else stdp_scale
    if not math.isfinite(plasticity_scale) or plasticity_scale < 0:
        raise ValueError("STDP scale must be finite and nonnegative")
    density = INPUT_TO_CORTEX_DENSITY if input_density is None else input_density
    weight_boost = INPUT_WEIGHT_BOOST if input_weight_boost is None else input_weight_boost
    validate_input_projection(density, weight_boost)
    brain = Brain(dt=1.0, seed=seed, integration_method=integration_method,
                  integration_max_step=integration_max_step)

    # Input region: all sensory neurons are excitatory so every pixel can project.
    input_region = brain.add_region(
        "input",
        RegionType.SENSORY,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_INPUT,
    )
    for _ in range(N_INPUT):
        input_region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)

    cortex = brain.add_region(
        "cortex",
        RegionType.ASSOCIATION,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_CORTEX_EXC + N_CORTEX_INH,
    )
    for _ in range(N_CORTEX_EXC):
        cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    for _ in range(N_CORTEX_INH):
        cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
    wire_cortex_microcircuit(cortex, exc_to_inh_weight=exc_to_inh_weight)

    brain.connect_regions(
        "input", "cortex", density=density,
        target_neuron_type=NeuronType.EXCITATORY if feedforward_exc_only else None,
    )
    brain.freeze_plasticity()
    proj = brain.enable_projection_plasticity(
        "input",
        "cortex",
        A_plus=STDP_A_PLUS * plasticity_scale,
        A_minus=STDP_A_MINUS * plasticity_scale,
    )
    ns = proj.n_synapses
    proj.syn_weight[:ns] *= weight_boost
    proj.syn_weight[:ns] = torch.clamp(
        proj.syn_weight[:ns],
        proj.syn_min_weight[:ns],
        proj.syn_max_weight[:ns],
    )

    # Learn only on feedforward synapses onto excitatory cortex neurons.
    cortex_types = cortex.neuron_type[:cortex.n_neurons]
    exc_post = cortex_types[proj.syn_post[:ns].to(torch.int64)] == NeuronType.EXCITATORY.value
    proj.syn_A_plus[:ns][~exc_post] = 0.0
    proj.syn_A_minus[:ns][~exc_post] = 0.0

    # Disable unrelated mechanisms so the prototype isolates STDP + readout.
    brain.disable_reward_modulated_plasticity()
    brain.reset_traces()
    brain.freeze_structural_plasticity()
    # Keep the WTA microcircuit fixed; adaptive theta remains active.
    brain.freeze_homeostatic_scaling()
    brain.disable_memory()
    brain.disable_oscillations()

    # Keep benchmark-level theta settings explicit so sweeps can override them.
    brain.homeostasis.theta_plus = THETA_PLUS
    brain.homeostasis.theta_leak = THETA_LEAK
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE

    return brain


def wire_cortex_microcircuit(cortex: Region, *, exc_to_inh_weight: float | None = None) -> None:
    """Wire matched exc-inh pairs and lateral inhibition of competing exc cells.

    Each exc[i] drives exactly inh[i]; each inh[i] suppresses every
    exc[j != i]. The coupling must activate the inhibitory partner for this
    topology to implement competition; topology alone does not guarantee it.
    An explicit coupling above 10 raises only the matched edges' weight bound.
    """
    coupling = EXC_TO_INH_WEIGHT if exc_to_inh_weight is None else exc_to_inh_weight
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError("Excitatory-to-inhibitory coupling must be finite and nonnegative")
    n = cortex.n_neurons
    if n == 0:
        return

    types = cortex.neuron_type[:n]
    exc_idx = torch.where(types == NeuronType.EXCITATORY.value)[0]
    inh_idx = torch.where(types == NeuronType.INHIBITORY.value)[0]
    n_matched = min(len(exc_idx), len(inh_idx))
    if n_matched == 0:
        return

    # 1:1 exc[i] -> inh[i]. Preserve the requested pulse amplitude instead of
    # silently clipping an experimental coupling to the generic glutamate cap.
    matched_start = cortex.n_synapses
    cortex.add_synapses(
        exc_idx[:n_matched].to(torch.int32),
        inh_idx[:n_matched].to(torch.int32),
        torch.full((n_matched,), coupling, dtype=torch.float32, device=DEVICE),
        torch.full((n_matched,), 1.0, dtype=torch.float32, device=DEVICE),
        torch.full((n_matched,), NeurotransmitterType.GLUTAMATE.value, dtype=torch.int32, device=DEVICE),
    )
    if coupling > 10.0:
        matched = slice(matched_start, matched_start + n_matched)
        cortex.syn_max_weight[matched] = coupling
        cortex.syn_weight[matched] = coupling

    # Each inh[i] -> every exc[j != i] (all-to-all minus self-pair).
    ii, jj = torch.meshgrid(torch.arange(n_matched, device=DEVICE), torch.arange(n_matched, device=DEVICE), indexing="ij")
    off_diag = ii != jj
    n_off_diag = int(off_diag.sum().item())
    cortex.add_synapses(
        inh_idx[ii[off_diag]].to(torch.int32),
        exc_idx[jj[off_diag]].to(torch.int32),
        torch.full((n_off_diag,), INH_LATERAL_WEIGHT, dtype=torch.float32, device=DEVICE),
        torch.full((n_off_diag,), 1.0, dtype=torch.float32, device=DEVICE),
        torch.full((n_off_diag,), NeurotransmitterType.GABA.value, dtype=torch.int32, device=DEVICE),
    )

def excitatory_cortex_indices(brain: Brain) -> torch.Tensor:
    cortex = brain.regions["cortex"]
    types = cortex.neuron_type[:cortex.n_neurons]
    alive = cortex.neuron_alive[:cortex.n_neurons]
    return torch.where((types == NeuronType.EXCITATORY.value) & alive)[0]


# --- Dataset helpers --------------------------------------------------------

def _pooled_unit_images(X: np.ndarray, factor: int) -> np.ndarray:
    n = X.shape[0]
    side = int(np.sqrt(X.shape[1]))
    imgs = X.reshape(n, side, side)
    new_side = side // factor
    pooled = imgs.reshape(n, new_side, factor, new_side, factor).mean(axis=(2, 4))
    flat = pooled.reshape(n, new_side * new_side)
    max_per_image = np.maximum(flat.max(axis=1, keepdims=True), 1e-6)
    return flat / max_per_image


def l1_equalization_target(X: np.ndarray, factor: int = DOWNSAMPLE) -> float:
    flat = _pooled_unit_images(X, factor)
    return float(np.median(flat.sum(axis=1)))


def downsample_images(
    X: np.ndarray,
    factor: int = DOWNSAMPLE,
    target_l1: float | None = None,
) -> np.ndarray:
    flat = _pooled_unit_images(X, factor)
    # Equalize total stimulus across images so sparse digits (e.g. "1")
    # get proportionally stronger drive than dense ones (e.g. "0").
    l1 = flat.sum(axis=1, keepdims=True)
    if target_l1 is None:
        target_l1 = float(np.median(l1))
    return np.clip(flat * (target_l1 / np.maximum(l1, 1e-6)), 0.0, 1.0)


def load_reduced_mnist(
    classes: tuple[int, ...] | None = None,
    train_per_class: int | None = None,
    test_per_class: int | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    train_per_class = TRAIN_PER_CLASS if train_per_class is None else train_per_class
    test_per_class = TEST_PER_CLASS if test_per_class is None else test_per_class
    data_home = str(Path(__file__).resolve().parent.parent / ".sklearn_data")
    print("Fetching MNIST...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=data_home,
    )
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int64)

    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []
    canonical_train = np.arange(0, 60_000, dtype=np.int64)
    canonical_test = np.arange(60_000, len(y), dtype=np.int64)
    for cls in classes:
        cls_train = canonical_train[y[canonical_train] == cls]
        cls_test = canonical_test[y[canonical_test] == cls]
        rng.shuffle(cls_train)
        rng.shuffle(cls_test)
        if len(cls_train) < train_per_class or len(cls_test) < test_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_train)} canonical train and "
                f"{len(cls_test)} canonical test samples"
            )
        train_idx.extend(cls_train[:train_per_class])
        test_idx.extend(cls_test[:test_per_class])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    target_l1 = l1_equalization_target(X_train_raw)
    X_train = downsample_images(X_train_raw, target_l1=target_l1)
    X_test = downsample_images(X_test_raw, target_l1=target_l1)
    return X_train, y[train_idx], X_test, y[test_idx]


def _balanced_subset(
    y: np.ndarray,
    classes: tuple[int, ...],
    per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Return indices for a balanced subset of at most per_class samples per class."""
    rng = np.random.default_rng(seed)
    indices = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:per_class])
    indices = np.asarray(indices, dtype=np.int64)
    rng.shuffle(indices)
    return indices


# --- Simulation helpers -----------------------------------------------------

def reset_brain_state(brain: Brain, rest_steps: int | None = None) -> None:
    if rest_steps is None:
        rest_steps = REST_STEPS
    quiet_steps(brain, rest_steps)
    for region in brain.regions.values():
        n = region.n_neurons
        region.current[:n] = 0.0
        region.fired[:n] = False
    brain.reset_traces()


def reset_inference_state(brain: Brain) -> None:
    """Reset transient state on an inference copy, preserving learned theta.

    Weights, topology, neuron parameters, time and cumulative counters are not
    reset. Inference copies disable oscillations so absolute time cannot leak
    sample position into the stimulus. Training uses reset_brain_state instead.
    """
    reset_independent_state(brain)


def apply_feedforward_stdp(brain: Brain) -> int:
    proj = brain.get_projection("input", "cortex")
    source = brain.regions["input"]
    target = brain.regions["cortex"]
    ns = proj.n_synapses
    if ns == 0 or not proj.plasticity_enabled:
        return 0
    return brain.stdp.apply_event(
        fired_pre=source.fired[:source.n_neurons],
        fired_post=target.fired[:target.n_neurons],
        syn_pre=proj.syn_pre[:ns],
        syn_post=proj.syn_post[:ns],
        pre_last_spike_arr=source.last_spike_time[:source.n_neurons],
        post_last_spike_arr=target.last_spike_time[:target.n_neurons],
        A_plus=proj.syn_A_plus[:ns],
        A_minus=proj.syn_A_minus[:ns],
        alive=proj.syn_alive[:ns],
        weights=proj.syn_weight[:ns],
        min_weight=proj.syn_min_weight[:ns],
        max_weight=proj.syn_max_weight[:ns],
        current_time=brain.time,
        event_indices=(proj._pre_events, proj._post_events),
    )


def present_sample(
    brain: Brain,
    x: np.ndarray,
    n_steps: int,
    learn: bool = False,
    *,
    collect_responses: bool = True,
    learning_rule: str = "pair",
    trace_tau: float = 20.0,
    trace_target: float = 0.2,
) -> tuple[np.ndarray, np.ndarray] | None:
    validate_learning_rule(learning_rule, trace_tau, trace_target)
    trace_rule = (PostTraceSTDP(brain, tau=trace_tau, target=trace_target)
                  if learn and learning_rule == "post_trace" else None)
    # The encoder still draws fresh noise at every step. Only the deterministic
    # conversion is moved outside the loop, including during training.
    stimulus = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    cortex = brain.regions["cortex"]
    if collect_responses:
        exc_idx = excitatory_cortex_indices(brain)
        before = cortex.total_spikes[exc_idx].clone()
        voltage_sum = torch.zeros(len(exc_idx), dtype=torch.float32, device=cortex.v.device)

    for _ in range(n_steps):
        brain.stimulate("input", stimulus)
        brain.step()
        if collect_responses:
            voltage_sum += cortex.v[exc_idx]
        if learn:
            if trace_rule is None:
                apply_feedforward_stdp(brain)
            else:
                trace_rule.step()

    if not collect_responses:
        return None
    counts = cortex.total_spikes[exc_idx] - before
    mean_voltage = voltage_sum / max(n_steps, 1)
    return counts.cpu().numpy(), mean_voltage.cpu().numpy()


def present_inference_sample(brain: Brain, x: np.ndarray, n_steps: int):
    """Present an independent image/repeat on an already frozen snapshot."""
    if INDEPENDENT_INFERENCE:
        reset_inference_state(brain)
    return present_sample(brain, x, n_steps, learn=False)


def can_reuse_inference(brain: Brain) -> bool:
    """Allow shortcuts only for independent, deterministic, frozen networks.

    Low-level callers can also pass training networks: these must keep their
    presentation/rest schedule even when the standard inference fast path is on.
    """
    return (FAST_INDEPENDENT_INFERENCE and INDEPENDENT_INFERENCE
            and can_reuse_independent_inference(brain))


@dataclass(frozen=True)
class ReadoutResponses:
    """Reusable features in input sample order, before fitting a decoder."""

    exc_indices: np.ndarray
    spikes: np.ndarray
    voltages: np.ndarray


def collect_readout_responses(brain: Brain, X: np.ndarray) -> ReadoutResponses:
    exc_idx = excitatory_cortex_indices(brain).cpu().numpy()
    # Preserve integer spike counts for the original top-k tie ordering.
    spikes = np.empty((len(X), len(exc_idx)), dtype=np.int32)
    voltages = np.empty((len(X), len(exc_idx)), dtype=np.float32)
    reuse = can_reuse_inference(brain)
    for i, x in enumerate(X):
        spikes[i], voltages[i] = present_inference_sample(brain, x, ASSIGN_PRESENT_STEPS)
        # Every transient will be reset before the next independent image.
        # Keep the historical rest schedule for all other protocols.
        if not reuse:
            reset_brain_state(brain)
    return ReadoutResponses(exc_idx, spikes, voltages)


def normalize_feedforward_weights(brain: Brain, target_sum: float | list[float] | torch.Tensor) -> None:
    """Normalize incoming weights to one scalar or a target per cortex neuron."""
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    weights = proj.syn_weight[:ns]
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    cortex = brain.regions["cortex"]
    types = cortex.neuron_type[:cortex.n_neurons]
    targets = torch.as_tensor(target_sum, dtype=torch.float32, device=weights.device)
    if targets.ndim > 1 or (targets.ndim == 1 and len(targets) != cortex.n_neurons):
        raise ValueError("Normalization targets must be scalar or one per cortex neuron")
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError("Normalization targets must be finite and nonnegative")

    exc_mask = (types[post] == NeuronType.EXCITATORY.value) & alive
    if not torch.any(exc_mask):
        return

    sums = torch.zeros(cortex.n_neurons, dtype=torch.float32, device=DEVICE)
    sums.index_add_(0, post[exc_mask], weights[exc_mask].to(torch.float32))

    post_sums = sums[post]
    scalable = exc_mask & (post_sums > 1e-9)
    if torch.any(scalable):
        if isinstance(target_sum, (int, float)):
            # Keep scalar/Tensor division: replacing it with Tensor/Tensor
            # changes rounding and can alter a subsequent spike trajectory.
            selected_targets = target_sum
        else:
            selected_targets = targets if targets.ndim == 0 else targets[post[scalable]]
        weights[scalable] *= (selected_targets / post_sums[scalable]).to(torch.float32)

    weights = torch.clamp(weights, proj.syn_min_weight[:ns], proj.syn_max_weight[:ns])
    proj.syn_weight[:ns] = weights


def compute_norm_target(brain: Brain, *, per_neuron: bool = False) -> float | torch.Tensor:
    """Fit normalization budgets from weights, without using image labels."""
    exc_idx = excitatory_cortex_indices(brain)
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    weight_sums = torch.zeros(brain.regions["cortex"].n_neurons, dtype=torch.float32, device=DEVICE)
    weight_sums.index_add_(0, post[alive], proj.syn_weight[:ns][alive].to(torch.float32))
    if per_neuron:
        return weight_sums
    return float(weight_sums[exc_idx].mean().item())


def assign_neuron_labels(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    responses = collect_readout_responses(brain, X)
    return _labels_from_responses(responses, y, classes)


def _labels_from_responses(responses, y, classes):
    per_class = np.zeros((len(classes), len(responses.exc_indices)), dtype=np.float32)

    class_to_row = {cls: i for i, cls in enumerate(classes)}
    for counts, label in zip(responses.spikes, y):
        active = np.flatnonzero(counts > 0)
        if len(active) > 0:
            top_k = min(ASSIGN_TOP_K, len(active))
            top_local = active[np.argsort(counts[active])[-top_k:]]
            per_class[class_to_row[int(label)], top_local] += 1.0

    labels = np.full(len(responses.exc_indices), -1, dtype=np.int64)
    active = per_class.max(axis=0) > 0
    labels[active] = np.asarray(classes, dtype=np.int64)[np.argmax(per_class[:, active], axis=0)]
    return responses.exc_indices, labels


def build_response_templates(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    responses = collect_readout_responses(brain, X)
    return _templates_from_responses(responses, y, classes)


def _templates_from_responses(responses, y, classes):
    spike_arr = responses.spikes.astype(np.float32)
    voltage_arr = responses.voltages

    spike_templates = np.zeros((len(classes), spike_arr.shape[1]), dtype=np.float32)
    voltage_templates = np.zeros((len(classes), voltage_arr.shape[1]), dtype=np.float32)
    for i, cls in enumerate(classes):
        cls_spike = spike_arr[y == cls]
        cls_voltage = voltage_arr[y == cls]
        if len(cls_spike) > 0:
            spike_templates[i] = cls_spike.mean(axis=0)
        if len(cls_voltage) > 0:
            voltage_templates[i] = cls_voltage.mean(axis=0)

    spike_templates /= (np.linalg.norm(spike_templates, axis=1, keepdims=True) + 1e-9)
    voltage_templates /= (np.linalg.norm(voltage_templates, axis=1, keepdims=True) + 1e-9)
    return spike_templates, voltage_templates


def predict_sample(
    brain: Brain,
    x: np.ndarray,
    spike_templates: np.ndarray,
    voltage_templates: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[int, np.ndarray]:
    classes = CLASSES if classes is None else classes
    if TEST_REPEATS < 1:
        raise ValueError("TEST_REPEATS must be positive")
    counts = np.zeros(spike_templates.shape[1], dtype=np.float32)
    voltage_sum = np.zeros(spike_templates.shape[1], dtype=np.float32)
    reuse = can_reuse_inference(brain)
    for repeat in range(TEST_REPEATS):
        if repeat == 0 or not reuse:
            spike_counts, mean_voltage = present_inference_sample(brain, x, TEST_PRESENT_STEPS)
        # Keep the original accumulation order, including its float32 rounding.
        # Only the repeated simulation is redundant, not the score definition.
        counts += spike_counts
        voltage_sum += mean_voltage
        if not reuse:
            reset_brain_state(brain)

    return classify_response(counts, voltage_sum, spike_templates, voltage_templates, classes)


def classify_response(counts, voltage_sum, spike_templates, voltage_templates, classes=None):
    """Apply the template decoder to already accumulated responses, without simulation."""
    classes = CLASSES if classes is None else classes
    spike_query = counts / (np.linalg.norm(counts) + 1e-9)
    voltage_query = voltage_sum / (np.linalg.norm(voltage_sum) + 1e-9)
    scores = (
        SPIKE_SCORE_WEIGHT * (spike_templates @ spike_query)
        + VOLTAGE_SCORE_WEIGHT * (voltage_templates @ voltage_query)
    )
    if scores.max() <= 0:
        return -1, scores
    return int(classes[int(np.argmax(scores))]), scores


def _inference_brain(brain: Brain) -> Brain:
    """Snapshot a fixed network for readout fitting and held-out evaluation."""
    return inference_brain(brain, independent=INDEPENDENT_INFERENCE)


def evaluate(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    spike_templates: np.ndarray,
    voltage_templates: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray]:
    classes = CLASSES if classes is None else classes
    eval_brain = _inference_brain(brain)
    preds = np.full(len(X), -1, dtype=np.int64)
    for i, x in enumerate(X):
        preds[i], _ = predict_sample(
            eval_brain, x, spike_templates, voltage_templates, classes,
        )
    return float(np.mean(preds == y)), preds


def build_readout_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    readout_per_class: int = READOUT_PER_CLASS,
    classes: tuple[int, ...] | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    readout_idx = _balanced_subset(y_train, classes, readout_per_class, seed=seed)
    return X_train[readout_idx], y_train[readout_idx]


def neuron_label_counts(
    neuron_labels: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> dict[int, int]:
    classes = CLASSES if classes is None else classes
    return {int(cls): int(np.sum(neuron_labels == cls)) for cls in classes}


def train_unsupervised(
    brain: Brain,
    X_train: np.ndarray,
    epochs: int = EPOCHS,
    train_present_steps: int = TRAIN_PRESENT_STEPS,
    seed: int = SEED,
    log_every: int = 50,
) -> float:
    norm_target = compute_norm_target(brain)
    print(f"  Weight normalization target: {norm_target:.1f}")

    t0 = time.time()
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(X_train))
        for j, idx in enumerate(order, start=1):
            present_sample(
                brain, X_train[idx], train_present_steps, learn=True,
                collect_responses=False,
            )
            normalize_feedforward_weights(brain, norm_target)
            reset_brain_state(brain)
            if log_every > 0 and (j % log_every == 0 or j == len(order)):
                print(
                    f"  Epoch {epoch + 1}/{epochs}  "
                    f"sample {j:4d}/{len(order)}  "
                    f"elapsed={time.time() - t0:5.1f}s"
                )
        proj = brain.get_projection("input", "cortex")
        w = proj.syn_weight[:proj.n_synapses]
        print(
            f"  -> after epoch {epoch + 1}: "
            f"feedforward weights mean={w.mean().item():.4f} max={w.max().item():.4f}"
        )

    print(f"  Training time: {time.time() - t0:.1f}s")
    return norm_target


def build_readout(
    brain: Brain,
    X_readout: np.ndarray,
    y_readout: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    readout_brain = _inference_brain(brain)
    if can_reuse_inference(readout_brain):
        responses = collect_readout_responses(readout_brain, X_readout)
        exc_idx, neuron_labels = _labels_from_responses(responses, y_readout, classes)
        spike_templates, voltage_templates = _templates_from_responses(
            responses, y_readout, classes,
        )
        return exc_idx, neuron_labels, spike_templates, voltage_templates
    exc_idx, neuron_labels = assign_neuron_labels(
        readout_brain, X_readout, y_readout, classes,
    )
    spike_templates, voltage_templates = build_response_templates(
        readout_brain, X_readout, y_readout, classes,
    )
    return exc_idx, neuron_labels, spike_templates, voltage_templates


# --- Main -------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("  MNIST BENCHMARK - Unsupervised STDP + Template Readout")
    print(f"  (10-class, {N_INPUT}+{N_CORTEX_EXC}+{N_CORTEX_INH}, {TRAIN_PER_CLASS}/class, {TRAIN_PRESENT_STEPS}ms)")
    print("=" * 68)

    t0 = time.time()
    X_train, y_train, X_test, y_test = load_reduced_mnist(
        classes=CLASSES,
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
    )
    elapsed = time.time() - t0
    print(f"\nLoaded reduced MNIST in {elapsed:.1f}s")
    print(
        f"  Classes: {list(CLASSES)} | "
        f"Train: {len(X_train)} samples | Test: {len(X_test)} samples | "
        f"Input neurons: {N_INPUT} ({IMAGE_SIDE}x{IMAGE_SIDE})"
    )

    print("\n" + "-" * 68)
    print("  Building brain")
    print("-" * 68)
    brain = build_brain(seed=SEED)
    print(brain.summary())

    print("\n" + "-" * 68)
    print("  Unsupervised training")
    print("-" * 68)
    train_unsupervised(brain, X_train)

    print("\n" + "-" * 68)
    print("  Building readout")
    print("-" * 68)
    t3 = time.time()
    # Use a balanced subset for readout to save time (labelling doesn't need all training data).
    X_readout, y_readout = build_readout_subset(X_train, y_train)
    print(f"  Using {len(X_readout)} samples for readout ({READOUT_PER_CLASS}/class)")
    exc_idx, neuron_labels, spike_templates, voltage_templates = build_readout(brain, X_readout, y_readout)
    labelled = neuron_labels >= 0
    print(f"  Excitatory cortex neurons: {len(exc_idx)}")
    print(f"  Labelled excitatory neurons: {int(np.sum(labelled))}/{len(exc_idx)}")
    for cls, count in neuron_label_counts(neuron_labels).items():
        print(f"    class {cls}: {count} neurons")
    print(f"  Readout build time: {time.time() - t3:.1f}s")

    print("\n" + "-" * 68)
    print("  Evaluation")
    print("-" * 68)
    t4 = time.time()
    acc, preds = evaluate(brain, X_test, y_test, spike_templates, voltage_templates)
    no_response = int(np.sum(preds < 0))
    cm = confusion_matrix(y_test, preds, labels=CLASSES)
    eval_time = time.time() - t4

    print(f"  Test accuracy:            {acc:.1%}")
    print(f"  Samples with no response: {no_response}/{len(y_test)}")
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    print("           " + "  ".join(f"{c:>5d}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = "  ".join(f"{v:>5d}" for v in cm[i])
        print(f"    {cls:>5d}:  {row}")

    total_time = time.time() - t0
    print(f"\n  Eval time: {eval_time:.1f}s | Total wall time: {total_time:.0f}s")
    print("\n" + "=" * 68)
    if acc >= 0.50:
        print("  OK: 10-class template accuracy meets the 50% check.")
    elif acc >= 0.30:
        print("  PARTIAL: above chance (10%) but not yet at target (>50%).")
    else:
        print("  FAIL: 10-class benchmark does not yet validate the protocol.")
    print("  Accuracy alone does not establish an STDP gain; use matched learning controls.")
    print("=" * 68)


if __name__ == "__main__":
    main()
