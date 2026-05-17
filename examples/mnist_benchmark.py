"""
MNIST benchmark — unsupervised STDP + template readout.

Full Step 3 benchmark following the Diehl & Cook 2015 architecture:

- true MNIST loaded from OpenML, all 10 digit classes
- 784 input neurons (one per pixel, rate coded)
- 1600 excitatory + 1600 inhibitory cortex neurons with WTA microcircuit
- unsupervised STDP on input->cortex feedforward synapses
- adaptive excitability thresholds + per-neuron weight normalization
- L1 intensity equalization for balanced cross-class drive
- class readout from cortex response templates (spike + voltage blend)

Target: 85-90% accuracy (Diehl & Cook 2015 achieved 95% with 6400 exc neurons).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import confusion_matrix, quiet_steps
from src.device import DEVICE

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    print("This prototype requires scikit-learn:")
    print("  pip install scikit-learn")
    raise SystemExit(1)

from src.brain import Brain
from src.neuron import FiringPattern, NeuronType
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


# --- Dataset scope ----------------------------------------------------------

CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
TRAIN_PER_CLASS = 6000
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
INH_LATERAL_WEIGHT = 12.0
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


# --- Brain construction -----------------------------------------------------


def build_brain(seed: int = SEED) -> Brain:
    brain = Brain(dt=1.0, seed=seed)

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
    wire_cortex_microcircuit(cortex)

    brain.connect_regions("input", "cortex", density=INPUT_TO_CORTEX_DENSITY)
    brain.freeze_plasticity()
    proj = brain.enable_projection_plasticity(
        "input",
        "cortex",
        A_plus=STDP_A_PLUS * STDP_SCALE,
        A_minus=STDP_A_MINUS * STDP_SCALE,
    )
    ns = proj.n_synapses
    proj.syn_weight[:ns] *= INPUT_WEIGHT_BOOST
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
    brain.reward_stdp.apply_target = lambda *args, **kwargs: 0
    brain.reset_traces()
    brain.freeze_structural_plasticity()
    brain.memory.capture_trace = lambda *args, **kwargs: None
    brain.memory.consolidate = lambda *args, **kwargs: 0

    # Keep benchmark-level theta settings explicit so sweeps can override them.
    brain.homeostasis.theta_plus = THETA_PLUS
    brain.homeostasis.theta_leak = THETA_LEAK
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE

    return brain


def wire_cortex_microcircuit(cortex: Region) -> None:
    """Wire 1:1 matched exc-inh pairs (Diehl & Cook 2015 WTA).

    Each exc[i] drives exactly inh[i]; each inh[i] suppresses every
    exc[j != i].  This creates much sharper winner-take-all competition
    than random sparse connectivity.
    """
    n = cortex.n_neurons
    if n == 0:
        return

    types = cortex.neuron_type[:n]
    exc_idx = torch.where(types == NeuronType.EXCITATORY.value)[0]
    inh_idx = torch.where(types == NeuronType.INHIBITORY.value)[0]
    n_matched = min(len(exc_idx), len(inh_idx))
    if n_matched == 0:
        return

    # 1:1 exc[i] -> inh[i] with strong, fast connections.
    cortex.add_synapses(
        exc_idx[:n_matched].to(torch.int32),
        inh_idx[:n_matched].to(torch.int32),
        torch.full((n_matched,), EXC_TO_INH_WEIGHT, dtype=torch.float32, device=DEVICE),
        torch.full((n_matched,), 1.0, dtype=torch.float32, device=DEVICE),
        torch.full((n_matched,), NeurotransmitterType.GLUTAMATE.value, dtype=torch.int32, device=DEVICE),
    )

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

def downsample_images(X: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    n = X.shape[0]
    side = int(np.sqrt(X.shape[1]))
    imgs = X.reshape(n, side, side)
    new_side = side // factor
    pooled = imgs.reshape(n, new_side, factor, new_side, factor).mean(axis=(2, 4))
    flat = pooled.reshape(n, new_side * new_side)
    max_per_image = np.maximum(flat.max(axis=1, keepdims=True), 1e-6)
    flat = flat / max_per_image
    # Equalize total stimulus across images so sparse digits (e.g. "1")
    # get proportionally stronger drive than dense ones (e.g. "0").
    l1 = flat.sum(axis=1, keepdims=True)
    target_l1 = np.median(l1)
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

    X = downsample_images(X)
    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        train_idx.extend(idx[:train_per_class])
        test_idx.extend(idx[train_per_class:train_per_class + test_per_class])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


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

def reset_brain_state(brain: Brain, rest_steps: int = REST_STEPS) -> None:
    quiet_steps(brain, rest_steps)
    for region in brain.regions.values():
        n = region.n_neurons
        region.current[:n] = 0.0
        region.fired[:n] = False
    brain.reset_traces()


def apply_feedforward_stdp(brain: Brain) -> int:
    proj = brain.get_projection("input", "cortex")
    source = brain.regions["input"]
    target = brain.regions["cortex"]
    ns = proj.n_synapses
    if ns == 0:
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
    )


def present_sample(
    brain: Brain,
    x: np.ndarray,
    n_steps: int,
    learn: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cortex = brain.regions["cortex"]
    exc_idx = excitatory_cortex_indices(brain)
    before = cortex.total_spikes[exc_idx].clone()
    voltage_sum = torch.zeros(len(exc_idx), dtype=torch.float32, device=cortex.v.device)

    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.step()
        voltage_sum += cortex.v[exc_idx]
        if learn:
            apply_feedforward_stdp(brain)

    counts = cortex.total_spikes[exc_idx] - before
    mean_voltage = voltage_sum / max(n_steps, 1)
    return counts.cpu().numpy(), mean_voltage.cpu().numpy()


def normalize_feedforward_weights(brain: Brain, target_sum: float) -> None:
    """Normalize incoming feedforward weight sum per excitatory cortex neuron."""
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    weights = proj.syn_weight[:ns]
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    cortex = brain.regions["cortex"]
    types = cortex.neuron_type[:cortex.n_neurons]

    exc_mask = (types[post] == NeuronType.EXCITATORY.value) & alive
    if not torch.any(exc_mask):
        return

    sums = torch.zeros(cortex.n_neurons, dtype=torch.float32, device=DEVICE)
    sums.index_add_(0, post[exc_mask], weights[exc_mask].to(torch.float32))

    post_sums = sums[post]
    scalable = exc_mask & (post_sums > 1e-9)
    if torch.any(scalable):
        weights[scalable] *= (target_sum / post_sums[scalable]).to(torch.float32)

    weights = torch.clamp(weights, proj.syn_min_weight[:ns], proj.syn_max_weight[:ns])
    proj.syn_weight[:ns] = weights


def compute_norm_target(brain: Brain) -> float:
    exc_idx = excitatory_cortex_indices(brain)
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    weight_sums = torch.zeros(brain.regions["cortex"].n_neurons, dtype=torch.float32, device=DEVICE)
    weight_sums.index_add_(0, post[alive], proj.syn_weight[:ns][alive].to(torch.float32))
    return float(weight_sums[exc_idx].mean().item())


def assign_neuron_labels(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    exc_idx = excitatory_cortex_indices(brain)
    per_class = np.zeros((len(classes), len(exc_idx)), dtype=np.float32)

    class_to_row = {cls: i for i, cls in enumerate(classes)}
    for x, label in zip(X, y):
        counts, _ = present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False)
        active = np.flatnonzero(counts > 0)
        if len(active) > 0:
            top_k = min(ASSIGN_TOP_K, len(active))
            top_local = active[np.argsort(counts[active])[-top_k:]]
            per_class[class_to_row[int(label)], top_local] += 1.0
        reset_brain_state(brain)

    labels = np.full(len(exc_idx), -1, dtype=np.int64)
    active = per_class.max(axis=0) > 0
    labels[active] = np.asarray(classes, dtype=np.int64)[np.argmax(per_class[:, active], axis=0)]
    return exc_idx.cpu().numpy(), labels


def build_response_templates(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    spike_responses = []
    voltage_responses = []
    for x in X:
        counts, mean_voltage = present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False)
        spike_responses.append(counts.astype(np.float64))
        voltage_responses.append(mean_voltage.astype(np.float64))
        reset_brain_state(brain)
    spike_arr = np.asarray(spike_responses, dtype=np.float32)
    voltage_arr = np.asarray(voltage_responses, dtype=np.float32)

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
    counts = np.zeros(spike_templates.shape[1], dtype=np.float32)
    voltage_sum = np.zeros(spike_templates.shape[1], dtype=np.float32)
    for _ in range(TEST_REPEATS):
        spike_counts, mean_voltage = present_sample(brain, x, TEST_PRESENT_STEPS, learn=False)
        counts += spike_counts
        voltage_sum += mean_voltage
        reset_brain_state(brain)

    spike_query = counts / (np.linalg.norm(counts) + 1e-9)
    voltage_query = voltage_sum / (np.linalg.norm(voltage_sum) + 1e-9)
    scores = (
        SPIKE_SCORE_WEIGHT * (spike_templates @ spike_query)
        + VOLTAGE_SCORE_WEIGHT * (voltage_templates @ voltage_query)
    )
    if scores.max() <= 0:
        return -1, scores
    return int(classes[int(np.argmax(scores))]), scores


def evaluate(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    spike_templates: np.ndarray,
    voltage_templates: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray]:
    classes = CLASSES if classes is None else classes
    preds = np.full(len(X), -1, dtype=np.int64)
    for i, x in enumerate(X):
        preds[i], _ = predict_sample(brain, x, spike_templates, voltage_templates, classes)
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
            present_sample(brain, X_train[idx], train_present_steps, learn=True)
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
    exc_idx, neuron_labels = assign_neuron_labels(brain, X_readout, y_readout, classes)
    spike_templates, voltage_templates = build_response_templates(brain, X_readout, y_readout, classes)
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
        print("  OK: 10-class MNIST benchmark validates unsupervised STDP + readout.")
    elif acc >= 0.30:
        print("  PARTIAL: above chance (10%) but not yet at target (>50%).")
    else:
        print("  FAIL: 10-class benchmark does not yet validate the protocol.")
    print("=" * 68)


if __name__ == "__main__":
    main()
