"""
Reduced MNIST prototype for validating unsupervised STDP + readout.

This is not the full Step 3 benchmark yet. Instead it validates the core
protocol on a smaller setting:

- true MNIST digits loaded from OpenML
- reduced class set (default: digits 0-3)
- 28x28 images downsampled to 14x14 (196 input neurons)
- unsupervised STDP only on input->cortex feedforward synapses
- class readout from cortex response templates built after STDP training

The goal is to prove that the current simulator can support the classic
"STDP features + neuron-label readout" workflow before attempting the
full 784->400 excitatory / 400 inhibitory MNIST benchmark.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    print("This prototype requires scikit-learn:")
    print("  pip install scikit-learn")
    raise SystemExit(1)

from src.brain import Brain, Projection
from src.neuron import FiringPattern, NeuronType
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


# --- Dataset / prototype scope ---------------------------------------------

CLASSES = (0, 1, 2, 3)
TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 10
DOWNSAMPLE = 2
IMAGE_SIDE = 28 // DOWNSAMPLE
N_INPUT = IMAGE_SIDE * IMAGE_SIDE


# --- Network / training hyperparameters ------------------------------------

N_CORTEX_EXC = 96
N_CORTEX_INH = 24
CORTEX_CONNECTIVITY = 0.0
EXC_TO_INH_DENSITY = 0.40
INH_TO_EXC_DENSITY = 0.80
INH_LATERAL_WEIGHT = 8.0
INPUT_TO_CORTEX_DENSITY = 0.25
INPUT_WEIGHT_BOOST = 6.0

TRAIN_PRESENT_STEPS = 25
ASSIGN_PRESENT_STEPS = 25
TEST_PRESENT_STEPS = 25
REST_STEPS = 10
EPOCHS = 4
ASSIGN_TOP_K = 12
TEST_REPEATS = 3
SPIKE_SCORE_WEIGHT = 0.7
VOLTAGE_SCORE_WEIGHT = 0.3

ENCODER_MAX_CURRENT = 30.0
ENCODER_NOISE = 0.03

STDP_SCALE = 0.8

THETA_PLUS = 0.15
THETA_DECAY = 1e-6

SEED = 42


# --- Brain construction -----------------------------------------------------

def _add_seeded_region(
    brain: Brain,
    name: str,
    region_type: RegionType,
    n_neurons: int,
    connectivity: float,
    max_neurons: int,
    seed: int,
) -> Region:
    region = Region(name, region_type, max_neurons, dt=brain.dt)
    region._rng = np.random.default_rng(seed)
    region.populate(n_neurons, connectivity)
    brain.regions[name] = region
    brain.oscillators.add_region(name, region_type)
    return region


def build_brain(seed: int = SEED) -> Brain:
    brain = Brain(dt=1.0, seed=seed)

    # Input region: all sensory neurons are excitatory so every pixel can project.
    input_region = _add_seeded_region(
        brain,
        "input",
        RegionType.SENSORY,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_INPUT,
        seed=seed + 1,
    )
    for _ in range(N_INPUT):
        input_region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)

    cortex = _add_seeded_region(
        brain,
        "cortex",
        RegionType.ASSOCIATION,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_CORTEX_EXC + N_CORTEX_INH,
        seed=seed + 2,
    )
    cortex_rng = np.random.default_rng(seed + 20)
    for _ in range(N_CORTEX_EXC):
        cortex.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
    for _ in range(N_CORTEX_INH):
        cortex.add_neuron(NeuronType.INHIBITORY, FiringPattern.FAST_SPIKING)
    wire_cortex_microcircuit(cortex, cortex_rng)

    brain.connect_regions("input", "cortex", density=INPUT_TO_CORTEX_DENSITY)
    proj = feedforward_proj(brain)
    ns = proj.n_synapses
    proj.syn_weight[:ns] *= INPUT_WEIGHT_BOOST
    np.clip(proj.syn_weight[:ns], proj.syn_min_weight[:ns],
            proj.syn_max_weight[:ns], out=proj.syn_weight[:ns])
    proj.syn_A_plus[:ns] = 0.01 * STDP_SCALE
    proj.syn_A_minus[:ns] = 0.012 * STDP_SCALE

    # Learn only on feedforward synapses onto excitatory cortex neurons.
    cortex_types = cortex.neuron_type[:cortex.n_neurons]
    exc_post = cortex_types[proj.syn_post[:ns]] == NeuronType.EXCITATORY
    proj.syn_A_plus[:ns][~exc_post] = 0.0
    proj.syn_A_minus[:ns][~exc_post] = 0.0

    # Disable unrelated mechanisms so the prototype isolates STDP + readout.
    brain.reward_stdp.apply_target = lambda *args, **kwargs: 0
    brain.reward_stdp.dopamine.clear()
    brain.growth.growth_interval = 10**9
    brain.memory.capture_trace = lambda *args, **kwargs: None
    brain.memory.consolidate = lambda *args, **kwargs: 0
    brain.metaplasticity.update_thresholds = lambda *args, **kwargs: None

    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE
    brain.encoder._rng = np.random.default_rng(seed + 100)

    return brain


def wire_cortex_microcircuit(cortex: Region, rng: np.random.Generator) -> None:
    n = cortex.n_neurons
    if n == 0:
        return

    types = cortex.neuron_type[:n]
    exc_idx = np.where(types == NeuronType.EXCITATORY)[0]
    inh_idx = np.where(types == NeuronType.INHIBITORY)[0]

    # Sparse recurrent excitation among excitatory neurons.
    if len(exc_idx) > 1 and CORTEX_CONNECTIVITY > 0:
        conn = rng.random((len(exc_idx), len(exc_idx))) < CORTEX_CONNECTIVITY
        np.fill_diagonal(conn, False)
        pre_local, post_local = np.where(conn)
        if len(pre_local) > 0:
            cortex.add_synapses(
                exc_idx[pre_local].astype(np.int32),
                exc_idx[post_local].astype(np.int32),
                rng.exponential(0.5, size=len(pre_local)),
                rng.uniform(1.0, 5.0, size=len(pre_local)),
                np.full(len(pre_local), NeurotransmitterType.GLUTAMATE.value),
            )

    # Excitatory neurons recruit inhibitory interneurons.
    if len(exc_idx) > 0 and len(inh_idx) > 0 and EXC_TO_INH_DENSITY > 0:
        conn = rng.random((len(exc_idx), len(inh_idx))) < EXC_TO_INH_DENSITY
        pre_local, post_local = np.where(conn)
        if len(pre_local) > 0:
            cortex.add_synapses(
                exc_idx[pre_local].astype(np.int32),
                inh_idx[post_local].astype(np.int32),
                rng.exponential(0.4, size=len(pre_local)),
                rng.uniform(1.0, 3.0, size=len(pre_local)),
                np.full(len(pre_local), NeurotransmitterType.GLUTAMATE.value),
            )

    # Fast inhibitory feedback enforces winner-take-all competition.
    if len(inh_idx) > 0 and len(exc_idx) > 0 and INH_TO_EXC_DENSITY > 0:
        conn = rng.random((len(inh_idx), len(exc_idx))) < INH_TO_EXC_DENSITY
        pre_local, post_local = np.where(conn)
        if len(pre_local) > 0:
            cortex.add_synapses(
                inh_idx[pre_local].astype(np.int32),
                exc_idx[post_local].astype(np.int32),
                np.full(len(pre_local), INH_LATERAL_WEIGHT),
                rng.uniform(1.0, 2.0, size=len(pre_local)),
                np.full(len(pre_local), NeurotransmitterType.GABA.value),
            )


def feedforward_proj(brain: Brain) -> Projection:
    return next(
        p for p in brain.projections
        if p.source_name == "input" and p.target_name == "cortex"
    )


def excitatory_cortex_indices(brain: Brain) -> np.ndarray:
    cortex = brain.regions["cortex"]
    types = cortex.neuron_type[:cortex.n_neurons]
    alive = cortex.neuron_alive[:cortex.n_neurons]
    return np.where((types == NeuronType.EXCITATORY) & alive)[0]


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


# --- Simulation helpers -----------------------------------------------------

def reset_brain_state(brain: Brain, rest_steps: int = REST_STEPS) -> None:
    for _ in range(rest_steps):
        brain.step()
    for region in brain.regions.values():
        n = region.n_neurons
        region.current[:n] = 0.0
        region.fired[:n] = False
    for proj in brain.projections:
        ns = proj.n_synapses
        if ns:
            proj.syn_eligibility[:ns] = 0.0


def apply_feedforward_stdp(brain: Brain) -> int:
    proj = feedforward_proj(brain)
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
    theta: np.ndarray | None = None,
    update_theta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cortex = brain.regions["cortex"]
    exc_idx = excitatory_cortex_indices(brain)
    before = cortex.total_spikes[exc_idx].copy()
    voltage_sum = np.zeros(len(exc_idx), dtype=np.float64)

    for _ in range(n_steps):
        brain.stimulate("input", x)
        if theta is not None:
            cortex.current[exc_idx] -= theta
        brain.step()
        voltage_sum += cortex.v[exc_idx]
        if learn:
            apply_feedforward_stdp(brain)
        if update_theta and theta is not None:
            fired_exc = cortex.fired[exc_idx]
            theta += THETA_PLUS * fired_exc
            theta *= (1.0 - THETA_DECAY)

    counts = cortex.total_spikes[exc_idx] - before
    mean_voltage = voltage_sum / max(n_steps, 1)
    return counts, mean_voltage


def normalize_feedforward_weights(brain: Brain, target_sum: float) -> None:
    """Normalize incoming feedforward weight sum per excitatory cortex neuron."""
    proj = feedforward_proj(brain)
    ns = proj.n_synapses
    exc_idx = excitatory_cortex_indices(brain)
    weights = proj.syn_weight[:ns]
    post = proj.syn_post[:ns]
    alive = proj.syn_alive[:ns]
    min_w = proj.syn_min_weight[:ns]
    max_w = proj.syn_max_weight[:ns]

    for idx in exc_idx:
        mask = (post == idx) & alive
        total = weights[mask].sum()
        if total > 1e-9:
            weights[mask] *= target_sum / total
    np.clip(weights, min_w, max_w, out=weights)


def assign_neuron_labels(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
    theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    exc_idx = excitatory_cortex_indices(brain)
    per_class = np.zeros((len(classes), len(exc_idx)), dtype=np.float64)

    class_to_row = {cls: i for i, cls in enumerate(classes)}
    for x, label in zip(X, y):
        counts, _ = present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False, theta=theta)
        active = np.flatnonzero(counts > 0)
        if len(active) > 0:
            top_k = min(ASSIGN_TOP_K, len(active))
            top_local = active[np.argsort(counts[active])[-top_k:]]
            per_class[class_to_row[int(label)], top_local] += 1.0
        reset_brain_state(brain)

    labels = np.full(len(exc_idx), -1, dtype=np.int64)
    active = per_class.max(axis=0) > 0
    labels[active] = np.asarray(classes, dtype=np.int64)[np.argmax(per_class[:, active], axis=0)]
    return exc_idx, labels


def build_response_templates(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
    theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    spike_responses = []
    voltage_responses = []
    for x in X:
        counts, mean_voltage = present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False, theta=theta)
        spike_responses.append(counts.astype(np.float64))
        voltage_responses.append(mean_voltage.astype(np.float64))
        reset_brain_state(brain)
    spike_arr = np.asarray(spike_responses, dtype=np.float64)
    voltage_arr = np.asarray(voltage_responses, dtype=np.float64)

    spike_templates = np.zeros((len(classes), spike_arr.shape[1]), dtype=np.float64)
    voltage_templates = np.zeros((len(classes), voltage_arr.shape[1]), dtype=np.float64)
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
    theta: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    classes = CLASSES if classes is None else classes
    counts = np.zeros(spike_templates.shape[1], dtype=np.float64)
    voltage_sum = np.zeros(spike_templates.shape[1], dtype=np.float64)
    for _ in range(TEST_REPEATS):
        spike_counts, mean_voltage = present_sample(brain, x, TEST_PRESENT_STEPS, learn=False, theta=theta)
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
    theta: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    classes = CLASSES if classes is None else classes
    preds = np.full(len(X), -1, dtype=np.int64)
    for i, x in enumerate(X):
        preds[i], _ = predict_sample(brain, x, spike_templates, voltage_templates, classes, theta=theta)
    return float(np.mean(preds == y)), preds


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> np.ndarray:
    classes = CLASSES if classes is None else classes
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    class_to_row = {cls: i for i, cls in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if p < 0:
            continue
        cm[class_to_row[int(t)], class_to_row[int(p)]] += 1
    return cm


# --- Main -------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("  REDUCED MNIST PROTOTYPE - Unsupervised STDP + Template Readout")
    print("  (Validate the protocol before full Step 3 scaling)")
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

    # Adaptive threshold array for excitatory cortex neurons.
    exc_idx = excitatory_cortex_indices(brain)
    theta = np.zeros(len(exc_idx), dtype=np.float64)

    # Compute initial incoming weight sum per neuron for normalization.
    proj = feedforward_proj(brain)
    ns = proj.n_synapses
    post = proj.syn_post[:ns]
    alive = proj.syn_alive[:ns]
    weight_sums = np.zeros(brain.regions["cortex"].n_neurons, dtype=np.float64)
    np.add.at(weight_sums, post[alive], proj.syn_weight[:ns][alive])
    norm_target = float(weight_sums[exc_idx].mean())
    print(f"  Weight normalization target: {norm_target:.1f}")

    print("\n" + "-" * 68)
    print("  Unsupervised training")
    print("-" * 68)
    t1 = time.time()
    for epoch in range(EPOCHS):
        order = np.random.default_rng(SEED + epoch).permutation(len(X_train))
        for j, idx in enumerate(order, start=1):
            present_sample(brain, X_train[idx], TRAIN_PRESENT_STEPS, learn=True,
                           theta=theta, update_theta=True)
            normalize_feedforward_weights(brain, norm_target)
            reset_brain_state(brain)
            if j % 100 == 0 or j == len(order):
                print(
                    f"  Epoch {epoch + 1}/{EPOCHS}  "
                    f"sample {j:4d}/{len(order)}  "
                    f"elapsed={time.time() - t1:5.1f}s"
                )
        proj = feedforward_proj(brain)
        w = proj.syn_weight[:proj.n_synapses]
        print(
            f"  -> after epoch {epoch + 1}: "
            f"feedforward weights mean={w.mean():.4f} max={w.max():.4f}  "
            f"theta mean={theta.mean():.2f} max={theta.max():.2f}"
        )

    print("\n" + "-" * 68)
    print("  Building readout")
    print("-" * 68)
    exc_idx, neuron_labels = assign_neuron_labels(brain, X_train, y_train)
    spike_templates, voltage_templates = build_response_templates(brain, X_train, y_train)
    labelled = neuron_labels >= 0
    print(f"  Excitatory cortex neurons: {len(exc_idx)}")
    print(f"  Labelled excitatory neurons: {int(np.sum(labelled))}/{len(exc_idx)}")
    for cls in CLASSES:
        print(f"    class {cls}: {int(np.sum(neuron_labels == cls))} neurons")

    print("\n" + "-" * 68)
    print("  Evaluation")
    print("-" * 68)
    acc, preds = evaluate(brain, X_test, y_test, spike_templates, voltage_templates)
    no_response = int(np.sum(preds < 0))
    cm = confusion_matrix(y_test, preds)

    print(f"  Test accuracy:            {acc:.1%}")
    print(f"  Samples with no response: {no_response}/{len(y_test)}")
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    print("           " + "  ".join(f"{c:>5d}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = "  ".join(f"{v:>5d}" for v in cm[i])
        print(f"    {cls:>5d}:  {row}")

    print("\n" + "=" * 68)
    if acc >= 0.60:
        print("  OK: the reduced 4-class MNIST prototype validates STDP + template readout.")
    elif acc >= 0.50:
        print("  PARTIAL: the 4-class protocol is promising, but still needs tuning before full MNIST.")
    else:
        print("  FAIL: the reduced 4-class prototype does not yet validate the protocol.")
    print("=" * 68)


if __name__ == "__main__":
    main()
