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


# --- Dataset / prototype scope ---------------------------------------------

CLASSES = (0, 1, 2, 3)
TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 10
DOWNSAMPLE = 2
IMAGE_SIDE = 28 // DOWNSAMPLE
N_INPUT = IMAGE_SIDE * IMAGE_SIDE


# --- Network / training hyperparameters ------------------------------------

N_CORTEX = 120
CORTEX_CONNECTIVITY = 0.05
INPUT_TO_CORTEX_DENSITY = 0.25
INPUT_WEIGHT_BOOST = 6.0

TRAIN_PRESENT_STEPS = 15
ASSIGN_PRESENT_STEPS = 20
TEST_PRESENT_STEPS = 20
REST_STEPS = 10
EPOCHS = 1
ASSIGN_TOP_K = 12
TEST_REPEATS = 2

ENCODER_MAX_CURRENT = 30.0
ENCODER_NOISE = 0.03

STDP_SCALE = 0.4
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
        n_neurons=N_CORTEX,
        connectivity=CORTEX_CONNECTIVITY,
        max_neurons=N_CORTEX,
        seed=seed + 2,
    )

    brain.connect_regions("input", "cortex", density=INPUT_TO_CORTEX_DENSITY)
    proj = feedforward_proj(brain)
    ns = proj.n_synapses
    proj.syn_weight[:ns] *= INPUT_WEIGHT_BOOST
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
    return flat / max_per_image


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
) -> np.ndarray:
    cortex = brain.regions["cortex"]
    exc_idx = excitatory_cortex_indices(brain)
    before = cortex.total_spikes[exc_idx].copy()

    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.step()
        if learn:
            apply_feedforward_stdp(brain)

    counts = cortex.total_spikes[exc_idx] - before
    return counts


def assign_neuron_labels(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    classes = CLASSES if classes is None else classes
    exc_idx = excitatory_cortex_indices(brain)
    per_class = np.zeros((len(classes), len(exc_idx)), dtype=np.float64)

    class_to_row = {cls: i for i, cls in enumerate(classes)}
    for x, label in zip(X, y):
        counts = present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False)
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
) -> np.ndarray:
    classes = CLASSES if classes is None else classes
    responses = []
    for x in X:
        responses.append(present_sample(brain, x, ASSIGN_PRESENT_STEPS, learn=False).astype(np.float64))
        reset_brain_state(brain)
    responses_arr = np.asarray(responses, dtype=np.float64)

    templates = np.zeros((len(classes), responses_arr.shape[1]), dtype=np.float64)
    for i, cls in enumerate(classes):
        cls_resp = responses_arr[y == cls]
        if len(cls_resp) > 0:
            templates[i] = cls_resp.mean(axis=0)
    norms = np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9
    return templates / norms


def predict_sample(
    brain: Brain,
    x: np.ndarray,
    templates: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[int, np.ndarray]:
    classes = CLASSES if classes is None else classes
    counts = np.zeros(templates.shape[1], dtype=np.float64)
    for _ in range(TEST_REPEATS):
        counts += present_sample(brain, x, TEST_PRESENT_STEPS, learn=False)
        reset_brain_state(brain)
    if counts.sum() <= 0:
        return -1, np.zeros(len(classes), dtype=np.float64)

    query = counts / (np.linalg.norm(counts) + 1e-9)
    scores = templates @ query
    if scores.max() <= 0:
        return -1, scores
    return int(classes[int(np.argmax(scores))]), scores


def evaluate(
    brain: Brain,
    X: np.ndarray,
    y: np.ndarray,
    templates: np.ndarray,
    classes: tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray]:
    classes = CLASSES if classes is None else classes
    preds = np.full(len(X), -1, dtype=np.int64)
    for i, x in enumerate(X):
        preds[i], _ = predict_sample(brain, x, templates, classes)
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

    print("\n" + "-" * 68)
    print("  Unsupervised training")
    print("-" * 68)
    t1 = time.time()
    for epoch in range(EPOCHS):
        order = np.random.default_rng(SEED + epoch).permutation(len(X_train))
        for j, idx in enumerate(order, start=1):
            present_sample(brain, X_train[idx], TRAIN_PRESENT_STEPS, learn=True)
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
            f"feedforward weights mean={w.mean():.4f} max={w.max():.4f}"
        )

    print("\n" + "-" * 68)
    print("  Building readout")
    print("-" * 68)
    exc_idx, neuron_labels = assign_neuron_labels(brain, X_train, y_train)
    templates = build_response_templates(brain, X_train, y_train)
    labelled = neuron_labels >= 0
    print(f"  Excitatory cortex neurons: {len(exc_idx)}")
    print(f"  Labelled excitatory neurons: {int(np.sum(labelled))}/{len(exc_idx)}")
    for cls in CLASSES:
        print(f"    class {cls}: {int(np.sum(neuron_labels == cls))} neurons")

    print("\n" + "-" * 68)
    print("  Evaluation")
    print("-" * 68)
    acc, preds = evaluate(brain, X_test, y_test, templates)
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
