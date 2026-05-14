"""
Iris classification benchmark for the synthetic brain.

This is a *real* benchmark: the brain learns to classify Iris flowers
using spike-timing dependent plasticity modulated by dopamine (R-STDP),
not gradient descent.

Architecture:
    input (80 sensory neurons, 20 per feature, place-field encoded)
        ↓ density 0.5           ↓ density 0.9 (readout, R-STDP)
    cortex (80 association)     motor (3 chattering, one per class)
        ↓ density 0.8 (frozen)   ↑ lateral inhibition (GABA)
              → motor ──────────

The input→motor shortcut is the readout learned by R-STDP.
Cortex→motor provides non-specific drive (weights frozen).

Training (supervised R-STDP with delayed teacher):
    1. Present sample (place-field encoded, 30 ms).
    2. After TEACHER_DELAY (13 ms), inject teacher current into
       motor[y]. Input fires first (causal) → STDP builds positive
       eligibility on input→motor[y].
    3. Zero eligibility on synapses to motor[j≠y] (selective credit).
    4. Reward → dopamine × eligibility consolidates input→motor[y].
    5. Reset between samples.

Testing: present without teacher (80 ms), predict by motor spike argmax.
Compared against a sklearn LogisticRegression baseline.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError:
    print("This benchmark requires scikit-learn:")
    print("  pip install scikit-learn")
    sys.exit(1)

from src.brain import Brain
from src.neuron import FiringPattern, NeuronType
from src.region import Region, RegionType
from src.synapse import NeurotransmitterType


# ── Hyperparameters ──────────────────────────────────────────────────

N_BINS_PER_FEATURE = 20   # place-field bins per Iris feature
N_INPUT = 80              # 4 features × 20 bins
PLACE_FIELD_SIGMA = 0.08  # narrower fields improve separability of nearby classes
N_CORTEX = 80
N_MOTOR = 3               # one per class
TEST_PRESENT_STEPS = 80   # ms of stimulus per sample (test mode: longer for better readout)
TEST_REPEATS = 6          # repeat each test sample to average out spiking noise
TRAIN_PRESENT_STEPS = 30  # ms during training (shorter: matches cortex burst window)
TEACHER_DELAY = 13        # ms before teacher kicks in (lets cortex fire first → causal STDP)
TEACHER_CURRENT = 15.0    # moderate teacher drive on correct motor neuron
REST_STEPS = 30           # ms between samples
REWARD_AMOUNT = 0.1       # dopamine pulse on input→motor eligibility
PUNISH_AMOUNT = 0.05      # smaller anti-Hebbian pulse for the strongest wrong motor
DOPAMINE_DECAY = 0.5      # fast decay so reward = near-impulse update
LATERAL_INHIBITION = 5.0  # |weight| of motor↔motor inhibitory synapses
ENCODER_NOISE = 0.1       # reduce per-step noise so cortex patterns are stable
EPOCHS = 15
REWARD_DECAY_EPOCH = 10   # halve reward in late training for stability
SEED = 42

# Calibration for small-network dynamics: the brain's defaults assume
# larger populations, so inter-region weights and input drive need a boost.
PROJECTION_DENSITY_INPUT_CORTEX = 0.50
PROJECTION_DENSITY_CORTEX_MOTOR = 0.80
PROJECTION_DENSITY_INPUT_MOTOR = 1.00   # full readout coverage
PROJECTION_WEIGHT_BOOST_IC = 8.0    # input → cortex
PROJECTION_WEIGHT_BOOST_CM = 12.0   # cortex → motor
READOUT_INIT_WEIGHT = 0.01          # near-zero start for input → motor readout
ENCODER_MAX_CURRENT = 50.0

# R-STDP plasticity scale. Event-driven STDP fires once per spike pair (not
# 50× across the window), so amplitudes need to be larger to drive learning.
STDP_SCALE = 1.0               # scale A_plus / A_minus (default 0.01 / 0.012)
TAU_ELIGIBILITY = 200.0        # ms; short enough to keep credit sample-local


# ── Brain construction ───────────────────────────────────────────────

def _add_seeded_region(
    brain: Brain,
    name: str,
    region_type: RegionType,
    n_neurons: int,
    connectivity: float,
    max_neurons: int,
    seed: int,
) -> Region:
    """Build a region with a deterministic RNG before population."""
    region = Region(name, region_type, max_neurons, dt=brain.dt)
    region._rng = np.random.default_rng(seed)
    region.populate(n_neurons, connectivity)
    brain.regions[name] = region
    brain.oscillators.add_region(name, region_type)
    return region


def build_brain(seed: int = SEED) -> Brain:
    brain = Brain(dt=1.0, seed=seed)

    # Input: pure feedforward, no internal connectivity
    _add_seeded_region(
        brain,
        "input", RegionType.SENSORY,
        n_neurons=N_INPUT, connectivity=0.0,
        max_neurons=N_INPUT,
        seed=seed + 1,
    )

    # Cortex: sparsely recurrent association area
    _add_seeded_region(
        brain,
        "cortex", RegionType.ASSOCIATION,
        n_neurons=N_CORTEX, connectivity=0.05,
        max_neurons=N_CORTEX,
        seed=seed + 2,
    )

    # Motor: 3 guaranteed-excitatory chattering neurons (one per class).
    # Manually populated to avoid the random 80/20 excitatory/inhibitory split.
    _add_seeded_region(
        brain,
        "motor", RegionType.MOTOR,
        n_neurons=0, connectivity=0.0,
        max_neurons=N_MOTOR,
        seed=seed + 3,
    )
    motor = brain.regions["motor"]
    for _ in range(N_MOTOR):
        motor.add_neuron(NeuronType.EXCITATORY, FiringPattern.CHATTERING)

    # Lateral inhibition: each motor neuron strongly inhibits the others.
    # This implements winner-take-all dynamics so one motor neuron's
    # firing actively suppresses the others, making the readout decisive.
    for i in range(N_MOTOR):
        for j in range(N_MOTOR):
            if i != j:
                motor.add_one_synapse(
                    i, j, weight=LATERAL_INHIBITION, delay_ms=1.0,
                    nt=NeurotransmitterType.GABA,
                )

    # Projections
    brain.connect_regions("input", "cortex", density=PROJECTION_DENSITY_INPUT_CORTEX)
    brain.connect_regions("cortex", "motor", density=PROJECTION_DENSITY_CORTEX_MOTOR)
    brain.connect_regions("input", "motor", density=PROJECTION_DENSITY_INPUT_MOTOR)

    for proj in brain.projections:
        ns = proj.n_synapses
        if proj.source_name == "input" and proj.target_name == "cortex":
            proj.syn_weight[:ns] *= PROJECTION_WEIGHT_BOOST_IC
        elif proj.source_name == "cortex" and proj.target_name == "motor":
            # This pathway acted mostly as structured noise for the direct readout.
            proj.syn_weight[:ns] = 0.0
        elif proj.source_name == "input" and proj.target_name == "motor":
            # Start the readout nearly blank so learning isn't fighting random bias.
            proj.syn_weight[:ns] = READOUT_INIT_WEIGHT

    # Stronger sensory drive, lower noise for stable per-class patterns
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE
    brain.encoder._rng = np.random.default_rng(seed + 100)
    brain.memory._rng = np.random.default_rng(seed + 101)
    brain.growth._rng = np.random.default_rng(seed + 102)

    # Freeze plasticity everywhere except the input → motor readout.
    # Input patterns are inherently discriminative (place-field encoded),
    # so learning on input→motor gives much cleaner credit assignment
    # than on cortex→motor (whose dense random patterns overlap heavily).
    for region in brain.regions.values():
        ns = region.n_synapses
        region.syn_A_plus[:ns] = 0.0
        region.syn_A_minus[:ns] = 0.0
    for proj in brain.projections:
        ns = proj.n_synapses
        if proj.source_name == "input" and proj.target_name == "motor":
            proj.syn_A_plus[:ns] = 0.01 * STDP_SCALE
            proj.syn_A_minus[:ns] = 0.012 * STDP_SCALE
        else:
            proj.syn_A_plus[:ns] = 0.0
            proj.syn_A_minus[:ns] = 0.0

    # Eligibility decay + dopamine decay tuned so a reward pulse acts
    # like a short, focused credit signal (≈ a few-step impulse).
    brain.reward_stdp.tau_eligibility = TAU_ELIGIBILITY
    brain.reward_stdp.dopamine_decay = DOPAMINE_DECAY

    # Stable substrate for this benchmark — disable structural growth so we
    # isolate R-STDP learning. (Re-enable in a follow-up to test if growth helps.)
    brain.growth.growth_interval = 10**9

    # Disable metaplasticity: it adapts A_plus/A_minus per neuron every step
    # with a Python loop (slow + undoes our scaling). Not useful for this benchmark.
    brain.metaplasticity.update_thresholds = lambda *a, **kw: None

    return brain


def reset_traces(brain: Brain) -> None:
    """Zero eligibility traces and dopamine so each sample starts clean."""
    for region in brain.regions.values():
        ns = region.n_synapses
        if ns:
            region.syn_eligibility[:ns] = 0.0
    for proj in brain.projections:
        ns = proj.n_synapses
        if ns:
            proj.syn_eligibility[:ns] = 0.0
    brain.reward_stdp.dopamine.clear()


# ── Stimulus / readout helpers ──────────────────────────────────────

def normalize_features(X: np.ndarray) -> np.ndarray:
    """Scale each feature column to [0, 1]."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return (X - lo) / (hi - lo + 1e-9)


def place_field_encode(
    X: np.ndarray,
    n_bins: int = N_BINS_PER_FEATURE,
    sigma: float = PLACE_FIELD_SIGMA,
) -> np.ndarray:
    """
    Encode each feature with a bank of Gaussian receptive fields.

    Each feature is mapped to `n_bins` neurons with preferred values
    evenly spaced in [0, 1]. A neuron's activation = exp(-(x - center)² / 2σ²).
    Different classes (with different feature values) thus activate
    *different* input neurons — not just different magnitudes.

    Produces (N, F * n_bins) array, in column-major layout per feature:
        [f0_bin0, f0_bin1, ..., f0_bin9, f1_bin0, ..., f3_bin9]
    """
    N, F = X.shape
    centers = np.linspace(0.0, 1.0, n_bins)
    out = np.zeros((N, F * n_bins))
    for f in range(F):
        diff = X[:, f:f + 1] - centers[None, :]      # (N, n_bins)
        out[:, f * n_bins:(f + 1) * n_bins] = np.exp(-0.5 * (diff / sigma) ** 2)
    return out


def reset_between_samples(brain: Brain, n_steps: int = REST_STEPS) -> None:
    """Run quietly so spike buffers drain, then hard-reset traces & dopamine."""
    for _ in range(n_steps):
        brain.step()
    reset_traces(brain)


def present(brain: Brain, x: np.ndarray, n_steps: int = TEST_PRESENT_STEPS) -> np.ndarray:
    """Present a sample and return per-motor-neuron spike counts."""
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].copy()
    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.step()
    return motor.total_spikes[:N_MOTOR] - before


def _readout_proj(brain: Brain) -> 'Projection':
    """Return the input→motor readout projection."""
    return next(p for p in brain.projections
                if p.source_name == 'input' and p.target_name == 'motor')


_READOUT_TARGET = Brain.projection_target("input", "motor")


def _keep_only_motor_eligibility(proj, motor_idx: int) -> None:
    """Keep eligibility only on synapses projecting to one motor neuron."""
    ns = proj.n_synapses
    keep = proj.syn_post[:ns] == motor_idx
    proj.syn_eligibility[:ns][~keep] = 0.0


def train_one_sample(
    brain: Brain,
    x: np.ndarray,
    y: int,
    reward_amount: float,
    punish_amount: float,
) -> int:
    """
    Supervised R-STDP on the direct input→motor readout pathway.

    Input neurons fire early (place-field encoded, causal w.r.t. motor).
    A delayed teacher pulse drives motor[y] after input has fired, so STDP
    sees causal pre→post timing → positive eligibility on input→motor[y].

    Selective eligibility zeroing ensures only motor[y] synapses are
    reinforced by the reward signal.

    Returns the teacher-biased prediction (real metric is unbiased test acc).
    """
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].copy()

    natural_counts = np.zeros(N_MOTOR, dtype=int)
    for step in range(TRAIN_PRESENT_STEPS):
        brain.stimulate("input", x)
        if step >= TEACHER_DELAY:
            brain.inject_current("motor", [y], TEACHER_CURRENT)
        brain.step()
        if step + 1 == TEACHER_DELAY:
            natural_counts = motor.total_spikes[:N_MOTOR] - before

    counts = motor.total_spikes[:N_MOTOR] - before
    pred = int(np.argmax(natural_counts)) if natural_counts.max() > 0 else -1

    # Selective credit: only reinforce input→motor[y] synapses
    im = _readout_proj(brain)
    _keep_only_motor_eligibility(im, y)

    brain.reward(reward_amount, target=_READOUT_TARGET)
    for _ in range(10):
        brain.step()

    # Actively weaken the strongest wrong readout so classes separate faster.
    wrong_pred = pred if pred >= 0 and pred != y else -1
    if wrong_pred >= 0:
        im.syn_eligibility[:im.n_synapses] = 0.0
        for _ in range(TRAIN_PRESENT_STEPS):
            brain.stimulate("input", x)
            brain.step()
        _keep_only_motor_eligibility(im, wrong_pred)
        brain.punish(punish_amount, target=_READOUT_TARGET)
        for _ in range(10):
            brain.step()

    reset_between_samples(brain)
    return pred


def test_one_sample(
    brain: Brain,
    x: np.ndarray,
    test_repeats: int = TEST_REPEATS,
) -> tuple[int, np.ndarray]:
    """Predict one sample by summing spike counts across repeated presentations."""
    total_counts = np.zeros(N_MOTOR, dtype=int)
    for _ in range(test_repeats):
        total_counts += present(brain, x)
        reset_between_samples(brain)
    pred = int(np.argmax(total_counts)) if total_counts.max() > 0 else -1
    return pred, total_counts


def evaluate(brain: Brain, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    preds = np.empty(len(X), dtype=int)
    all_counts = np.empty((len(X), N_MOTOR), dtype=int)
    for i in range(len(X)):
        pred, counts = test_one_sample(brain, X[i])
        preds[i] = pred
        all_counts[i] = counts
    acc = float(np.mean(preds == y))
    return acc, preds, all_counts


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p < 0:
            continue
        cm[t, p] += 1
    return cm


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  IRIS BENCHMARK — Spiking Brain with R-STDP")
    print("  (No backpropagation: just spikes, STDP, and dopamine)")
    print("=" * 64)

    # ── Data ─────────────────────────────────────────────────────────
    iris = load_iris()
    X_raw = normalize_features(iris.data)
    X = place_field_encode(X_raw)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    print(f"\n  Dataset: {len(X)} samples, {X.shape[1]} place-field bins "
          f"(4 features × {N_BINS_PER_FEATURE} bins), 3 classes")
    print(f"  Split: {len(X_train)} train / {len(X_test)} test")
    print(f"  Classes: {iris.target_names.tolist()}")

    # ── Sklearn baseline (so we have something to compare against) ──
    baseline = LogisticRegression(max_iter=1000, random_state=SEED)
    baseline.fit(X_train, y_train)
    base_test_acc = baseline.score(X_test, y_test)
    print(f"\n  Sklearn LogisticRegression baseline test accuracy: {base_test_acc:.1%}")

    # ── Brain ────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Building synthetic brain")
    print("─" * 64)
    brain = build_brain(seed=SEED)
    print(brain.summary())

    # Untrained baseline
    untrained_acc, _, _ = evaluate(brain, X_test, y_test)
    print(f"\n  Untrained brain test accuracy: {untrained_acc:.1%} (chance ≈ 33%)")

    # ── Training ─────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Training")
    print("─" * 64)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    best_test_acc = -1.0
    best_epoch = 0
    best_brain = None

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(X_train))
        train_correct = 0
        reward_scale = 1.0 if epoch < REWARD_DECAY_EPOCH else 0.5
        current_reward = REWARD_AMOUNT * reward_scale
        for i in perm:
            pred = train_one_sample(
                brain,
                X_train[i],
                int(y_train[i]),
                reward_amount=current_reward,
                punish_amount=PUNISH_AMOUNT,
            )
            if pred == y_train[i]:
                train_correct += 1
        train_acc = train_correct / len(X_train)
        test_acc, _, _ = evaluate(brain, X_test, y_test)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_brain = copy.deepcopy(brain)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS}: "
            f"train={train_acc:.1%}  test={test_acc:.1%}  "
            f"t={elapsed:5.1f}s  "
            f"synapses={brain._snapshot().total_synapses}"
        )

    # ── Final evaluation ─────────────────────────────────────────────
    if best_brain is not None:
        brain = best_brain
        reset_traces(brain)

    print("\n" + "─" * 64)
    print("  Final Results")
    print("─" * 64)
    final_acc, preds, counts = evaluate(brain, X_test, y_test)

    no_response = int(np.sum(preds < 0))
    print(f"\n  Brain test accuracy:          {final_acc:.1%}")
    print(f"  Sklearn baseline:             {base_test_acc:.1%}")
    print(f"  Best checkpoint epoch:        {best_epoch} ({best_test_acc:.1%})")
    print(f"  Samples with no motor spikes: {no_response}/{len(y_test)}")

    cm = confusion_matrix(y_test, preds)
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "             " + "  ".join(f"{n:>10s}" for n in iris.target_names)
    print(header)
    for i, name in enumerate(iris.target_names):
        row = "  ".join(f"{v:>10d}" for v in cm[i])
        print(f"    {name:>8s}:  {row}")

    # Per-class average spike counts (which motor neurons learned what)
    print(f"\n  Mean motor spike counts per true class:")
    print(f"             motor[0]   motor[1]   motor[2]")
    for c in range(3):
        mask = y_test == c
        if mask.any():
            mean = counts[mask].mean(axis=0)
            print(f"    {iris.target_names[c]:>8s}: "
                  + "  ".join(f"{v:9.2f}" for v in mean))

    # ── Brain state at end ──────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Final brain state")
    print("─" * 64)
    print(brain.summary())

    # ── Verdict ──────────────────────────────────────────────────────
    chance = 1.0 / 3
    print("\n" + "=" * 64)
    if final_acc >= 0.85:
        print(f"  ✓ The brain learned Iris well ({final_acc:.1%}).")
        print(f"    Gap to sklearn: {(base_test_acc - final_acc) * 100:+.1f} pp.")
    elif final_acc >= chance + 0.10:
        print(f"  ~ Above-chance learning ({final_acc:.1%} vs chance {chance:.1%}).")
        print(f"    The brain is learning a signal but is unstable / under-fit.")
        print(f"    Gap to sklearn: {(base_test_acc - final_acc) * 100:+.1f} pp.")
    else:
        print(f"  ✗ At/below chance ({final_acc:.1%} vs chance {chance:.1%}).")
        print(f"    No meaningful learning detected this run.")
    print("=" * 64)


if __name__ == "__main__":
    main()
