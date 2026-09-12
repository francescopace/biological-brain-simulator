"""
Iris classification benchmark for the synthetic brain.

The benchmark trains a readout with spike-timing dependent plasticity modulated
by dopamine (R-STDP), selects an epoch on validation data, and compares its
spike-only predictions with a labelled logistic-regression baseline.

Architecture:
    input (80 excitatory sensory neurons, 20 place-field bins per feature)
        ↓ density 0.5           ↓ density 1.0 (readout, R-STDP)
    cortex (80 association)     motor (3 chattering, one per class)
        ↓ density 0.8 (frozen)   ↑ lateral inhibition (GABA)
              → motor ──────────

The input→motor shortcut is the readout learned by R-STDP.
Cortex→motor weights are zero and frozen; this pathway provides no drive.

Training (supervised R-STDP with delayed teacher):
    1. Present sample (place-field encoded, 30 ms).
    2. After TEACHER_DELAY (13 ms), inject teacher current into
       motor[y]. Input fires first (causal) → STDP builds positive
       eligibility on input→motor[y].
    3. Restrict eligibility and new STDP events to motor[y] for the
       entire reward pulse (selective credit).
    4. Reward → dopamine × eligibility consolidates input→motor[y].
    5. Reset between samples.

Evaluation starts each sample from the same transient state on a frozen copy,
without teacher, noise or oscillations. Each 80ms response is computed once and
reused in the six-repeat vote. Voltage fallback is reported separately from
spike-only accuracy. --validation-only leaves test-set scoring disabled;
--output NEW_DIR records results and immutable per-epoch checkpoints, without
providing CLI resume support.
"""

import argparse
import copy
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import confusion_matrix, quiet_steps
from examples._utils import can_reuse_independent_inference, inference_brain, reset_independent_state
from examples.training_checkpoint import array_digest, atomic_json, save_training_checkpoint

try:
    import sklearn
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError:
    print("This benchmark requires scikit-learn:")
    print("  pip install scikit-learn")
    sys.exit(1)

from src.brain import Brain
from src.device import DEVICE
from src.neuron import FiringPattern, NeuronType
from src.region import RegionType


# ── Hyperparameters ──────────────────────────────────────────────────

N_BINS_PER_FEATURE = 20   # place-field bins per Iris feature
N_INPUT = 80              # 4 features × 20 bins
PLACE_FIELD_SIGMA = 0.08  # narrower fields improve separability of nearby classes
N_CORTEX = 80
N_MOTOR = 3               # one per class
TEST_PRESENT_STEPS = 80   # ms of stimulus per sample (test mode: longer for better readout)
TEST_REPEATS = 6          # preserve the six-vote sum; identical frozen responses are reused
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

def build_brain(seed: int = SEED) -> Brain:
    brain = Brain(dt=1.0, seed=seed)

    # Input: pure feedforward, no internal connectivity
    sensory = brain.add_region(
        "input",
        RegionType.SENSORY,
        n_neurons=0, connectivity=0.0,
        max_neurons=N_INPUT,
    )
    # connect_regions uses excitatory sources only: every encoded bin needs
    # an output pathway, not the generic region's random 80/20 type split.
    for _ in range(N_INPUT):
        sensory.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)

    # Cortex: sparsely recurrent association area
    brain.add_region(
        "cortex",
        RegionType.ASSOCIATION,
        n_neurons=N_CORTEX, connectivity=0.05,
        max_neurons=N_CORTEX,
    )

    # Motor: 3 guaranteed-excitatory chattering neurons (one per class).
    # Manually populated to avoid the random 80/20 excitatory/inhibitory split.
    brain.add_region(
        "motor",
        RegionType.MOTOR,
        n_neurons=0, connectivity=0.0,
        max_neurons=N_MOTOR,
    )
    motor = brain.regions["motor"]
    for _ in range(N_MOTOR):
        motor.add_neuron(NeuronType.EXCITATORY, FiringPattern.CHATTERING)

    # Lateral inhibition: each motor neuron strongly inhibits the others.
    # This implements winner-take-all dynamics so one motor neuron's
    # firing actively suppresses the others, making the readout decisive.
    motor.add_lateral_inhibition(weight=LATERAL_INHIBITION, delay_ms=1.0)

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
        proj.syn_weight[:ns] = torch.clamp(proj.syn_weight[:ns],
                                           proj.syn_min_weight[:ns], proj.syn_max_weight[:ns])

    # Stronger sensory drive, lower noise for stable per-class patterns
    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE

    # Freeze plasticity everywhere except the input → motor readout.
    # Input patterns are inherently discriminative (place-field encoded),
    # so learning on input→motor gives much cleaner credit assignment
    # than on cortex→motor (whose dense random patterns overlap heavily).
    brain.freeze_plasticity()
    brain.enable_projection_plasticity(
        "input",
        "motor",
        A_plus=0.01 * STDP_SCALE,
        A_minus=0.012 * STDP_SCALE,
    )

    # Eligibility decay + dopamine decay tuned so a reward pulse acts
    # like a short, focused credit signal (≈ a few-step impulse).
    brain.reward_stdp.tau_eligibility = TAU_ELIGIBILITY
    brain.reward_stdp.dopamine_decay = DOPAMINE_DECAY

    # Stable substrate for this benchmark — disable structural growth so we
    # isolate R-STDP learning. (Re-enable in a follow-up to test if growth helps.)
    brain.freeze_structural_plasticity()
    brain.freeze_homeostatic_scaling()

    return brain


# ── Stimulus / readout helpers ──────────────────────────────────────

def normalize_features(
    X: np.ndarray,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> np.ndarray:
    """Scale features using bounds fitted on the training set."""
    if lo is None:
        lo = X.min(axis=0)
    if hi is None:
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


def reset_between_samples(brain: Brain, n_steps: int | None = None) -> None:
    """Run quietly so spike buffers drain, then hard-reset traces & dopamine."""
    quiet_steps(brain, REST_STEPS if n_steps is None else n_steps)
    brain.reset_traces()


def present(
    brain: Brain,
    x: np.ndarray,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Present a sample and return spike counts plus mean motor voltage."""
    n_steps = TEST_PRESENT_STEPS if n_steps is None else n_steps
    if n_steps < 1:
        raise ValueError("Presentation steps must be positive")
    stimulus = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].clone()
    voltage_sum = torch.zeros(N_MOTOR, dtype=torch.float32, device=DEVICE)
    for _ in range(n_steps):
        brain.stimulate("input", stimulus)
        brain.step()
        voltage_sum += motor.v[:N_MOTOR]
    counts = motor.total_spikes[:N_MOTOR] - before
    return counts.cpu().numpy(), (voltage_sum / n_steps).cpu().numpy()


def _readout_proj(brain: Brain) -> 'Projection':
    """Return the input→motor readout projection."""
    return brain.get_projection("input", "motor")


_READOUT_TARGET = Brain.projection_target("input", "motor")


def _reinforce_motor(
    brain: Brain,
    motor_idx: int,
    amount: float,
    positive: bool,
) -> None:
    """Restrict the full dopamine pulse to the selected class readout."""
    with brain.reward_stdp.restrict_to_posts(_READOUT_TARGET, [motor_idx]):
        if positive:
            brain.reward(amount, target=_READOUT_TARGET)
        else:
            brain.punish(amount, target=_READOUT_TARGET)
        quiet_steps(brain, 10)


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

    A postsynaptic mask remains active throughout each reward/punishment
    pulse, so only the selected motor's synapses receive credit.

    Returns the pre-teacher prediction. This is a response diagnostic, not
    a training accuracy estimate.
    """
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].clone()
    stimulus = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

    natural_counts = torch.zeros(N_MOTOR, dtype=torch.int32, device=DEVICE)
    for step in range(TRAIN_PRESENT_STEPS):
        brain.stimulate("input", stimulus)
        if step >= TEACHER_DELAY:
            brain.inject_current("motor", [y], TEACHER_CURRENT)
        brain.step()
        if step + 1 == TEACHER_DELAY:
            natural_counts = motor.total_spikes[:N_MOTOR] - before

    pred = int(torch.argmax(natural_counts).item()) if natural_counts.max().item() > 0 else -1

    # Selective credit: only reinforce input→motor[y] synapses
    im = _readout_proj(brain)
    _reinforce_motor(brain, y, reward_amount, positive=True)

    # Actively weaken the strongest wrong readout so classes separate faster.
    wrong_pred = pred if pred >= 0 and pred != y else -1
    if wrong_pred >= 0:
        im.syn_eligibility[:im.n_synapses] = 0.0
        for _ in range(TRAIN_PRESENT_STEPS):
            brain.stimulate("input", stimulus)
            brain.step()
        _reinforce_motor(brain, wrong_pred, punish_amount, positive=False)

    reset_between_samples(brain)
    return pred


def test_one_sample(
    brain: Brain,
    x: np.ndarray,
    test_repeats: int | None = None,
    *, independent: bool = False, reuse_identical: bool = False,
) -> tuple[int, int, np.ndarray]:
    """Return spike-only and voltage-fallback predictions."""
    test_repeats = TEST_REPEATS if test_repeats is None else test_repeats
    if test_repeats < 1:
        raise ValueError("Test repeats must be positive")
    reuse = independent and reuse_identical and can_reuse_independent_inference(brain)
    total_counts = np.zeros(N_MOTOR, dtype=int)
    total_voltage = np.zeros(N_MOTOR, dtype=np.float64)
    for repeat in range(test_repeats):
        if not reuse or repeat == 0:
            if independent:
                reset_independent_state(brain)
            counts, voltage = present(brain, x)
        total_counts += counts
        total_voltage += voltage
        if not independent:
            reset_between_samples(brain)
    spike_pred = int(np.argmax(total_counts)) if total_counts.max() > 0 else -1
    fallback_pred = spike_pred if spike_pred >= 0 else int(np.argmax(total_voltage))
    return spike_pred, fallback_pred, total_counts


@dataclass
class IrisEvaluation:
    spike_accuracy: float
    fallback_accuracy: float
    spike_predictions: np.ndarray
    fallback_predictions: np.ndarray
    counts: np.ndarray


def evaluate(brain: Brain, X: np.ndarray, y: np.ndarray, *,
             independent: bool = True, fast: bool = True) -> IrisEvaluation:
    # Evaluation advances membrane, homeostatic and RNG state. Keep it
    # side-effect free so validation cannot influence later training.
    if len(X) != len(y) or len(y) == 0:
        raise ValueError("Evaluation needs matching, nonempty samples and labels")
    if independent:
        eval_brain = inference_brain(brain)
    else:
        # Retain the former sequential protocol for explicit diagnostics only.
        eval_brain = copy.deepcopy(brain)
        eval_brain.reset_traces()
        eval_brain.freeze_adaptive_thresholds()
        eval_brain.encoder.noise_level = 0.0
    spike_preds = np.empty(len(X), dtype=int)
    fallback_preds = np.empty(len(X), dtype=int)
    all_counts = np.empty((len(X), N_MOTOR), dtype=int)
    for i in range(len(X)):
        spike_pred, fallback_pred, counts = test_one_sample(
            eval_brain, X[i], independent=independent, reuse_identical=fast)
        spike_preds[i] = spike_pred
        fallback_preds[i] = fallback_pred
        all_counts[i] = counts
    return IrisEvaluation(
        spike_accuracy=float(np.mean(spike_preds == y)),
        fallback_accuracy=float(np.mean(fallback_preds == y)),
        spike_predictions=spike_preds,
        fallback_predictions=fallback_preds,
        counts=all_counts,
    )
# ── Main ─────────────────────────────────────────────────────────────

def evaluation_payload(result: IrisEvaluation) -> dict:
    return {"spike_accuracy": result.spike_accuracy, "fallback_accuracy": result.fallback_accuracy,
            "spike_predictions": result.spike_predictions.tolist(),
            "fallback_predictions": result.fallback_predictions.tolist(), "counts": result.counts.tolist(),
            "silent_samples": int(np.sum(result.spike_predictions < 0))}


def main(*, output: Path | None = None, validation_only: bool = False):
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/_utils.py", root / "examples/training_checkpoint.py",
               *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config = {name: value for name, value in globals().items()
              if name.isupper() and not name.startswith("_") and isinstance(value, (int, float, str, bool))}
    payload = {"complete": False, "pid": os.getpid(), "started_at": time.time(),
               "source_sha256": hashes, "config": config, "validation_only": validation_only,
               "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
               "inference": "independent, frozen, deterministic; identical repeats reused",
               "epochs": []}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "summary.json", payload)

    def publish():
        if output is not None:
            atomic_json(output / "summary.json", payload)

    def checkpoint(label, model, progress):
        if output is not None:
            save_training_checkpoint(model, output / "checkpoints" / label,
                                      {"protocol": {"config": config, "source_sha256": hashes,
                                                    "dataset": payload["dataset"]}, **progress})

    print("=" * 64)
    print("  IRIS BENCHMARK — Spiking Brain with R-STDP")
    print("  (No backpropagation: just spikes, STDP, and dopamine)")
    print("=" * 64)

    # ── Data ─────────────────────────────────────────────────────────
    iris = load_iris()
    train_val_ids, test_ids = train_test_split(
        np.arange(len(iris.data)),
        test_size=0.2, random_state=SEED, stratify=iris.target,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.2, random_state=SEED + 1, stratify=iris.target[train_val_ids],
    )
    X_train_raw, X_val_raw, X_test_raw = (iris.data[ids] for ids in (train_ids, val_ids, test_ids))
    y_train, y_val, y_test = (iris.target[ids] for ids in (train_ids, val_ids, test_ids))
    lo = X_train_raw.min(axis=0)
    hi = X_train_raw.max(axis=0)
    X_train = place_field_encode(normalize_features(X_train_raw, lo, hi))
    X_val = place_field_encode(normalize_features(X_val_raw, lo, hi))
    X_test = place_field_encode(normalize_features(X_test_raw, lo, hi))
    payload["dataset"] = {"train_ids": train_ids.tolist(), "validation_ids": val_ids.tolist(),
                          "test_ids": test_ids.tolist(), "normalization_lo": lo.tolist(),
                          "normalization_hi": hi.tolist(), "train_sha256": array_digest(X_train, y_train),
                          "validation_sha256": array_digest(X_val, y_val),
                          "test_sha256": array_digest(X_test, y_test)}
    publish()
    print(f"\n  Dataset: {len(iris.data)} samples, {X_train.shape[1]} place-field bins "
          f"(4 features × {N_BINS_PER_FEATURE} bins), 3 classes")
    print(
        f"  Split: {len(X_train)} train / {len(X_val)} validation / "
        f"{len(X_test)} test"
    )
    print(f"  Classes: {iris.target_names.tolist()}")

    # ── Sklearn baseline (so we have something to compare against) ──
    baseline = LogisticRegression(max_iter=1000, random_state=SEED)
    baseline.fit(X_train, y_train)
    base_val_acc = baseline.score(X_val, y_val)
    print(f"\n  Sklearn LogisticRegression validation accuracy: {base_val_acc:.1%}")

    # ── Brain ────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Building synthetic brain")
    print("─" * 64)
    brain = build_brain(seed=SEED)
    print(brain.summary())
    checkpoint("initial", brain, {"completed_epochs": 0})

    # Untrained baseline
    untrained = evaluate(brain, X_val, y_val)
    payload["initial_validation"] = evaluation_payload(untrained)
    payload["baseline_validation_accuracy"] = float(base_val_acc)
    publish()
    print(
        f"\n  Untrained brain validation accuracy: "
        f"{untrained.spike_accuracy:.1%} (chance ≈ 33%)"
    )

    # ── Training ─────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Training")
    print("─" * 64)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    best_val_acc = -1.0
    best_epoch = 0
    best_brain = None

    for epoch in range(EPOCHS):
        perm = rng.permutation(len(X_train))
        pre_teacher_correct = 0
        pre_teacher_responses = 0
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
                pre_teacher_correct += 1
            if pred >= 0:
                pre_teacher_responses += 1
        pre_teacher_acc = pre_teacher_correct / len(X_train)
        pre_teacher_response_rate = pre_teacher_responses / len(X_train)
        validation = evaluate(brain, X_val, y_val)
        val_acc = validation.spike_accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_brain = copy.deepcopy(brain)
        elapsed = time.time() - t0
        payload["epochs"].append({"epoch": epoch + 1, "pre_teacher_accuracy": pre_teacher_acc,
                                  "pre_teacher_response_rate": pre_teacher_response_rate,
                                  "validation": evaluation_payload(validation), "elapsed_s": elapsed})
        payload.update(best_epoch=best_epoch, best_validation_accuracy=best_val_acc)
        checkpoint(f"epoch_{epoch + 1:03d}", brain,
                   {"completed_epochs": epoch + 1, "shuffle_rng_state": rng.bit_generator.state,
                    "best_epoch": best_epoch, "epochs": payload["epochs"]})
        publish()
        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS}: "
            f"pre_teacher_acc={pre_teacher_acc:.1%}  "
            f"pre_teacher_response={pre_teacher_response_rate:.1%}  "
            f"val_spike={val_acc:.1%}  "
            f"val_fallback={validation.fallback_accuracy:.1%}  "
            f"t={elapsed:5.1f}s  "
            f"synapses={brain._snapshot().total_synapses}"
        )

    # ── Final evaluation ─────────────────────────────────────────────
    if best_brain is not None:
        brain = best_brain
        brain.reset_traces()

    print("\n" + "─" * 64)
    print("  Final Results")
    print("─" * 64)
    final_X, final_y = (X_val, y_val) if validation_only else (X_test, y_test)
    final = evaluate(brain, final_X, final_y)
    final_acc = final.spike_accuracy
    preds = final.spike_predictions
    counts = final.counts
    base_test_acc = baseline.score(final_X, final_y)
    payload["final_partition"] = "validation" if validation_only else "test"
    payload["final"] = evaluation_payload(final)
    payload["baseline_final_accuracy"] = float(base_test_acc)
    checkpoint("selected", brain, {"completed_epochs": best_epoch, "selected_by": "validation"})
    if validation_only:
        print("\n  Validation-only run: no test-set scoring. These are selection-set results.")

    no_response = int(np.sum(preds < 0))
    print(f"\n  Brain spike-only accuracy:    {final_acc:.1%}")
    print(f"  With voltage fallback:        {final.fallback_accuracy:.1%}")
    print(f"  Sklearn baseline:             {base_test_acc:.1%}")
    print(f"  Best checkpoint epoch:        {best_epoch} ({best_val_acc:.1%} validation)")
    print(f"  Samples with no motor spikes: {no_response}/{len(final_y)}")

    cm = confusion_matrix(final_y, preds, labels=range(3))
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
        mask = final_y == c
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
    if hashes != {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError("Benchmark sources changed during the run")
    payload.update(complete=True, finished_at=time.time(), source_unchanged=True)
    publish()
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New result/checkpoint directory; no CLI resume support")
    parser.add_argument("--validation-only", action="store_true", help="Do not score the test split")
    args = parser.parse_args()
    main(output=args.output, validation_only=args.validation_only)
