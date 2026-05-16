"""
Catastrophic forgetting benchmark — SNN vs MLP.

Tests whether sequential task learning destroys previous knowledge.

Protocol:
  1. Train on MNIST digits 0-4 (Task A)
  2. Evaluate on 0-4 → accuracy_A_before
  3. Train on MNIST digits 5-9 (Task B) — same network, no reset
  4. Evaluate on 5-9 → accuracy_B
  5. Re-evaluate on 0-4 → accuracy_A_after
  6. Forgetting = accuracy_A_before - accuracy_A_after
  7. Compare SNN forgetting vs MLP forgetting

A positive result: SNN retains significant accuracy on Task A after
learning Task B; MLP baseline drops to near-chance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.mnist_benchmark import (
    SEED,
    build_brain,
    load_reduced_mnist,
    present_sample,
    reset_brain_state,
    normalize_feedforward_weights,
    excitatory_cortex_indices,
    build_response_templates,
    predict_sample,
)
from src.device import DEVICE

TASK_A_CLASSES = (0, 1, 2, 3, 4)
TASK_B_CLASSES = (5, 6, 7, 8, 9)
ALL_CLASSES = TASK_A_CLASSES + TASK_B_CLASSES
TRAIN_PER_CLASS = 300
TEST_PER_CLASS = 50
PRESENT_STEPS = 200
REST_STEPS = 50


# --- MLP baseline -----------------------------------------------------------

def mlp_train_eval(X_train, y_train, X_test, y_test, classes,
                   W1=None, b1=None, W2=None, b2=None,
                   hidden=128, epochs=50, lr=0.01):
    """Train MLP and return (accuracy, weights). Reuses weights if provided."""
    n_classes = len(ALL_CLASSES)
    n_features = X_train.shape[1]
    class_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}

    if W1 is None:
        W1 = torch.randn(n_features, hidden, device=DEVICE) * 0.01
        b1 = torch.zeros(hidden, device=DEVICE)
        W2 = torch.randn(hidden, n_classes, device=DEVICE) * 0.01
        b2 = torch.zeros(n_classes, device=DEVICE)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor([class_to_idx[int(c)] for c in y_train], dtype=torch.int64, device=DEVICE)

    for _ in range(epochs):
        h = torch.relu(X_t @ W1 + b1)
        logits = h @ W2 + b2
        log_probs = logits - logits.logsumexp(dim=1, keepdim=True)

        probs = log_probs.exp()
        d_logits = probs.clone()
        d_logits[torch.arange(len(y_t), device=DEVICE), y_t] -= 1.0
        d_logits /= len(y_t)

        dW2 = h.T @ d_logits
        db2 = d_logits.sum(0)
        dh = d_logits @ W2.T
        dh[h <= 0] = 0.0
        dW1 = X_t.T @ dh
        db1 = dh.sum(0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Evaluate on the specified test set
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_te_idx = torch.tensor([class_to_idx[int(c)] for c in y_test], dtype=torch.int64, device=DEVICE)
    with torch.no_grad():
        h_te = torch.relu(X_te @ W1 + b1)
        preds = (h_te @ W2 + b2).argmax(dim=1)
        correct = (preds == y_te_idx).sum().item()

    return correct / len(y_test), W1, b1, W2, b2


# --- SNN sequential training ------------------------------------------------

def snn_sequential(X_a, y_a, X_b, y_b, X_test_a, y_test_a, X_test_b, y_test_b):
    """Train SNN on task A then B; evaluate after each phase."""
    brain = build_brain(seed=SEED)

    exc_idx = excitatory_cortex_indices(brain)
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    weight_sums = torch.zeros(brain.regions["cortex"].n_neurons, dtype=torch.float32, device=DEVICE)
    weight_sums.index_add_(0, post[alive], proj.syn_weight[:ns][alive].to(torch.float32))
    norm_target = float(weight_sums[exc_idx].mean().item())

    def train_phase(X, y):
        order = np.random.default_rng(SEED).permutation(len(X))
        for idx in order:
            present_sample(brain, X[idx], PRESENT_STEPS, learn=True)
            normalize_feedforward_weights(brain, norm_target)
            reset_brain_state(brain, REST_STEPS)

    def eval_phase(X_train_for_templates, y_train_for_templates, X_test, y_test, classes):
        spike_t, voltage_t = build_response_templates(brain, X_train_for_templates, y_train_for_templates, classes)
        correct = 0
        for x, label in zip(X_test, y_test):
            pred, _ = predict_sample(brain, x, spike_t, voltage_t, classes)
            if pred == int(label):
                correct += 1
        return correct / len(y_test)

    # Phase 1: Train on Task A
    print("  Training on Task A (digits 0-4)...")
    t0 = time.time()
    train_phase(X_a, y_a)
    print(f"    done in {time.time() - t0:.0f}s")

    # Evaluate on Task A
    acc_a_before = eval_phase(X_a, y_a, X_test_a, y_test_a, TASK_A_CLASSES)
    print(f"  Task A accuracy after training A: {acc_a_before:.1%}")

    # Phase 2: Train on Task B (same network)
    print("  Training on Task B (digits 5-9)...")
    t1 = time.time()
    train_phase(X_b, y_b)
    print(f"    done in {time.time() - t1:.0f}s")

    # Evaluate on Task B
    acc_b = eval_phase(X_b, y_b, X_test_b, y_test_b, TASK_B_CLASSES)
    print(f"  Task B accuracy after training B: {acc_b:.1%}")

    # Re-evaluate on Task A (forgetting test)
    acc_a_after = eval_phase(X_a, y_a, X_test_a, y_test_a, TASK_A_CLASSES)
    print(f"  Task A accuracy after training B: {acc_a_after:.1%}")

    return acc_a_before, acc_b, acc_a_after


# --- Main -------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  CATASTROPHIC FORGETTING BENCHMARK — SNN vs MLP")
    print("  (Train on 0-4, then 5-9, re-test on 0-4)")
    print("=" * 68)

    X_a, y_a, X_test_a, y_test_a = load_reduced_mnist(
        classes=TASK_A_CLASSES, train_per_class=TRAIN_PER_CLASS, test_per_class=TEST_PER_CLASS,
    )
    X_b, y_b, X_test_b, y_test_b = load_reduced_mnist(
        classes=TASK_B_CLASSES, train_per_class=TRAIN_PER_CLASS, test_per_class=TEST_PER_CLASS,
    )

    # --- SNN ---
    print("\n--- SNN ---")
    snn_a_before, snn_b, snn_a_after = snn_sequential(
        X_a, y_a, X_b, y_b, X_test_a, y_test_a, X_test_b, y_test_b,
    )
    snn_forgetting = snn_a_before - snn_a_after

    # --- MLP ---
    print("\n--- MLP ---")
    print("  Training on Task A (digits 0-4)...")
    mlp_a_before, W1, b1, W2, b2 = mlp_train_eval(X_a, y_a, X_test_a, y_test_a, TASK_A_CLASSES)
    print(f"  Task A accuracy after training A: {mlp_a_before:.1%}")

    print("  Training on Task B (digits 5-9)...")
    mlp_b, W1, b1, W2, b2 = mlp_train_eval(X_b, y_b, X_test_b, y_test_b, TASK_B_CLASSES,
                                             W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"  Task B accuracy after training B: {mlp_b:.1%}")

    mlp_a_after, _, _, _, _ = mlp_train_eval(X_a, y_a, X_test_a, y_test_a, TASK_A_CLASSES,
                                              W1=W1, b1=b1, W2=W2, b2=b2, epochs=0)
    print(f"  Task A accuracy after training B: {mlp_a_after:.1%}")
    mlp_forgetting = mlp_a_before - mlp_a_after

    # --- Summary ---
    print("\n" + "=" * 68)
    print("  Summary")
    print("=" * 68)
    print(f"  {'':>20s}  {'SNN':>8s}  {'MLP':>8s}")
    print(f"  {'Task A (before B)':>20s}  {snn_a_before:>7.1%}  {mlp_a_before:>7.1%}")
    print(f"  {'Task B':>20s}  {snn_b:>7.1%}  {mlp_b:>7.1%}")
    print(f"  {'Task A (after B)':>20s}  {snn_a_after:>7.1%}  {mlp_a_after:>7.1%}")
    print(f"  {'Forgetting':>20s}  {snn_forgetting:>+7.1%}  {mlp_forgetting:>+7.1%}")
    winner = "SNN" if snn_forgetting < mlp_forgetting else "MLP"
    print(f"\n  Less forgetting: {winner}")
    print("=" * 68)


if __name__ == "__main__":
    main()
