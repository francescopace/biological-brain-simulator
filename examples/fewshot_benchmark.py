"""
Few-shot learning benchmark — SNN vs MLP baseline.

Measures how quickly the SNN learns from limited data compared to a
simple MLP trained with backpropagation on the same data budget.

Protocol:
  - Train on K samples/class for K in [1, 5, 10, 50, 100, 300]
  - Evaluate on 50/class held-out test set
  - Compare SNN accuracy vs a 1-hidden-layer MLP baseline
  - Report accuracy vs data budget curve

A positive result: SNN reaches usable accuracy with fewer samples
than the MLP, suggesting local plasticity extracts more per example.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.mnist_benchmark import (
    CLASSES,
    SEED,
    build_brain,
    load_reduced_mnist,
    present_sample,
    reset_brain_state,
    normalize_feedforward_weights,
    excitatory_cortex_indices,
    build_response_templates,
    predict_sample,
    _balanced_subset,
)
from src.device import DEVICE

BUDGETS = [1, 5, 10, 50, 100, 300]
PRESENT_STEPS = 200
REST_STEPS = 50
TEST_PER_CLASS = 50


# --- MLP baseline -----------------------------------------------------------

def mlp_baseline(X_train, y_train, X_test, y_test, classes, hidden=128, epochs=50, lr=0.01):
    """Train a simple 1-hidden-layer MLP and return test accuracy."""
    n_classes = len(classes)
    n_features = X_train.shape[1]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    W1 = torch.randn(n_features, hidden, device=DEVICE) * 0.01
    b1 = torch.zeros(hidden, device=DEVICE)
    W2 = torch.randn(hidden, n_classes, device=DEVICE) * 0.01
    b2 = torch.zeros(n_classes, device=DEVICE)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor([class_to_idx[int(c)] for c in y_train], dtype=torch.int64, device=DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_te = torch.tensor([class_to_idx[int(c)] for c in y_test], dtype=torch.int64, device=DEVICE)

    for _ in range(epochs):
        h = torch.relu(X_t @ W1 + b1)
        logits = h @ W2 + b2
        log_probs = logits - logits.logsumexp(dim=1, keepdim=True)
        loss = -log_probs[torch.arange(len(y_t), device=DEVICE), y_t].mean()

        # Manual backward (simple 2-layer network)
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

    with torch.no_grad():
        h_te = torch.relu(X_te @ W1 + b1)
        preds = (h_te @ W2 + b2).argmax(dim=1)
        correct = (preds == y_te).sum().item()
    return correct / len(y_te)


# --- SNN evaluation ----------------------------------------------------------

def snn_fewshot(X_train, y_train, X_test, y_test, classes):
    """Train the SNN on the given (small) dataset and return test accuracy."""
    brain = build_brain(seed=SEED)

    exc_idx = excitatory_cortex_indices(brain)
    proj = brain.get_projection("input", "cortex")
    ns = proj.n_synapses
    post = proj.syn_post[:ns].to(torch.int64)
    alive = proj.syn_alive[:ns]
    weight_sums = torch.zeros(brain.regions["cortex"].n_neurons, dtype=torch.float32, device=DEVICE)
    weight_sums.index_add_(0, post[alive], proj.syn_weight[:ns][alive].to(torch.float32))
    norm_target = float(weight_sums[exc_idx].mean().item())

    order = np.random.default_rng(SEED).permutation(len(X_train))
    for idx in order:
        present_sample(brain, X_train[idx], PRESENT_STEPS, learn=True)
        normalize_feedforward_weights(brain, norm_target)
        reset_brain_state(brain, REST_STEPS)

    spike_templates, voltage_templates = build_response_templates(brain, X_train, y_train)

    correct = 0
    for x, label in zip(X_test, y_test):
        pred, _ = predict_sample(brain, x, spike_templates, voltage_templates, classes)
        if pred == int(label):
            correct += 1
    return correct / len(y_test)


# --- Main -------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  FEW-SHOT LEARNING BENCHMARK — SNN vs MLP")
    print("  (MNIST, varying samples/class)")
    print("=" * 68)

    max_train = max(BUDGETS)
    X_all, y_all, X_test, y_test = load_reduced_mnist(
        classes=CLASSES,
        train_per_class=max_train,
        test_per_class=TEST_PER_CLASS,
    )

    print(f"\n  Test set: {len(X_test)} samples")
    print(f"  Budgets: {BUDGETS} samples/class\n")

    results = []
    for k in BUDGETS:
        print(f"--- K={k} samples/class ({k * len(CLASSES)} total) ---")
        t0 = time.time()

        subset = _balanced_subset(y_all, CLASSES, k, seed=SEED)
        X_sub, y_sub = X_all[subset], y_all[subset]

        snn_acc = snn_fewshot(X_sub, y_sub, X_test, y_test, CLASSES)
        mlp_acc = mlp_baseline(X_sub, y_sub, X_test, y_test, CLASSES)

        elapsed = time.time() - t0
        results.append((k, snn_acc, mlp_acc))
        print(f"  SNN: {snn_acc:.1%}  |  MLP: {mlp_acc:.1%}  |  {elapsed:.0f}s\n")

    print("=" * 68)
    print("  Summary")
    print("=" * 68)
    print(f"  {'K':>5s}  {'SNN':>8s}  {'MLP':>8s}  {'SNN-MLP':>8s}")
    for k, snn_acc, mlp_acc in results:
        diff = snn_acc - mlp_acc
        marker = " <-- SNN wins" if diff > 0.01 else ""
        print(f"  {k:>5d}  {snn_acc:>7.1%}  {mlp_acc:>7.1%}  {diff:>+7.1%}{marker}")
    print("=" * 68)


if __name__ == "__main__":
    main()
