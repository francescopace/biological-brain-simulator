"""
Graceful degradation benchmark — SNN vs MLP.

Tests whether the SNN tolerates random neuron damage better than a
conventional model with equivalent weight ablation.

Protocol:
  1. Train SNN and MLP on MNIST (300/class, 10 classes)
  2. Evaluate both → baseline accuracy
  3. For damage levels [5%, 10%, 20%, 30%, 50%]:
     a. SNN: randomly kill that fraction of excitatory cortex neurons
     b. MLP: randomly zero that fraction of hidden-layer weights
     c. Evaluate both on the same test set
  4. Report accuracy drop per damage level

A positive result: SNN degrades more gracefully (smaller accuracy
drop per % of damage) than the MLP.
"""

from __future__ import annotations

import copy
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
)
from src.device import DEVICE
from src.persistence import save_brain, load_brain

TRAIN_PER_CLASS = 300
TEST_PER_CLASS = 50
PRESENT_STEPS = 200
REST_STEPS = 50
DAMAGE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


# --- MLP baseline -----------------------------------------------------------

def mlp_train(X_train, y_train, classes, hidden=256, epochs=50, lr=0.01):
    """Train MLP and return weights."""
    n_classes = len(classes)
    n_features = X_train.shape[1]
    class_to_idx = {c: i for i, c in enumerate(classes)}

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

    return W1, b1, W2, b2


def mlp_eval(W1, b1, W2, b2, X_test, y_test, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_te = torch.tensor([class_to_idx[int(c)] for c in y_test], dtype=torch.int64, device=DEVICE)
    with torch.no_grad():
        h = torch.relu(X_te @ W1 + b1)
        preds = (h @ W2 + b2).argmax(dim=1)
        return (preds == y_te).sum().item() / len(y_te)


def mlp_damage(W1, b1, W2, b2, frac, seed=42):
    """Zero out a fraction of hidden units (columns of W1, rows of W2)."""
    W1d, b1d, W2d, b2d = W1.clone(), b1.clone(), W2.clone(), b2.clone()
    hidden = W1d.size(1)
    n_kill = int(hidden * frac)
    if n_kill == 0:
        return W1d, b1d, W2d, b2d
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(seed)
    kill_idx = torch.randperm(hidden, generator=rng, device=DEVICE)[:n_kill]
    W1d[:, kill_idx] = 0.0
    b1d[kill_idx] = 0.0
    W2d[kill_idx, :] = 0.0
    return W1d, b1d, W2d, b2d


# --- SNN damage --------------------------------------------------------------

def snn_damage(brain, frac, seed=42):
    """Kill a fraction of excitatory cortex neurons and their synapses."""
    cortex = brain.regions["cortex"]
    exc_idx = excitatory_cortex_indices(brain)
    n_kill = int(len(exc_idx) * frac)
    if n_kill == 0:
        return

    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(seed)
    kill_idx = exc_idx[torch.randperm(len(exc_idx), generator=rng, device=DEVICE)[:n_kill]]

    cortex.neuron_alive[kill_idx] = False

    ns = cortex.n_synapses
    if ns > 0:
        pre_dead = torch.isin(cortex.syn_pre[:ns], kill_idx.to(torch.int32))
        post_dead = torch.isin(cortex.syn_post[:ns], kill_idx.to(torch.int32))
        cortex.syn_alive[:ns][pre_dead | post_dead] = False

    # Also kill projection synapses targeting dead neurons
    for proj in brain.projections:
        pns = proj.n_synapses
        if pns > 0:
            post_dead = torch.isin(proj.syn_post[:pns], kill_idx.to(torch.int32))
            proj.syn_alive[:pns][post_dead] = False


def snn_eval(brain, X_train, y_train, X_test, y_test, classes):
    spike_t, voltage_t = build_response_templates(brain, X_train, y_train, classes)
    correct = 0
    for x, label in zip(X_test, y_test):
        pred, _ = predict_sample(brain, x, spike_t, voltage_t, classes)
        if pred == int(label):
            correct += 1
    return correct / len(y_test)


# --- Main -------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  GRACEFUL DEGRADATION BENCHMARK — SNN vs MLP")
    print("  (Train, then damage neurons/weights, re-evaluate)")
    print("=" * 68)

    X_train, y_train, X_test, y_test = load_reduced_mnist(
        classes=CLASSES, train_per_class=TRAIN_PER_CLASS, test_per_class=TEST_PER_CLASS,
    )

    # --- Train SNN ---
    print("\n  Training SNN...")
    t0 = time.time()
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
    print(f"    done in {time.time() - t0:.0f}s")

    # Save trained brain for reuse
    save_path = Path("/tmp/degradation_brain")
    save_brain(brain, save_path)

    # --- Train MLP ---
    print("  Training MLP...")
    W1, b1, W2, b2 = mlp_train(X_train, y_train, CLASSES)

    # --- Damage sweep ---
    print(f"\n  Damage levels: {[f'{d:.0%}' for d in DAMAGE_LEVELS]}\n")

    results = []
    for frac in DAMAGE_LEVELS:
        print(f"  --- {frac:.0%} damage ---")

        # SNN: reload clean brain and damage
        brain_d = load_brain(save_path)
        snn_damage(brain_d, frac, seed=SEED)
        snn_acc = snn_eval(brain_d, X_train, y_train, X_test, y_test, CLASSES)

        # MLP: damage weights
        W1d, b1d, W2d, b2d = mlp_damage(W1, b1, W2, b2, frac, seed=SEED)
        mlp_acc = mlp_eval(W1d, b1d, W2d, b2d, X_test, y_test, CLASSES)

        results.append((frac, snn_acc, mlp_acc))
        print(f"    SNN: {snn_acc:.1%}  |  MLP: {mlp_acc:.1%}")

    # --- Summary ---
    print("\n" + "=" * 68)
    print("  Summary")
    print("=" * 68)
    snn_base = results[0][1]
    mlp_base = results[0][2]
    print(f"  {'Damage':>8s}  {'SNN':>8s}  {'SNN drop':>9s}  {'MLP':>8s}  {'MLP drop':>9s}")
    for frac, snn_acc, mlp_acc in results:
        snn_drop = snn_base - snn_acc
        mlp_drop = mlp_base - mlp_acc
        print(f"  {frac:>7.0%}  {snn_acc:>7.1%}  {snn_drop:>+8.1%}  {mlp_acc:>7.1%}  {mlp_drop:>+8.1%}")
    print("=" * 68)


if __name__ == "__main__":
    main()
