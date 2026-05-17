"""
MNIST diagnosis experiments for the unsupervised STDP benchmark.

This script runs short, focused experiments that separate three likely
failure modes behind the 64.6% plateau:

1. Readout mismatch: the cortex may encode more than the current decoder
   extracts.
2. Winner lock-in: strong lateral inhibition may let easy classes
   monopolize excitatory neurons too early.
3. Plasticity calibration drift: long presentations change STDP event
   counts and may require retuning STDP/theta dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import examples.mnist_benchmark as mn
from examples._utils import confusion_matrix
from src.persistence import load_brain, save_brain


@dataclass(frozen=True)
class ProtocolConfig:
    train_per_class: int = 100
    readout_per_class: int = 30
    test_per_class: int = 20
    train_present_steps: int = 100
    assign_present_steps: int = 50
    test_present_steps: int = 50
    rest_steps: int = 25
    epochs: int = 1
    test_repeats: int = 2
    spike_score_weight: float = 0.7
    voltage_score_weight: float = 0.3
    stdp_scale: float = 0.2
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.0105
    inh_lateral_weight: float = 12.0
    exc_to_inh_weight: float = 8.0
    theta_plus: float = 0.10
    theta_leak: float = 0.005
    seed: int = mn.SEED


FAST_BASE = ProtocolConfig()


@contextmanager
def patched_benchmark_globals(config: ProtocolConfig):
    keys = {
        "TRAIN_PER_CLASS": config.train_per_class,
        "READOUT_PER_CLASS": config.readout_per_class,
        "TEST_PER_CLASS": config.test_per_class,
        "TRAIN_PRESENT_STEPS": config.train_present_steps,
        "ASSIGN_PRESENT_STEPS": config.assign_present_steps,
        "TEST_PRESENT_STEPS": config.test_present_steps,
        "REST_STEPS": config.rest_steps,
        "EPOCHS": config.epochs,
        "TEST_REPEATS": config.test_repeats,
        "SPIKE_SCORE_WEIGHT": config.spike_score_weight,
        "VOLTAGE_SCORE_WEIGHT": config.voltage_score_weight,
        "STDP_SCALE": config.stdp_scale,
        "STDP_A_PLUS": config.stdp_a_plus,
        "STDP_A_MINUS": config.stdp_a_minus,
        "INH_LATERAL_WEIGHT": config.inh_lateral_weight,
        "EXC_TO_INH_WEIGHT": config.exc_to_inh_weight,
        "THETA_PLUS": config.theta_plus,
        "THETA_LEAK": config.theta_leak,
    }
    old = {key: getattr(mn, key) for key in keys}
    try:
        for key, value in keys.items():
            setattr(mn, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(mn, key, value)


def load_dataset(config: ProtocolConfig):
    return mn.load_reduced_mnist(
        classes=mn.CLASSES,
        train_per_class=config.train_per_class,
        test_per_class=config.test_per_class,
        seed=config.seed,
    )


def label_balance_metrics(neuron_labels: np.ndarray) -> dict[str, float | int | dict[str, int]]:
    counts = mn.neuron_label_counts(neuron_labels, mn.CLASSES)
    labelled = int(np.sum(neuron_labels >= 0))
    total = len(neuron_labels)
    count_arr = np.asarray([counts[int(cls)] for cls in mn.CLASSES], dtype=np.float64)
    if labelled > 0:
        probs = count_arr / labelled
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log2(probs[nz])).sum() / np.log2(len(mn.CLASSES)))
        max_share = float(probs.max())
    else:
        entropy = 0.0
        max_share = 1.0
    return {
        "labelled": labelled,
        "unlabelled": total - labelled,
        "label_entropy": entropy,
        "max_class_share": max_share,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }


def format_counts(class_counts: dict[str, int]) -> str:
    return " ".join(f"{cls}:{count}" for cls, count in class_counts.items())


def save_trained_brain(brain, label: str) -> Path:
    root = Path(tempfile.gettempdir()) / "brain_mnist_diagnosis" / label
    save_brain(brain, root)
    return root


def train_protocol(config: ProtocolConfig, label: str):
    with patched_benchmark_globals(config):
        X_train, y_train, X_test, y_test = load_dataset(config)
        print(
            f"  Dataset: train={len(X_train)} test={len(X_test)} "
            f"train_steps={config.train_present_steps} rest={config.rest_steps}"
        )
        brain = mn.build_brain(seed=config.seed)
        print(brain.summary())
        mn.train_unsupervised(
            brain,
            X_train,
            epochs=config.epochs,
            train_present_steps=config.train_present_steps,
            seed=config.seed,
            log_every=max(25, len(X_train) // 5),
        )
        path = save_trained_brain(brain, label)
        return path, X_train, y_train, X_test, y_test


def evaluate_protocol(
    brain_path: Path,
    config: ProtocolConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict[str, object]:
    with patched_benchmark_globals(config):
        brain = load_brain(brain_path)
        readout_start = time.time()
        X_readout, y_readout = mn.build_readout_subset(
            X_train,
            y_train,
            readout_per_class=config.readout_per_class,
            classes=mn.CLASSES,
            seed=config.seed,
        )
        exc_idx, neuron_labels, spike_templates, voltage_templates = mn.build_readout(
            brain,
            X_readout,
            y_readout,
            classes=mn.CLASSES,
        )
        readout_time = time.time() - readout_start
        eval_start = time.time()
        acc, preds = mn.evaluate(brain, X_test, y_test, spike_templates, voltage_templates, mn.CLASSES)
        eval_time = time.time() - eval_start
        balance = label_balance_metrics(neuron_labels)
        result = {
            "label": label,
            "config": asdict(config),
            "accuracy": acc,
            "no_response": int(np.sum(preds < 0)),
            "readout_time_s": readout_time,
            "eval_time_s": eval_time,
            "labelled": balance["labelled"],
            "unlabelled": balance["unlabelled"],
            "label_entropy": balance["label_entropy"],
            "max_class_share": balance["max_class_share"],
            "class_counts": balance["class_counts"],
            "confusion_matrix": confusion_matrix(y_test, preds, labels=mn.CLASSES).tolist(),
            "exc_neurons": int(len(exc_idx)),
        }
    print(
        f"  {label:24s} acc={acc:5.1%} no_resp={result['no_response']:3d} "
        f"entropy={result['label_entropy']:.3f} max_share={result['max_class_share']:.3f}"
    )
    print(f"    class_counts: {format_counts(result['class_counts'])}")
    return result


def best_result(results: list[dict[str, object]]) -> dict[str, object]:
    return max(
        results,
        key=lambda item: (
            float(item["accuracy"]),
            float(item["label_entropy"]),
            -float(item["max_class_share"]),
        ),
    )


def run_readout_stress() -> dict[str, object]:
    print("\n" + "=" * 72)
    print("READOUT-ONLY STRESS TEST")
    print("=" * 72)
    train_config = FAST_BASE
    brain_path, X_train, y_train, X_test, y_test = train_protocol(train_config, "readout_stress")

    variants = [
        ("baseline_50x2_blend", train_config),
        (
            "matched_100x2_blend",
            replace(train_config, assign_present_steps=100, test_present_steps=100),
        ),
        (
            "matched_100x5_blend",
            replace(train_config, assign_present_steps=100, test_present_steps=100, test_repeats=5),
        ),
        (
            "matched_100x5_spike",
            replace(
                train_config,
                assign_present_steps=100,
                test_present_steps=100,
                test_repeats=5,
                spike_score_weight=1.0,
                voltage_score_weight=0.0,
            ),
        ),
        (
            "matched_100x5_voltage",
            replace(
                train_config,
                assign_present_steps=100,
                test_present_steps=100,
                test_repeats=5,
                spike_score_weight=0.0,
                voltage_score_weight=1.0,
            ),
        ),
    ]

    results = [
        evaluate_protocol(brain_path, variant_config, X_train, y_train, X_test, y_test, label)
        for label, variant_config in variants
    ]
    winner = best_result(results)
    baseline = results[0]
    return {
        "experiment": "readout_stress",
        "train_config": asdict(train_config),
        "results": results,
        "best": winner,
        "delta_vs_baseline": float(winner["accuracy"]) - float(baseline["accuracy"]),
    }


def run_competition_sweep() -> dict[str, object]:
    print("\n" + "=" * 72)
    print("COMPETITION SWEEP")
    print("=" * 72)
    results = []
    for inh_weight in (6.0, 8.0, 10.0, 12.0):
        config = replace(FAST_BASE, inh_lateral_weight=inh_weight)
        label = f"inh_lateral={inh_weight:.1f}"
        print(f"\n-- {label} --")
        brain_path, X_train, y_train, X_test, y_test = train_protocol(config, f"competition_{inh_weight:.1f}")
        results.append(evaluate_protocol(brain_path, config, X_train, y_train, X_test, y_test, label))
    return {
        "experiment": "competition_sweep",
        "results": results,
        "best": best_result(results),
    }


def run_stdp_sweep() -> dict[str, object]:
    print("\n" + "=" * 72)
    print("STDP SCALE SWEEP")
    print("=" * 72)
    results = []
    for scale in (0.10, 0.20, 0.30, 0.40):
        config = replace(FAST_BASE, stdp_scale=scale)
        label = f"stdp_scale={scale:.2f}"
        print(f"\n-- {label} --")
        brain_path, X_train, y_train, X_test, y_test = train_protocol(config, f"stdp_{scale:.2f}")
        results.append(evaluate_protocol(brain_path, config, X_train, y_train, X_test, y_test, label))
    return {
        "experiment": "stdp_sweep",
        "results": results,
        "best": best_result(results),
    }


def run_theta_validation() -> dict[str, object]:
    print("\n" + "=" * 72)
    print("THETA / HOMEOSTASIS VALIDATION")
    print("=" * 72)
    variants = [
        ("theta_default", replace(FAST_BASE, theta_plus=0.10, theta_leak=0.005)),
        ("theta_low_plus", replace(FAST_BASE, theta_plus=0.05, theta_leak=0.005)),
        ("theta_high_plus", replace(FAST_BASE, theta_plus=0.15, theta_leak=0.005)),
        ("theta_fast_leak", replace(FAST_BASE, theta_plus=0.10, theta_leak=0.010)),
    ]
    results = []
    for label, config in variants:
        print(f"\n-- {label} --")
        brain_path, X_train, y_train, X_test, y_test = train_protocol(config, label)
        results.append(evaluate_protocol(brain_path, config, X_train, y_train, X_test, y_test, label))
    return {
        "experiment": "theta_validation",
        "results": results,
        "best": best_result(results),
    }


def choose_long_run_candidate(summaries: list[dict[str, object]]) -> dict[str, object]:
    readout = next(item for item in summaries if item["experiment"] == "readout_stress")
    competition = next(item for item in summaries if item["experiment"] == "competition_sweep")
    stdp = next(item for item in summaries if item["experiment"] == "stdp_sweep")
    theta = next(item for item in summaries if item["experiment"] == "theta_validation")

    candidates = [
        {
            "source": "readout",
            "label": readout["best"]["label"],
            "accuracy": readout["best"]["accuracy"],
            "label_entropy": readout["best"]["label_entropy"],
            "max_class_share": readout["best"]["max_class_share"],
            "reason": "best decoder-only variant",
        },
        {
            "source": "competition",
            "label": competition["best"]["label"],
            "accuracy": competition["best"]["accuracy"],
            "label_entropy": competition["best"]["label_entropy"],
            "max_class_share": competition["best"]["max_class_share"],
            "reason": "best inhibition sweep point",
        },
        {
            "source": "stdp",
            "label": stdp["best"]["label"],
            "accuracy": stdp["best"]["accuracy"],
            "label_entropy": stdp["best"]["label_entropy"],
            "max_class_share": stdp["best"]["max_class_share"],
            "reason": "best STDP sweep point",
        },
        {
            "source": "theta",
            "label": theta["best"]["label"],
            "accuracy": theta["best"]["accuracy"],
            "label_entropy": theta["best"]["label_entropy"],
            "max_class_share": theta["best"]["max_class_share"],
            "reason": "best theta/homeostasis point",
        },
    ]

    # Prefer readout if it delivers a strong gain without retraining.
    if float(readout["delta_vs_baseline"]) >= 0.05:
        return {
            "selected_from": "readout",
            "label": readout["best"]["label"],
            "reason": "Decoder-only changes produced a >=5pp gain, so another long retrain is unnecessary before fixing readout.",
            "candidates": candidates,
        }

    best = max(
        candidates,
        key=lambda item: (
            float(item["accuracy"]),
            float(item["label_entropy"]),
            -float(item["max_class_share"]),
        ),
    )
    return {
        "selected_from": best["source"],
        "label": best["label"],
        "reason": f"Best short-run trade-off came from {best['reason']}.",
        "candidates": candidates,
    }


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short MNIST diagnosis sweeps.")
    parser.add_argument(
        "--experiment",
        choices=["all", "readout", "competition", "stdp", "theta"],
        default="all",
        help="Which diagnosis block to run.",
    )
    parser.add_argument(
        "--summary",
        default="results/mnist_diagnosis_summary.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    start = time.time()
    summaries = []

    if args.experiment in {"all", "readout"}:
        summaries.append(run_readout_stress())
    if args.experiment in {"all", "competition"}:
        summaries.append(run_competition_sweep())
    if args.experiment in {"all", "stdp"}:
        summaries.append(run_stdp_sweep())
    if args.experiment in {"all", "theta"}:
        summaries.append(run_theta_validation())

    payload: dict[str, object] = {
        "base_config": asdict(FAST_BASE),
        "elapsed_s": time.time() - start,
        "summaries": summaries,
    }
    if args.experiment == "all":
        payload["recommended_long_run"] = choose_long_run_candidate(summaries)

    write_summary(Path(args.summary), payload)
    print("\n" + "=" * 72)
    print(f"Summary written to {args.summary}")
    if "recommended_long_run" in payload:
        rec = payload["recommended_long_run"]
        print(
            f"Recommended next long run: {rec['selected_from']} / {rec['label']}\n"
            f"Reason: {rec['reason']}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
