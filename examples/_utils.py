from __future__ import annotations

import numpy as np

from src.brain import Brain


def quiet_steps(brain: Brain, n_steps: int) -> None:
    for _ in range(n_steps):
        brain.step()


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64)
    cm = np.zeros((len(label_arr), len(label_arr)), dtype=np.int64)
    label_to_idx = {int(label): idx for idx, label in enumerate(label_arr)}
    for true_label, pred_label in zip(y_true, y_pred):
        pred_int = int(pred_label)
        if pred_int < 0:
            continue
        cm[label_to_idx[int(true_label)], label_to_idx[pred_int]] += 1
    return cm
