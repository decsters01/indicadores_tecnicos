from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .features import (
    spins_to_colors,
    spins_to_highlow,
    spins_to_dozens,
    spins_to_columns,
    predict_next_by_rolling_frequency,
)

_TASK_TO_ENCODER = {
    "color": spins_to_colors,
    "highlow": spins_to_highlow,
    "dozen": spins_to_dozens,
    "column": spins_to_columns,
}

@dataclass
class TaskResult:
    task: str
    window: int
    accuracy: float
    n_samples: int


def _task_truth(spins: pd.Series, task: str) -> pd.Series:
    encoded = _TASK_TO_ENCODER[task](spins)
    return encoded


def evaluate_task_accuracy(spins: pd.Series, task: str, window: int) -> TaskResult:
    if task not in _TASK_TO_ENCODER:
        raise ValueError(f"task inválida: {task}")
    if window <= 0:
        raise ValueError("window deve ser > 0")

    truth = _task_truth(spins, task)
    preds = predict_next_by_rolling_frequency(spins, task, window)

    # Alinhar: previsão em t usa janelas até t, então comparamos pred[t] contra truth[t]
    # Ignorar períodos iniciais sem janela; ignorar casos 'Z' na verdade? Para simulação de aposta,
    # consideramos 'Z' como perda (sempre errado) se task não inclui zero.
    valid = preds.notna()
    truth2 = truth[valid]
    preds2 = preds[valid]

    # Qualquer ocorrência de 'Z' nos rótulos verdadeiros conta como erro, pois o preditor não prevê 'Z'
    correct = (truth2 != 'Z') & (truth2 == preds2)
    acc = correct.mean() if len(correct) > 0 else np.nan

    return TaskResult(task=task, window=window, accuracy=float(acc), n_samples=int(len(correct)))


def evaluate_all_tasks_summary(spins: pd.Series, windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
    rows = []
    for task in ["color", "highlow", "dozen", "column"]:
        for w in windows:
            r = evaluate_task_accuracy(spins, task, w)
            rows.append({"task": r.task, "window": r.window, "accuracy": r.accuracy, "n": r.n_samples})
    df = pd.DataFrame(rows).sort_values(["task", "window"]).reset_index(drop=True)
    return df