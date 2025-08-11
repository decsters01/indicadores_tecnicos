from __future__ import annotations

from typing import Dict, List, Tuple, Callable
import itertools
import numpy as np
import pandas as pd

from .ml import train_and_evaluate


def _iter_grid(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def grid_search(spins: pd.Series, task: str, grid: Dict[str, List],
                feature_builder: Callable = None, max_models: int | None = None) -> pd.DataFrame:
    rows = []
    for i, params in enumerate(_iter_grid(grid)):
        if max_models is not None and i >= max_models:
            break
        fb = feature_builder
        # train_and_evaluate aceita apenas hiperparâmetros básicos; passamos via kwargs
        out = train_and_evaluate(
            spins,
            task=task,
            n_lags=params.get("n_lags", 20),
            windows=params.get("windows", [20, 50, 100]),
            lr=params.get("lr", 0.05),
            epochs=params.get("epochs", 300),
            reg=params.get("reg", 1e-3),
        ) if fb is None else train_and_evaluate_with_builder(
            spins,
            task=task,
            feature_builder=fb,
            n_lags=params.get("n_lags", 20),
            windows=params.get("windows", [20, 50, 100]),
            lr=params.get("lr", 0.05),
            epochs=params.get("epochs", 300),
            reg=params.get("reg", 1e-3),
        )
        rows.append({"params": params, "val_accuracy": out["val_accuracy"], "n_train": out["n_train"], "n_val": out["n_val"]})
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    return df


def train_and_evaluate_with_builder(spins: pd.Series, task: str, feature_builder: Callable, n_lags: int, windows: List[int], lr: float, epochs: int, reg: float):
    # Carrega internamente o modelo e substitui a construção de features
    from .ml import LogisticOVR, _TASK_LABELS
    X, y = feature_builder(spins, task, n_lags=n_lags, windows=windows)
    classes = _TASK_LABELS[task]
    split = int(len(X) * 0.8)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xva, yva = X.iloc[split:], y.iloc[split:]
    model = LogisticOVR(classes_=classes)
    model.fit(Xtr, ytr, lr=lr, epochs=epochs, reg=reg)
    acc = (model.predict(Xva) == yva).mean()
    return {"model": model, "feature_names": X.columns.tolist(), "val_accuracy": float(acc), "n_train": int(len(Xtr)), "n_val": int(len(Xva))}