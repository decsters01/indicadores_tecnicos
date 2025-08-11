from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Convenções de rótulos
# - Cores: "R" (red), "B" (black), "Z" (zero)
# - High/Low: "H", "L", "Z"
# - Dúzias: "D1", "D2", "D3", "Z"
# - Colunas: "C1", "C2", "C3", "Z"

_RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
_BLACK_NUMBERS = set(range(1,37)) - _RED_NUMBERS

_COLUMN_1 = {1,4,7,10,13,16,19,22,25,28,31,34}
_COLUMN_2 = {2,5,8,11,14,17,20,23,26,29,32,35}
_COLUMN_3 = {3,6,9,12,15,18,21,24,27,30,33,36}

_DOZEN_1 = set(range(1,13))
_DOZEN_2 = set(range(13,25))
_DOZEN_3 = set(range(25,37))


def encode_spin_european(n: int) -> Dict[str, str]:
    if n == 0:
        return {"color": "Z", "highlow": "Z", "dozen": "Z", "column": "Z"}
    color = "R" if n in _RED_NUMBERS else "B"
    highlow = "L" if 1 <= n <= 18 else "H"
    if n in _DOZEN_1:
        dozen = "D1"
    elif n in _DOZEN_2:
        dozen = "D2"
    else:
        dozen = "D3"
    if n in _COLUMN_1:
        column = "C1"
    elif n in _COLUMN_2:
        column = "C2"
    else:
        column = "C3"
    return {"color": color, "highlow": highlow, "dozen": dozen, "column": column}


def _ensure_series(spins: pd.Series) -> pd.Series:
    if not isinstance(spins, pd.Series):
        spins = pd.Series(spins)
    return spins.astype(int)


def spins_to_colors(spins: pd.Series) -> pd.Series:
    spins = _ensure_series(spins)
    return spins.apply(lambda n: "Z" if n == 0 else ("R" if n in _RED_NUMBERS else "B"))


def spins_to_highlow(spins: pd.Series) -> pd.Series:
    spins = _ensure_series(spins)
    return spins.apply(lambda n: "Z" if n == 0 else ("L" if 1 <= n <= 18 else "H"))


def spins_to_dozens(spins: pd.Series) -> pd.Series:
    spins = _ensure_series(spins)
    def _dz(n: int) -> str:
        if n == 0:
            return "Z"
        if n in _DOZEN_1:
            return "D1"
        if n in _DOZEN_2:
            return "D2"
        return "D3"
    return spins.apply(_dz)


def spins_to_columns(spins: pd.Series) -> pd.Series:
    spins = _ensure_series(spins)
    def _col(n: int) -> str:
        if n == 0:
            return "Z"
        if n in _COLUMN_1:
            return "C1"
        if n in _COLUMN_2:
            return "C2"
        return "C3"
    return spins.apply(_col)


_TASK_TO_ENCODER = {
    "color": spins_to_colors,
    "highlow": spins_to_highlow,
    "dozen": spins_to_dozens,
    "column": spins_to_columns,
}

_TASK_LABELS = {
    "color": ["R", "B"],
    "highlow": ["H", "L"],
    "dozen": ["D1", "D2", "D3"],
    "column": ["C1", "C2", "C3"],
}


def rolling_category_frequencies(spins: pd.Series, task: str, window: int) -> pd.DataFrame:
    if task not in _TASK_TO_ENCODER:
        raise ValueError(f"task inválida: {task}")
    if window <= 0:
        raise ValueError("window deve ser > 0")
    labels = _TASK_LABELS[task]
    encoded = _TASK_TO_ENCODER[task](spins)
    # one-hot ignorando Z
    freq_frames = []
    for label in labels:
        indicator = (encoded == label).astype(float)
        freq = indicator.rolling(window=window, min_periods=window).mean()
        freq_frames.append(freq.rename(label))
    return pd.concat(freq_frames, axis=1)


def predict_next_by_rolling_frequency(spins: pd.Series, task: str, window: int) -> pd.Series:
    freqs = rolling_category_frequencies(spins, task, window)
    # Escolhe a categoria com maior frequência nas últimas 'window' observações
    pred = freqs.idxmax(axis=1)
    return pred