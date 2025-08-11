from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from forex_indicators import rsi, ema, macd, adx, zscore
from .fx_adapter import spins_to_price, price_to_ohlcv
from .features import (
    spins_to_colors,
    spins_to_highlow,
    spins_to_dozens,
    spins_to_columns,
    rolling_category_frequencies,
)

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


def _one_hot_last_k(labels: pd.Series, classes: List[str], k: int) -> pd.DataFrame:
    frames = []
    for lag in range(1, k + 1):
        lab_lag = labels.shift(lag)
        for c in classes:
            frames.append(((lab_lag == c).astype(float)).rename(f"lag{lag}_{c}"))
    return pd.concat(frames, axis=1)


def _fx_signals(spins: pd.Series) -> pd.DataFrame:
    price = spins_to_price(spins, method="cumsum_demeaned")
    df = price_to_ohlcv(price)
    # Indicadores básicos
    rsi14 = rsi(df["close"], 14).rename("rsi14")
    ema14 = ema(df["close"], 14).rename("ema14")
    macd_df = macd(df["close"])  # usa 'hist'
    macd_hist = macd_df["hist"].rename("macd_hist")
    adx14 = adx(df["high"], df["low"], df["close"], 14)["adx"].rename("adx14")
    # Normalizações simples
    zc20 = zscore(df["close"], 20).rename("zclose20")
    sig = pd.concat([rsi14, ema14, macd_hist, adx14, zc20], axis=1)
    # Lags de 1 e 2
    sig_l1 = sig.shift(1).add_suffix("_l1")
    sig_l2 = sig.shift(2).add_suffix("_l2")
    return pd.concat([sig_l1, sig_l2], axis=1)


def build_features_with_fx(spins: pd.Series, task: str, n_lags: int = 20, windows: List[int] = [20, 50, 100]) -> Tuple[pd.DataFrame, pd.Series]:
    if task not in _TASK_TO_ENCODER:
        raise ValueError("task inválida")
    labels = _TASK_TO_ENCODER[task](spins)
    classes = _TASK_LABELS[task]

    lag_df = _one_hot_last_k(labels, classes, n_lags)

    freq_frames = []
    for w in windows:
        f = rolling_category_frequencies(spins, task, w).shift(1)
        f = f.rename(columns={c: f"freq{w}_{c}" for c in f.columns})
        freq_frames.append(f)

    fx_df = _fx_signals(spins)

    X = pd.concat([lag_df] + freq_frames + [fx_df], axis=1)
    y = labels

    mask_valid = (y != 'Z')
    X = X[mask_valid]
    y = y[mask_valid]
    X = X.dropna()
    y = y.loc[X.index]
    return X, y