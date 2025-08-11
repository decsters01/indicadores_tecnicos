from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from forex_indicators import (
    sma, ema, rsi, macd, stochastic, bollinger_bands, atr, adx, roc, zscore,
)
from .features import spins_to_colors, spins_to_highlow, spins_to_dozens, spins_to_columns
from .fx_adapter import spins_to_price, price_to_ohlcv


TaskEncoder = {
    "color": spins_to_colors,
    "highlow": spins_to_highlow,
    "dozen": spins_to_dozens,
    "column": spins_to_columns,
}


@dataclass
class IndicatorSpec:
    name: str
    inputs: List[str]  # subset of ["open","high","low","close","volume"]
    func: Callable
    kwargs: Dict


# Curadoria mínima de indicadores populares
INDICATORS: List[IndicatorSpec] = [
    IndicatorSpec("sma_14", ["close"], sma, {"period": 14}),
    IndicatorSpec("ema_14", ["close"], ema, {"period": 14}),
    IndicatorSpec("rsi_14", ["close"], rsi, {"period": 14}),
    IndicatorSpec("roc_12", ["close"], roc, {"period": 12}),
    IndicatorSpec("zscore_20", ["close"], zscore, {"period": 20}),
    IndicatorSpec("macd_hist", ["close"], macd, {}),  # usaremos a coluna 'hist'
    IndicatorSpec("stoch_k", ["high","low","close"], stochastic, {}),  # 'stoch_k'
    IndicatorSpec("bb_mid", ["close"], bollinger_bands, {}),  # 'bb_mid'
    IndicatorSpec("atr_14", ["high","low","close"], atr, {"period": 14}),
    IndicatorSpec("adx_14", ["high","low","close"], adx, {"period": 14}),
]


def _compute_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.Series:
    args = [df[c] for c in spec.inputs]
    out = spec.func(*args, **spec.kwargs) if spec.kwargs else spec.func(*args)
    if isinstance(out, pd.DataFrame):
        col_map = {
            "macd_hist": "hist",
            "stoch_k": "stoch_k",
            "bb_mid": "bb_mid",
            "adx_14": "adx",  # poderia usar adx ou plus_di/minus_di
        }
        key = col_map.get(spec.name)
        if key is None or key not in out.columns:
            # fallback: primeira coluna
            return out.iloc[:, 0]
        return out[key]
    return out


def _bin_series(x: pd.Series, bins: int) -> pd.Series:
    # Usa quantis; trata NaN mantendo-os
    try:
        q = pd.qcut(x.dropna(), q=bins, duplicates='drop')
    except ValueError:
        # muitos empates; reduz bins
        bins2 = max(2, min(bins, x.dropna().nunique()))
        if bins2 < 2:
            return pd.Series(index=x.index, dtype="category")
        q = pd.qcut(x.dropna(), q=bins2, duplicates='drop')
    binned = pd.Series(index=x.index, dtype=q.dtype)
    binned.loc[q.index] = q
    return binned


def walkforward_accuracy(spins: pd.Series, task: str, spec: IndicatorSpec, window: int = 200, bins: int = 10) -> float:
    if task not in TaskEncoder:
        raise ValueError("task inválida")
    # Construir OHLCV a partir dos giros
    price = spins_to_price(spins, method="cumsum_demeaned")
    df = price_to_ohlcv(price)

    # Indicador numérico
    ind = _compute_indicator(df, spec)
    if not isinstance(ind, pd.Series):
        ind = pd.Series(ind, index=df.index)
    signal = ind.astype(float)

    # Alinhar para prever próximo giro: usar signal_(t-1) -> y_t
    signal_lag = signal.shift(1)
    y = TaskEncoder[task](spins)

    # Discretizar sinal em bins
    bins_ser = _bin_series(signal_lag, bins=bins)

    # Walk-forward: mapa P(y|bin) na janela, prever por argmax
    correct = []
    for t in range(window + 1, len(spins)):
        hist_idx = bins_ser.index[:t-1]
        hist_bins = bins_ser.iloc[:t-1]
        hist_y = y.iloc[1:t]  # y_t começa em 1 por causa do lag
        # Filtrar apenas pares alinhados válidos
        valid = hist_bins.notna() & hist_y.notna()
        if valid.sum() == 0 or pd.isna(bins_ser.iloc[t-1]):
            correct.append(False)
            continue
        hb = hist_bins[valid]
        hy = hist_y[valid]
        # Frequências condicionais para o bin corrente
        current_bin = bins_ser.iloc[t-1]
        mask = hb == current_bin
        if mask.sum() == 0:
            # fallback: maioria global
            pred = hy[hy != 'Z'].mode()
            pred_label = pred.iloc[0] if len(pred) > 0 else None
        else:
            counts = hy[mask & (hy != 'Z')].value_counts()
            pred_label = counts.idxmax() if len(counts) > 0 else None
        truth = y.iloc[t]
        correct.append(pred_label is not None and truth != 'Z' and truth == pred_label)
    return float(np.mean(correct)) if len(correct) else np.nan


def evaluate_indicators_on_tasks(spins: pd.Series, tasks: List[str] = None, window: int = 200, bins: int = 10) -> pd.DataFrame:
    if tasks is None:
        tasks = ["color", "highlow", "dozen", "column"]
    rows = []
    for spec in INDICATORS:
        for task in tasks:
            acc = walkforward_accuracy(spins, task, spec, window=window, bins=bins)
            rows.append({"indicator": spec.name, "task": task, "window": window, "bins": bins, "accuracy": acc})
    return pd.DataFrame(rows).sort_values(["task","indicator"]).reset_index(drop=True)