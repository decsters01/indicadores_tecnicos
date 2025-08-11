from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ==========================
# Helpers internos
# ==========================

def _validate_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} deve ser um pandas.Series")
    return series.astype(float)


def _validate_ohlcv(df_like, required=("high", "low", "close")) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        df = df_like.copy()
    else:
        raise TypeError("Entrada deve ser um pandas.DataFrame com colunas open, high, low, close, volume")

    cols = set(c.lower() for c in df.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {missing}")

    # Normaliza nomes para minúsculas
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    series = _validate_series(series, "series")
    if period <= 0:
        raise ValueError("period deve ser > 0")
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


# ==========================
# 1) SMA - Simple Moving Average
# ==========================

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average (Média Móvel Simples)

    Parâmetros:
    - series: série de preços (por ex. close)
    - period: janela de cálculo
    """
    series = _validate_series(series, "series")
    if period <= 0:
        raise ValueError("period deve ser > 0")
    return series.rolling(window=period, min_periods=period).mean()


# ==========================
# 2) EMA - Exponential Moving Average
# ==========================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Média Móvel Exponencial)"""
    series = _validate_series(series, "series")
    if period <= 0:
        raise ValueError("period deve ser > 0")
    return series.ewm(span=period, adjust=False).mean()


# ==========================
# 3) WMA - Weighted Moving Average
# ==========================

def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (Média Móvel Ponderada)"""
    series = _validate_series(series, "series")
    if period <= 0:
        raise ValueError("period deve ser > 0")

    weights = np.arange(1, period + 1, dtype=float)

    def _calc(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        return np.dot(x, weights[::-1]) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(lambda x: _calc(np.asarray(x)), raw=False)


# ==========================
# 4) MACD - Moving Average Convergence Divergence
# ==========================

def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """MACD com linha de sinal e histograma"""
    close = _validate_series(close, "close")
    ema_fast = ema(close, fast_period)
    ema_slow = ema(close, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


# ==========================
# 5) RSI - Relative Strength Index (Wilder)
# ==========================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder"""
    close = _validate_series(close, "close")
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = _wilder_ema(gains, period)
    avg_loss = _wilder_ema(losses, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# ==========================
# 6) Stochastic Oscillator (%K, %D)
# ==========================

def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Oscilador Estocástico (%K e %D)"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


# ==========================
# 7) Bollinger Bands
# ==========================

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger (média, superior, inferior)"""
    close = _validate_series(close, "close")
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


# ==========================
# 8) ATR - Average True Range
# ==========================

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return _wilder_ema(tr, period)


# ==========================
# 9) ADX (+DI, -DI)
# ==========================

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index com +DI e -DI"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smoothed = _wilder_ema(pd.Series(tr, index=close.index), period)
    plus_dm_smoothed = _wilder_ema(pd.Series(plus_dm, index=close.index), period)
    minus_dm_smoothed = _wilder_ema(pd.Series(minus_dm, index=close.index), period)

    plus_di = 100 * (plus_dm_smoothed / tr_smoothed.replace(0.0, np.nan))
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed.replace(0.0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = _wilder_ema(dx, period)

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


# ==========================
# 10) CCI - Commodity Channel Index
# ==========================

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    tp = (high + low + close) / 3.0
    sma_tp = sma(tp, period)
    mean_dev = tp.rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev)


# ==========================
# 11) OBV - On-Balance Volume
# ==========================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


# ==========================
# 12) MFI - Money Flow Index
# ==========================

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    delta_tp = tp.diff()

    pos_mf = np.where(delta_tp > 0, raw_mf, 0.0)
    neg_mf = np.where(delta_tp < 0, raw_mf, 0.0)

    pos_mf_sum = pd.Series(pos_mf, index=tp.index).rolling(window=period, min_periods=period).sum()
    neg_mf_sum = pd.Series(neg_mf, index=tp.index).rolling(window=period, min_periods=period).sum()

    mr = pos_mf_sum / neg_mf_sum.replace(0.0, np.nan)
    return 100 - (100 / (1 + mr))


# ==========================
# 13) Parabolic SAR
# ==========================

def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR (implementação clássica). Retorna série SAR.

    Observação: cálculo iterativo e pode variar ligeiramente de outras bibliotecas."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    length = len(high)
    sar = np.zeros(length)

    # Inicialização
    uptrend = True
    af = step
    ep = low.iloc[0]
    sar[0] = low.iloc[0]

    if length >= 2:
        if high.iloc[1] + low.iloc[1] < high.iloc[0] + low.iloc[0]:
            uptrend = False
            ep = high.iloc[0]
            sar[0] = high.iloc[0]
        else:
            uptrend = True
            ep = low.iloc[0]
            sar[0] = low.iloc[0]

    for i in range(1, length):
        prev_sar = sar[i - 1]
        if uptrend:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[i] if i < length else low.iloc[i - 1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = high.iloc[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[i] if i < length else high.iloc[i - 1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = low.iloc[i]
                af = step

    return pd.Series(sar, index=high.index)


# ==========================
# 14) Ichimoku Cloud
# ==========================

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Ichimoku Cloud: Tenkan, Kijun, Senkou A, Senkou B, Chikou"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    chikou = close.shift(-26)

    return pd.DataFrame({
        "ichimoku_tenkan": tenkan,
        "ichimoku_kijun": kijun,
        "ichimoku_senkou_a": senkou_a,
        "ichimoku_senkou_b": senkou_b,
        "ichimoku_chikou": chikou,
    })


# ==========================
# 15) Pivot Points (Clássico)
# ==========================

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Pivot Points Clássicos (P, S1-3, R1-3)"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)

    return pd.DataFrame({
        "pp": p,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
        "r3": r3,
        "s3": s3,
    })


# ==========================
# 16) VWAP - Volume Weighted Average Price
# ==========================

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP cumulativo (não reinicia por sessão)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    tp = (high + low + close) / 3.0
    cum_vp = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0.0, np.nan)
    return cum_vp / cum_vol


# ==========================
# 17) ROC - Rate of Change
# ==========================

def roc(series: pd.Series, period: int = 12, as_percent: bool = True) -> pd.Series:
    """Rate of Change do preço"""
    series = _validate_series(series, "series")
    shifted = series.shift(period)
    roc_val = series / shifted - 1.0
    if as_percent:
        roc_val *= 100.0
    return roc_val


# ==========================
# 18) Williams %R
# ==========================

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    return -100 * (highest_high - close) / (highest_high - lowest_low)


# ==========================
# 19) TRIX
# ==========================

def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """TRIX: ROC de 1 período da EMA tripla"""
    close = _validate_series(close, "close")
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 100 * (ema3 / ema3.shift(1) - 1.0)


# ==========================
# 20) Donchian Channels
# ==========================

def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Canais de Donchian (superior, inferior, meio)"""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return pd.DataFrame({"donchian_upper": upper, "donchian_lower": lower, "donchian_mid": middle})