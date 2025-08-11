from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ==========================
# Helpers internos
# ==========================

def _validate_series(series: pd.Series, name: str) -> pd.Series:
    # Aceita pandas.Series diretamente; para arrays/listas, converte para Series
    if not isinstance(series, pd.Series):
        if isinstance(series, (np.ndarray, list, tuple, pd.Index)):
            series = pd.Series(series)
        else:
            raise TypeError(f"{name} deve ser um pandas.Series ou array-like")
    try:
        return series.astype(float)
    except Exception as exc:
        raise TypeError(f"{name} não pôde ser convertido para float: {exc}")


def normalize_ohlcv(df_like: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas para open/high/low/close/volume.

    - Aceita variações comuns de provedores (ex.: Close, Adj Close, VOL, etc.)
    - Mantém colunas extras intactas
    - Se existir apenas 'adj close', cria 'close' a partir dela
    """
    if not isinstance(df_like, pd.DataFrame):
        raise TypeError("Entrada deve ser um pandas.DataFrame")

    df = df_like.copy()

    def _canon(name: str) -> str:
        name = name.lower()
        # remove caracteres não alfanuméricos para facilitar matching
        return "".join(ch for ch in name if ch.isalnum())

    targets_to_aliases = {
        "open": {"open", "o", "op", "openingprice"},
        "high": {"high", "h", "hi"},
        "low": {"low", "l", "lo"},
        "close": {"close", "c", "cl", "last", "price", "adjclose"},
        "volume": {"volume", "vol", "v", "qty", "quantity"},
    }

    alias_to_target = {}
    for target, aliases in targets_to_aliases.items():
        for alias in aliases:
            alias_to_target[alias] = target

    rename_map: dict[str, str] = {}
    for col in df.columns:
        canon = _canon(str(col))
        if canon in alias_to_target:
            rename_map[col] = alias_to_target[canon]

    if rename_map:
        df = df.rename(columns=rename_map)

    # Se houver apenas adj close mapeado para 'close', já estará coberto.
    # Garante nomes em minúsculas para consistência
    df = df.rename(columns=lambda c: str(c).lower())

    # Caso 'close' não exista mas 'adj close' exista (não mapeado por algum motivo)
    if "close" not in df.columns:
        for c in list(df.columns):
            if _canon(c) == "adjclose":
                df["close"] = df[c]
                break

    return df


def _validate_ohlcv(df_like, required=("high", "low", "close")) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        df = normalize_ohlcv(df_like)
    else:
        raise TypeError("Entrada deve ser um pandas.DataFrame com colunas open, high, low, close, volume")

    cols = set(c.lower() for c in df.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {missing}")

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


# ==========================
# 21) Keltner Channels
# ==========================

def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels: média EMA(close, period) ± multiplier * ATR(atr_period)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    mid = ema(close, period)
    atr_val = atr(high, low, close, atr_period)
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return pd.DataFrame({"kc_mid": mid, "kc_upper": upper, "kc_lower": lower})


# ==========================
# 22) Supertrend
# ==========================

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicador: retorna linhas de tendência e sinal (+1/-1)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_band_basic = hl2 + multiplier * atr_val
    lower_band_basic = hl2 - multiplier * atr_val

    upper_band = upper_band_basic.copy()
    lower_band = lower_band_basic.copy()

    for i in range(1, len(close)):
        if not np.isnan(upper_band.iat[i - 1]):
            upper_band.iat[i] = (
                upper_band_basic.iat[i]
                if (upper_band_basic.iat[i] < upper_band.iat[i - 1]) or (close.iat[i - 1] > upper_band.iat[i - 1])
                else upper_band.iat[i - 1]
            )
        if not np.isnan(lower_band.iat[i - 1]):
            lower_band.iat[i] = (
                lower_band_basic.iat[i]
                if (lower_band_basic.iat[i] > lower_band.iat[i - 1]) or (close.iat[i - 1] < lower_band.iat[i - 1])
                else lower_band.iat[i - 1]
            )

    supertrend_line = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=float)

    for i in range(len(close)):
        if i == 0:
            supertrend_line.iat[i] = np.nan
            trend.iat[i] = 1.0
            continue
        prev_line = supertrend_line.iat[i - 1]
        prev_trend = trend.iat[i - 1]
        if pd.isna(prev_line):
            prev_line = upper_band.iat[i - 1]
            prev_trend = 1.0
        if close.iat[i] > upper_band.iat[i - 1]:
            supertrend_line.iat[i] = lower_band.iat[i]
            trend.iat[i] = 1.0
        elif close.iat[i] < lower_band.iat[i - 1]:
            supertrend_line.iat[i] = upper_band.iat[i]
            trend.iat[i] = -1.0
        else:
            supertrend_line.iat[i] = prev_line
            trend.iat[i] = prev_trend
            if (prev_trend == 1.0) and (lower_band.iat[i] > prev_line):
                supertrend_line.iat[i] = lower_band.iat[i]
            if (prev_trend == -1.0) and (upper_band.iat[i] < prev_line):
                supertrend_line.iat[i] = upper_band.iat[i]

    return pd.DataFrame({
        "supertrend": supertrend_line,
        "supertrend_trend": trend,
        "supertrend_upper": upper_band,
        "supertrend_lower": lower_band,
    })


# ==========================
# 23) PPO - Percentage Price Oscillator
# ==========================

def ppo(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """PPO, sua linha de sinal e histograma."""
    close = _validate_series(close, "close")
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)
    signal_line = ema(ppo_line, signal)
    hist = ppo_line - signal_line
    return pd.DataFrame({"ppo": ppo_line, "ppo_signal": signal_line, "ppo_hist": hist})


# ==========================
# 24) KAMA - Kaufman Adaptive Moving Average
# ==========================

def kama(
    series: pd.Series,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    series = _validate_series(series, "series")
    change = series.diff(er_period).abs()
    volatility = series.diff().abs().rolling(er_period).sum()
    er = change / volatility.replace(0.0, np.nan)

    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama_vals = pd.Series(index=series.index, dtype=float)
    for i, idx in enumerate(series.index):
        if i == 0:
            kama_vals.iat[i] = series.iat[i]
        else:
            kama_vals.iat[i] = kama_vals.iat[i - 1] + sc.iat[i] * (series.iat[i] - kama_vals.iat[i - 1])
    return kama_vals


# ==========================
# 25) TSI - True Strength Index
# ==========================

def tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index."""
    close = _validate_series(close, "close")
    momentum = close.diff()
    ema1 = ema(momentum.fillna(0.0), long)
    ema2 = ema(ema1, short)
    ema1_abs = ema(momentum.abs().fillna(0.0), long)
    ema2_abs = ema(ema1_abs, short)
    return 100 * (ema2 / ema2_abs.replace(0.0, np.nan))


# ==========================
# 26) DPO - Detrended Price Oscillator
# ==========================

def dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """DPO: close deslocado menos SMA(period)"""
    close = _validate_series(close, "close")
    shift = int(period / 2) + 1
    return close.shift(shift) - sma(close, period)


# ==========================
# 27) Aroon (Up, Down, Osc)
# ==========================

def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon Up/Down e Oscillator."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    def _last_idx_of_max(x: np.ndarray) -> int:
        return int(np.argmax(x))

    def _last_idx_of_min(x: np.ndarray) -> int:
        return int(np.argmin(x))

    rolling_high_idx = high.rolling(window=period, min_periods=period).apply(_last_idx_of_max, raw=True)
    rolling_low_idx = low.rolling(window=period, min_periods=period).apply(_last_idx_of_min, raw=True)

    aroon_up = 100 * (period - 1 - rolling_high_idx) / (period - 1)
    aroon_down = 100 * (period - 1 - rolling_low_idx) / (period - 1)
    osc = aroon_up - aroon_down
    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down, "aroon_osc": osc})


# ==========================
# 28) CMF - Chaikin Money Flow
# ==========================

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfv = mfm * volume
    return mfv.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum()


# ==========================
# 29) Chaikin Oscillator
# ==========================

def adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line (ADL)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfv = mfm * volume
    return mfv.cumsum()


def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, short: int = 3, long: int = 10) -> pd.Series:
    """Chaikin Oscillator = EMA(ADL, short) - EMA(ADL, long)."""
    line = adl(high, low, close, volume)
    return ema(line, short) - ema(line, long)


# ==========================
# 30) Elder Ray Index (Bull/Bear Power)
# ==========================

def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> pd.DataFrame:
    """Elder Ray: Bull Power e Bear Power."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    base_ema = ema(close, period)
    bull = high - base_ema
    bear = low - base_ema
    return pd.DataFrame({"elder_bull": bull, "elder_bear": bear})


# ==========================
# 31) Force Index
# ==========================

def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.DataFrame:
    """Force Index bruto e suavizado por EMA(period)."""
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    fi_raw = close.diff() * volume
    fi_ema = ema(fi_raw.fillna(0.0), period)
    return pd.DataFrame({"force_index": fi_raw, "force_index_ema": fi_ema})


# ==========================
# 32) Ultimate Oscillator
# ==========================

def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Ultimate Oscillator (0..100)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()

    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return uo


# ==========================
# 33) Ease of Movement (EOM)
# ==========================

def eom(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Ease of Movement com SMA(period)."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    volume = _validate_series(volume, "volume")
    mid_move = ((high + low) / 2.0).diff()
    box_ratio = (volume / (high - low).replace(0.0, np.nan))
    raw = mid_move / box_ratio.replace(0.0, np.nan)
    return sma(raw, period)


# ==========================
# 34) Mass Index
# ==========================

def mass_index(high: pd.Series, low: pd.Series, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    """Mass Index padrão: soma de 25 períodos da razão EMA(range)/EMA(EMA(range))."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    rng = (high - low).abs()
    ema1 = ema(rng, ema_period)
    ema2 = ema(ema1, ema_period)
    ratio = ema1 / ema2.replace(0.0, np.nan)
    return ratio.rolling(sum_period).sum()


# ==========================
# 35) Qstick
# ==========================

def qstick(open_: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Qstick = SMA(close - open, period)."""
    open_ = _validate_series(open_, "open")
    close = _validate_series(close, "close")
    return sma(close - open_, period)


# ==========================
# 36) Vortex Indicator
# ==========================

def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Vortex Indicator: VI+ e VI-."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus})


# ==========================
# 37) Stochastic RSI
# ==========================

def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic RSI: stochRSI e sinal (SMA d_period)."""
    close = _validate_series(close, "close")
    rsi_val = rsi(close, rsi_period)
    min_rsi = rsi_val.rolling(stoch_period, min_periods=stoch_period).min()
    max_rsi = rsi_val.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = (rsi_val - min_rsi) / (max_rsi - min_rsi)
    signal = sma(stoch, d_period)
    return pd.DataFrame({"stoch_rsi": 100 * stoch, "stoch_rsi_signal": 100 * signal})


# ==========================
# 38) PVO - Percentage Volume Oscillator
# ==========================

def pvo(volume: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """PVO (sobre volume), sua linha de sinal e histograma."""
    volume = _validate_series(volume, "volume")
    ema_fast = ema(volume, fast)
    ema_slow = ema(volume, slow)
    pvo_line = 100 * (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)
    signal_line = ema(pvo_line, signal)
    hist = pvo_line - signal_line
    return pd.DataFrame({"pvo": pvo_line, "pvo_signal": signal_line, "pvo_hist": hist})


# ==========================
# 39) KST - Know Sure Thing
# ==========================

def kst(close: pd.Series) -> pd.DataFrame:
    """KST padrão e sinal (SMA 9)."""
    close = _validate_series(close, "close")
    # Componentes padrão (Pring):
    r1 = roc(close, 10, as_percent=False).rolling(10).mean()
    r2 = roc(close, 15, as_percent=False).rolling(10).mean()
    r3 = roc(close, 20, as_percent=False).rolling(10).mean()
    r4 = roc(close, 30, as_percent=False).rolling(15).mean()
    kst_line = 100 * (r1 * 1 + r2 * 2 + r3 * 3 + r4 * 4)
    signal = kst_line.rolling(9).mean()
    return pd.DataFrame({"kst": kst_line, "kst_signal": signal})


# ==========================
# 40) Balance of Power (BOP)
# ==========================

def bop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Balance of Power: (close - open) / (high - low)."""
    open_ = _validate_series(open_, "open")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    return (close - open_) / (high - low).replace(0.0, np.nan)


# ==========================
# 41) HMA - Hull Moving Average
# ==========================

def hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average."""
    series = _validate_series(series, "series")
    if period <= 1:
        return series.copy()
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_full = wma(series, period)
    wma_half = wma(series, half)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_p)


# ==========================
# 42) DEMA - Double Exponential Moving Average
# ==========================

def dema(series: pd.Series, period: int = 20) -> pd.Series:
    """DEMA = 2*EMA - EMA(EMA)."""
    series = _validate_series(series, "series")
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


# ==========================
# 43) TEMA - Triple Exponential Moving Average
# ==========================

def tema(series: pd.Series, period: int = 20) -> pd.Series:
    """TEMA = 3*EMA1 - 3*EMA2 + EMA3."""
    series = _validate_series(series, "series")
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


# ==========================
# 44) TRIMA - Triangular Moving Average
# ==========================

def trima(series: pd.Series, period: int = 20) -> pd.Series:
    """Triangular Moving Average via SMA dupla com janelas ajustadas."""
    series = _validate_series(series, "series")
    if period <= 1:
        return series.copy()
    p1 = (period + 1) // 2
    p2 = period - p1 + 1
    return sma(sma(series, p1), p2)


# ==========================
# 45) CMO - Chande Momentum Oscillator
# ==========================

def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """CMO = 100*(sum(ups) - sum(downs)) / (sum(ups) + sum(downs))."""
    close = _validate_series(close, "close")
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    sum_up = up.rolling(period, min_periods=period).sum()
    sum_down = down.rolling(period, min_periods=period).sum()
    return 100 * (sum_up - sum_down) / (sum_up + sum_down)


# ==========================
# 46) Fisher Transform (Ehlers)
# ==========================

def fisher_transform(series: pd.Series, period: int = 9) -> pd.Series:
    """Ehlers Fisher Transform em série normalizada por HH/LL."""
    series = _validate_series(series, "series")
    highest = series.rolling(period).max()
    lowest = series.rolling(period).min()
    x = 0.33 * 2 * ((series - lowest) / (highest - lowest) - 0.5)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f = pd.Series(index=series.index, dtype=float)
    prev_val = 0.0
    for i in range(len(series)):
        xi = x.iat[i]
        xi = np.clip(xi * 0.67 + prev_val * 0.67, -0.999, 0.999)
        fi = 0.5 * np.log((1 + xi) / (1 - xi))
        f.iat[i] = fi
        prev_val = xi
    return f


# ==========================
# 47) Coppock Curve
# ==========================

def coppock_curve(close: pd.Series, roc1: int = 14, roc2: int = 11, wma_period: int = 10) -> pd.Series:
    """Coppock = WMA(ROC(roc1, %) + ROC(roc2, %), wma_period)."""
    close = _validate_series(close, "close")
    r1 = roc(close, roc1, as_percent=True)
    r2 = roc(close, roc2, as_percent=True)
    return wma(r1 + r2, wma_period)


# ==========================
# 48) Connors RSI (CRSI)
# ==========================

def connors_rsi(close: pd.Series, rsi_period: int = 3, streak_rsi_period: int = 2, pr_period: int = 100) -> pd.Series:
    """Connors RSI = média de RSI(close,rsi_period), RSI(streak,streak_rsi_period) e PercentRank(ROC1, pr_period)."""
    close = _validate_series(close, "close")
    rsi1 = rsi(close, rsi_period)
    # streak de ganhos/perdas consecutivos
    delta = close.diff()
    streak = pd.Series(index=close.index, dtype=float)
    streak.iat[0] = 0.0
    for i in range(1, len(close)):
        if delta.iat[i] > 0:
            streak.iat[i] = max(0.0, streak.iat[i - 1]) + 1.0
        elif delta.iat[i] < 0:
            streak.iat[i] = min(0.0, streak.iat[i - 1]) - 1.0
        else:
            streak.iat[i] = 0.0
    rsi2 = rsi(streak.fillna(0.0), streak_rsi_period)
    roc1 = roc(close, 1, as_percent=False)

    def percent_rank_window(x: np.ndarray) -> float:
        last = x[-1]
        rank = (x <= last).sum() - 1  # exclui o último
        return 100 * rank / (len(x) - 1) if len(x) > 1 else np.nan

    pr = roc1.rolling(pr_period, min_periods=pr_period).apply(percent_rank_window, raw=True)
    return (rsi1 + rsi2 + pr) / 3.0


# ==========================
# 49) SMI - Stochastic Momentum Index
# ==========================

def smi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smoothing: int = 3, signal: int = 3) -> pd.DataFrame:
    """SMI com linha de sinal."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    m = (hh + ll) / 2.0
    d = close - m
    sd = ema(ema(d, smoothing), smoothing)
    sr = ema(ema(hh - ll, smoothing), smoothing)
    smi_val = 100 * (sd / (0.5 * sr).replace(0.0, np.nan))
    signal_line = sma(smi_val, signal)
    return pd.DataFrame({"smi": smi_val, "smi_signal": signal_line})


# ==========================
# 50) RVI - Relative Vigor Index (simplificado)
# ==========================

def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """RVI simplificado: SMA(close-open)/SMA(high-low)."""
    open_ = _validate_series(open_, "open")
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    num = (close - open_).rolling(period).mean()
    den = (high - low).rolling(period).mean()
    return num / den.replace(0.0, np.nan)


# ==========================
# 51) Alligator (Jaw/Teeth/Lips)
# ==========================

def alligator(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Alligator de Bill Williams usando SMMA aproximada por suavização de Wilder sobre preço mediano.
    Jaw = SMMA(13) deslocado 8; Teeth = SMMA(8) deslocado 5; Lips = SMMA(5) deslocado 3.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    median = (high + low) / 2.0
    jaw = _wilder_ema(median, 13).shift(8)
    teeth = _wilder_ema(median, 8).shift(5)
    lips = _wilder_ema(median, 5).shift(3)
    return pd.DataFrame({"alligator_jaw": jaw, "alligator_teeth": teeth, "alligator_lips": lips})


# ==========================
# 52) Gator Oscillator
# ==========================

def gator_oscillator(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Gator Oscillator: |Jaw-Teeth| e |Teeth-Lips| baseado no Alligator."""
    alli = alligator(high, low)
    upper = (alli["alligator_jaw"] - alli["alligator_teeth"]).abs()
    lower = (alli["alligator_teeth"] - alli["alligator_lips"]).abs()
    return pd.DataFrame({"gator_upper": upper, "gator_lower": lower})


# ==========================
# 53) Fractals (Bill Williams)
# ==========================

def fractals(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Fractais: pontos onde high/low central é extremo em janela de 5 barras."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    up = (high.shift(2) < high.shift(0)) & (high.shift(1) < high.shift(0)) & (high.shift(-1) < high.shift(0)) & (high.shift(-2) < high.shift(0))
    down = (low.shift(2) > low.shift(0)) & (low.shift(1) > low.shift(0)) & (low.shift(-1) > low.shift(0)) & (low.shift(-2) > low.shift(0))
    return pd.DataFrame({"fractal_up": up.fillna(False), "fractal_down": down.fillna(False)})


# ==========================
# 54) Bollinger %B
# ==========================

def bollinger_percent_b(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B = (close - lower) / (upper - lower)."""
    bands = bollinger_bands(close, period, num_std)
    upper = bands["bb_upper"]
    lower = bands["bb_lower"]
    return (close - lower) / (upper - lower)


# ==========================
# 55) Bollinger Bandwidth
# ==========================

def bollinger_bandwidth(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bandwidth = (upper - lower) / middle."""
    bands = bollinger_bands(close, period, num_std)
    return (bands["bb_upper"] - bands["bb_lower"]) / bands["bb_mid"].replace(0.0, np.nan)


# ==========================
# 56) Chandelier Exit (long/short)
# ==========================

def chandelier_exit(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22, multiplier: float = 3.0) -> pd.DataFrame:
    """Chandelier Exit longo e curto."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    atr_val = atr(high, low, close, 22)
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    long_exit = highest - multiplier * atr_val
    short_exit = lowest + multiplier * atr_val
    return pd.DataFrame({"chandelier_long": long_exit, "chandelier_short": short_exit})


# ==========================
# 57) KVO - Klinger Volume Oscillator (simplificado)
# ==========================

def kvo(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.DataFrame:
    """Klinger Volume Oscillator: EMA(VF, fast) - EMA(VF, slow) e sinal."""
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    tp = (high + low + close)
    dm = tp.diff()
    trend = np.where(dm > 0, 1, np.where(dm < 0, -1, 0))
    vf = trend * volume * (2 * (dm.abs() / (tp.shift(1) + tp)) - 1)
    vf = pd.Series(vf, index=close.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    line = ema(vf, fast) - ema(vf, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"kvo": line, "kvo_signal": sig})


# ==========================
# 58) Moving Average Envelopes
# ==========================

def ma_envelopes(series: pd.Series, period: int = 20, percent: float = 0.025) -> pd.DataFrame:
    """Envelopes em torno de SMA: upper/lower = SMA*(1±percent)."""
    series = _validate_series(series, "series")
    mid = sma(series, period)
    upper = mid * (1 + percent)
    lower = mid * (1 - percent)
    return pd.DataFrame({"ma_env_mid": mid, "ma_env_upper": upper, "ma_env_lower": lower})


# ==========================
# 59) Ulcer Index
# ==========================

def ulcer_index(series: pd.Series, period: int = 14) -> pd.Series:
    """Ulcer Index baseado nos drawdowns percentuais."""
    series = _validate_series(series, "series")
    rolling_max = series.rolling(period, min_periods=period).max()
    drawdown = 100 * (series - rolling_max) / rolling_max
    return np.sqrt((drawdown.pow(2)).rolling(period, min_periods=period).mean())


# ==========================
# 60) Z-Score do preço
# ==========================

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score = (series - SMA) / STD."""
    series = _validate_series(series, "series")
    mean = sma(series, period)
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return (series - mean) / std