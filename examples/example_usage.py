import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from forex_indicators import (
    sma, ema, wma, macd, rsi, stochastic, bollinger_bands, atr, adx, cci, obv, mfi,
    parabolic_sar, ichimoku, pivot_points, vwap, roc, williams_r, trix, donchian_channels,
    normalize_ohlcv,
)

np.random.seed(42)

n = 300
index = pd.date_range("2023-01-01", periods=n, freq="D")
price = 1.10 + np.cumsum(np.random.normal(0, 0.001, size=n))
high = price + np.random.uniform(0.0005, 0.0020, size=n)
low = price - np.random.uniform(0.0005, 0.0020, size=n)
open_ = price + np.random.uniform(-0.001, 0.001, size=n)
close = price + np.random.uniform(-0.001, 0.001, size=n)
volume = np.random.randint(1000, 10000, size=n).astype(float)

df = pd.DataFrame({
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
}, index=index)

# Exemplo: se suas colunas vierem com nomes variados (ex.: 'Close', 'Adj Close', 'VOL'),
# você pode normalizar para open/high/low/close/volume com:
# df = normalize_ohlcv(df)

# Médias móveis
df["sma_20"] = sma(df["close"], 20)
df["ema_20"] = ema(df["close"], 20)
df["wma_20"] = wma(df["close"], 20)

# MACD
macd_df = macd(df["close"]) 
df = pd.concat([df, macd_df], axis=1)

# RSI
df["rsi_14"] = rsi(df["close"]) 

# Estocástico
stoch_df = stochastic(df["high"], df["low"], df["close"]) 
df = pd.concat([df, stoch_df], axis=1)

# Bandas de Bollinger
bb_df = bollinger_bands(df["close"]) 
df = pd.concat([df, bb_df], axis=1)

# ATR
df["atr_14"] = atr(df["high"], df["low"], df["close"]) 

# ADX
adx_df = adx(df["high"], df["low"], df["close"]) 
df = pd.concat([df, adx_df], axis=1)

# CCI
df["cci_20"] = cci(df["high"], df["low"], df["close"]) 

# OBV
df["obv"] = obv(df["close"], df["volume"]) 

# MFI
df["mfi_14"] = mfi(df["high"], df["low"], df["close"], df["volume"]) 

# Parabolic SAR
df["psar"] = parabolic_sar(df["high"], df["low"]) 

# Ichimoku
ichi_df = ichimoku(df["high"], df["low"], df["close"]) 
df = pd.concat([df, ichi_df], axis=1)

# Pivot Points
pp_df = pivot_points(df["high"], df["low"], df["close"]) 
df = pd.concat([df, pp_df], axis=1)

# VWAP
df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"]) 

# ROC
df["roc_12"] = roc(df["close"], 12)

# Williams %R
df["willr_14"] = williams_r(df["high"], df["low"], df["close"]) 

# TRIX
df["trix_15"] = trix(df["close"]) 

# Donchian
don_df = donchian_channels(df["high"], df["low"]) 
df = pd.concat([df, don_df], axis=1)

print(df.tail(5))

# Salva resultado
df.to_csv("/workspace/examples/outputs/indicators_output.csv")
print("Arquivo salvo em /workspace/examples/outputs/indicators_output.csv")