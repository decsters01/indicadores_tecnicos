from __future__ import annotations

import numpy as np
import pandas as pd


def spins_to_price(spins: pd.Series, method: str = "raw") -> pd.Series:
    if not isinstance(spins, pd.Series):
        spins = pd.Series(spins)
    s = spins.astype(float)
    if method == "raw":
        return s
    elif method == "cumsum_demeaned":
        # Remove média para introduzir uma série com tendência/oscilações
        mean_val = s[s != 0].mean() if (s != 0).any() else s.mean()
        return (s - mean_val).fillna(0.0).cumsum()
    else:
        raise ValueError("method deve ser 'raw' ou 'cumsum_demeaned'")


def price_to_ohlcv(price: pd.Series) -> pd.DataFrame:
    if not isinstance(price, pd.Series):
        price = pd.Series(price, dtype=float)
    price = price.astype(float)
    open_ = price.shift(1)
    close = price
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    volume = pd.Series(1.0, index=price.index)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df