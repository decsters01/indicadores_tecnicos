from __future__ import annotations

import numpy as np
import pandas as pd

# Gera spins IID uniformes para roleta europeia (0..36)

def generate_european_spins(n: int, seed: int | None = None, index: pd.Index | None = None) -> pd.Series:
    if n <= 0:
        raise ValueError("n deve ser > 0")
    rng = np.random.default_rng(seed)
    spins = rng.integers(low=0, high=37, size=n)
    return pd.Series(spins, index=index)