import unittest
import numpy as np
import pandas as pd

import os
import sys

# Garantir import do pacote local
sys.path.insert(0, "/workspace")
from forex_indicators import *  # noqa: F401,F403


class TestIndicators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        n = 300
        cls.index = pd.date_range("2023-01-01", periods=n, freq="D")
        price = 1.10 + np.cumsum(np.random.normal(0, 0.001, size=n))
        cls.high = pd.Series(price + np.random.uniform(0.0005, 0.0020, size=n), index=cls.index)
        cls.low = pd.Series(price - np.random.uniform(0.0005, 0.0020, size=n), index=cls.index)
        cls.open = pd.Series(price + np.random.uniform(-0.001, 0.001, size=n), index=cls.index)
        cls.close = pd.Series(price + np.random.uniform(-0.001, 0.001, size=n), index=cls.index)
        cls.volume = pd.Series(np.random.randint(1000, 10000, size=n).astype(float), index=cls.index)

    def assertSeriesLike(self, s: pd.Series):
        self.assertIsInstance(s, pd.Series)
        self.assertEqual(len(s), len(self.index))

    def assertFrameLike(self, df: pd.DataFrame, cols):
        self.assertIsInstance(df, pd.DataFrame)
        for c in cols:
            self.assertIn(c, df.columns)
        self.assertEqual(len(df), len(self.index))

    def test_mas_basicas(self):
        self.assertSeriesLike(sma(self.close, 20))
        self.assertSeriesLike(ema(self.close, 20))
        self.assertSeriesLike(wma(self.close, 20))

    def test_macd_rsi(self):
        macd_df = macd(self.close)
        self.assertFrameLike(macd_df, ["macd", "signal", "hist"])
        self.assertSeriesLike(rsi(self.close))

    def test_stoch_bb_atr_adx(self):
        self.assertFrameLike(stochastic(self.high, self.low, self.close), ["stoch_k", "stoch_d"])
        self.assertFrameLike(bollinger_bands(self.close), ["bb_mid", "bb_upper", "bb_lower"])
        self.assertSeriesLike(atr(self.high, self.low, self.close))
        self.assertFrameLike(adx(self.high, self.low, self.close), ["plus_di", "minus_di", "adx"])

    def test_vol_price(self):
        self.assertSeriesLike(cci(self.high, self.low, self.close))
        self.assertSeriesLike(obv(self.close, self.volume))
        self.assertSeriesLike(mfi(self.high, self.low, self.close, self.volume))

    def test_misc_1(self):
        self.assertSeriesLike(parabolic_sar(self.high, self.low))
        self.assertFrameLike(ichimoku(self.high, self.low, self.close), [
            "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou"
        ])
        self.assertFrameLike(pivot_points(self.high, self.low, self.close), ["pp", "r1", "s1", "r2", "s2", "r3", "s3"])
        self.assertSeriesLike(vwap(self.high, self.low, self.close, self.volume))
        self.assertSeriesLike(roc(self.close))
        self.assertSeriesLike(williams_r(self.high, self.low, self.close))
        self.assertSeriesLike(trix(self.close))
        self.assertFrameLike(donchian_channels(self.high, self.low), ["donchian_upper", "donchian_lower", "donchian_mid"])

    def test_new_20(self):
        self.assertFrameLike(keltner_channels(self.high, self.low, self.close), ["kc_mid", "kc_upper", "kc_lower"])
        st = supertrend(self.high, self.low, self.close)
        self.assertFrameLike(st, ["supertrend", "supertrend_trend", "supertrend_upper", "supertrend_lower"])
        self.assertFrameLike(ppo(self.close), ["ppo", "ppo_signal", "ppo_hist"])
        self.assertSeriesLike(kama(self.close))
        self.assertSeriesLike(tsi(self.close))
        self.assertSeriesLike(dpo(self.close))
        self.assertFrameLike(aroon(self.high, self.low), ["aroon_up", "aroon_down", "aroon_osc"])
        self.assertSeriesLike(cmf(self.high, self.low, self.close, self.volume))
        self.assertSeriesLike(adl(self.high, self.low, self.close, self.volume))
        self.assertSeriesLike(chaikin_oscillator(self.high, self.low, self.close, self.volume))
        self.assertFrameLike(elder_ray(self.high, self.low, self.close), ["elder_bull", "elder_bear"])
        self.assertFrameLike(force_index(self.close, self.volume), ["force_index", "force_index_ema"])
        self.assertSeriesLike(ultimate_oscillator(self.high, self.low, self.close))
        self.assertSeriesLike(eom(self.high, self.low, self.volume))
        self.assertSeriesLike(mass_index(self.high, self.low))
        self.assertSeriesLike(qstick(self.open, self.close))
        self.assertFrameLike(vortex(self.high, self.low, self.close), ["vi_plus", "vi_minus"])
        self.assertFrameLike(stoch_rsi(self.close), ["stoch_rsi", "stoch_rsi_signal"])
        self.assertFrameLike(pvo(self.volume), ["pvo", "pvo_signal", "pvo_hist"])
        self.assertFrameLike(kst(self.close), ["kst", "kst_signal"])
        self.assertSeriesLike(bop(self.open, self.high, self.low, self.close))

    def test_more_20(self):
        self.assertSeriesLike(hma(self.close))
        self.assertSeriesLike(dema(self.close))
        self.assertSeriesLike(tema(self.close))
        self.assertSeriesLike(trima(self.close))
        self.assertSeriesLike(cmo(self.close))
        self.assertSeriesLike(fisher_transform(self.close))
        self.assertSeriesLike(coppock_curve(self.close))
        self.assertSeriesLike(connors_rsi(self.close))
        self.assertFrameLike(smi(self.high, self.low, self.close), ["smi", "smi_signal"])
        self.assertSeriesLike(rvi(self.open, self.high, self.low, self.close))
        self.assertFrameLike(alligator(self.high, self.low), ["alligator_jaw", "alligator_teeth", "alligator_lips"])
        self.assertFrameLike(gator_oscillator(self.high, self.low), ["gator_upper", "gator_lower"])
        self.assertFrameLike(fractals(self.high, self.low), ["fractal_up", "fractal_down"])
        self.assertSeriesLike(bollinger_percent_b(self.close))
        self.assertSeriesLike(bollinger_bandwidth(self.close))
        self.assertFrameLike(chandelier_exit(self.high, self.low, self.close), ["chandelier_long", "chandelier_short"])
        self.assertFrameLike(kvo(self.high, self.low, self.close, self.volume), ["kvo", "kvo_signal"])
        self.assertFrameLike(ma_envelopes(self.close), ["ma_env_mid", "ma_env_upper", "ma_env_lower"])
        self.assertSeriesLike(ulcer_index(self.close))
        self.assertSeriesLike(zscore(self.close))


if __name__ == "__main__":
    unittest.main()