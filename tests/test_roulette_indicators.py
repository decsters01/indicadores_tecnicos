import unittest
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, "/workspace")

from roulette_indicators import (
    encode_spin_european,
    spins_to_colors,
    spins_to_highlow,
    spins_to_dozens,
    spins_to_columns,
    rolling_category_frequencies,
    predict_next_by_rolling_frequency,
    generate_european_spins,
    evaluate_task_accuracy,
    evaluate_all_tasks_summary,
)


class TestRouletteIndicators(unittest.TestCase):
    def setUp(self):
        self.n = 5000
        self.spins = generate_european_spins(self.n, seed=123)

    def test_encoding(self):
        e0 = encode_spin_european(0)
        self.assertEqual(e0["color"], "Z")
        self.assertEqual(e0["highlow"], "Z")
        self.assertEqual(e0["dozen"], "Z")
        self.assertEqual(e0["column"], "Z")

        e7 = encode_spin_european(7)
        self.assertIn(e7["color"], ["R", "B"])  # 7 é vermelho
        self.assertEqual(e7["color"], "R")
        self.assertEqual(e7["highlow"], "L")
        self.assertIn(e7["dozen"], ["D1"])        # 1..12
        self.assertIn(e7["column"], ["C1"])       # coluna 1

    def test_series_encoders(self):
        colors = spins_to_colors(self.spins)
        self.assertEqual(len(colors), self.n)
        hl = spins_to_highlow(self.spins)
        self.assertEqual(len(hl), self.n)
        dz = spins_to_dozens(self.spins)
        self.assertEqual(len(dz), self.n)
        cols = spins_to_columns(self.spins)
        self.assertEqual(len(cols), self.n)

    def test_rolling_freq_and_prediction(self):
        freqs = rolling_category_frequencies(self.spins, task="color", window=50)
        self.assertTrue(set(["R","B"]).issubset(set(freqs.columns)))
        preds = predict_next_by_rolling_frequency(self.spins, task="color", window=50)
        self.assertEqual(len(preds), self.n)

    def test_evaluation_accuracy_levels(self):
        # Em dados IID, esperamos resultados próximos a probabilidades-base
        # Cores e High/Low ~ 18/37 ~ 0.486; Dúzias/Colunas ~ 12/37 ~ 0.324
        tol = 0.03
        for task, expected in [("color", 18/37), ("highlow", 18/37), ("dozen", 12/37), ("column", 12/37)]:
            r = evaluate_task_accuracy(self.spins, task, window=50)
            self.assertTrue(abs(r.accuracy - expected) < tol, msg=f"{task} acc={r.accuracy} ~ {expected}")

        df = evaluate_all_tasks_summary(self.spins, windows=[20, 50, 100])
        self.assertEqual(sorted(df["task"].unique().tolist()), ["color","column","dozen","highlow"])


if __name__ == "__main__":
    unittest.main()