import unittest
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "/workspace")

from roulette_indicators.synthetic import generate_european_spins
from roulette_indicators.fx_predict import evaluate_indicators_on_tasks, INDICATORS


class TestRouletteFxPredict(unittest.TestCase):
    def test_run_and_baselines(self):
        spins = generate_european_spins(5000, seed=7)
        df = evaluate_indicators_on_tasks(spins, window=200, bins=8)
        self.assertFalse(df.empty)
        # Checar que contém todas tasks
        self.assertEqual(sorted(df["task"].unique().tolist()), ["color","column","dozen","highlow"])
        # Em dados IID, acurácia deve ficar próxima das probabilidades base
        base = {"color": 18/37, "highlow": 18/37, "dozen": 12/37, "column": 12/37}
        tol = 0.06
        for task, expected in base.items():
            accs = df[df.task == task].accuracy.dropna()
            self.assertTrue(len(accs) > 0)
            # média por task perto do baseline
            self.assertTrue(abs(accs.mean() - expected) < tol)


if __name__ == "__main__":
    unittest.main()