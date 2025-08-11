import unittest
import os, sys
sys.path.insert(0, "/workspace")

from roulette_indicators.synthetic import generate_european_spins
from roulette_indicators.ml import train_and_evaluate

class TestRouletteML(unittest.TestCase):
    def test_train_evaluate_runs(self):
        spins = generate_european_spins(4000, seed=11)
        out = train_and_evaluate(spins, task="color", n_lags=10, windows=[10, 20], lr=0.05, epochs=150, reg=1e-3)
        self.assertIn("val_accuracy", out)
        # Em dados IID, deve ficar próximo do baseline 18/37
        baseline = 18/37
        self.assertTrue(abs(out["val_accuracy"] - baseline) < 0.1)

if __name__ == "__main__":
    unittest.main()