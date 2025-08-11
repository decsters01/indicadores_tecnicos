import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roulette_indicators.synthetic import generate_european_spins
from roulette_indicators.tuning import grid_search
from roulette_indicators.feature_sets import build_features_with_fx

spins = generate_european_spins(30000, seed=77)

grid = {
    "n_lags": [10, 20],
    "windows": [[20, 50], [20, 50, 100]],
    "lr": [0.05, 0.02],
    "epochs": [200, 400],
    "reg": [1e-3, 5e-4],
}

for task in ["color", "highlow", "dozen", "column"]:
    df = grid_search(spins, task, grid, feature_builder=build_features_with_fx, max_models=6)
    print(f"Task {task} - Top resultados:\n", df.head(3))