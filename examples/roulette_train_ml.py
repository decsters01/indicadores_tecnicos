import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from roulette_indicators.synthetic import generate_european_spins
from roulette_indicators.ml import train_and_evaluate

spins = generate_european_spins(20000, seed=2025)

results = {}
for task in ["color", "highlow", "dozen", "column"]:
    out = train_and_evaluate(spins, task=task, n_lags=20, windows=[20, 50, 100], lr=0.05, epochs=300, reg=1e-3)
    results[task] = out
    print(f"Task={task} val_acc={out['val_accuracy']:.4f} n_train={out['n_train']} n_val={out['n_val']}")

out_dir = "/workspace/examples/outputs"
os.makedirs(out_dir, exist_ok=True)

# Salvar modelos
for task, out in results.items():
    path = os.path.join(out_dir, f"ml_{task}_logreg_ovr.json")
    out["model"].save(path, out["feature_names"])
    print(f"Salvo: {path}")