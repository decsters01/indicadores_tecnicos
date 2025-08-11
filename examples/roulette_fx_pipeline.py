import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from roulette_indicators.synthetic import generate_european_spins
from roulette_indicators.fx_predict import evaluate_indicators_on_tasks

spins = generate_european_spins(30000, seed=123)
results = evaluate_indicators_on_tasks(spins, window=200, bins=10)
print(results)

out_dir = "/workspace/examples/outputs"
os.makedirs(out_dir, exist_ok=True)
results.to_csv(os.path.join(out_dir, "roulette_fx_results.csv"), index=False)
print(f"Arquivo salvo em {out_dir}/roulette_fx_results.csv")

# Mostrar top-3 por tarefa
for task in results.task.unique():
    top = results[results.task == task].sort_values("accuracy", ascending=False).head(3)
    print(f"Task {task}:\n{top[['indicator','accuracy']].to_string(index=False)}\n")