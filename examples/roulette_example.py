import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from roulette_indicators import generate_european_spins, evaluate_all_tasks_summary

n = 20000
spins = generate_european_spins(n, seed=42)
results = evaluate_all_tasks_summary(spins, windows=[20, 50, 100])
print(results)

out_dir = "/workspace/examples/outputs"
os.makedirs(out_dir, exist_ok=True)
results.to_csv(os.path.join(out_dir, "roulette_performance.csv"), index=False)
print(f"Arquivo salvo em {out_dir}/roulette_performance.csv")