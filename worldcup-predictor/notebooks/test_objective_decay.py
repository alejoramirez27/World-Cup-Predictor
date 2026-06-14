"""Prueba del objetivo y del decaimiento temporal:
  - squared error vs count:poisson (este último es el default desde el
    inicio; aquí se mide cuánto aporta realmente)
  - half-life de 2, 4 y 6 años vs sin decaimiento
Métricas habituales sobre validación 2024-2025, con Dixon-Coles."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (TRAIN_END, TRAIN_START, estimate_rho, evaluate,
                   prepare_data, train_models)

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

variants = {
    "sqerr":        {"objective": "reg:squarederror"},
    "poisson":      {},
    "poisson_hl2":  {"half_life_years": 2.0},
    "poisson_hl4":  {"half_life_years": 4.0},
    "poisson_hl6":  {"half_life_years": 6.0},
}

print(f"{'variante':<14}{'log-loss':>10}{'brier':>10}{'acc 1X2':>10}"
      f"{'mae_h':>8}{'mae_a':>8}{'rho':>9}")
for tag, kwargs in variants.items():
    models = train_models(df, **kwargs)
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    m = models["metrics"]
    print(f"{tag:<14}{ev['log_loss']:>10.4f}{ev['brier_score']:>10.4f}"
          f"{ev['accuracy_1x2']:>10.4f}{m['mae_home']:>8.4f}"
          f"{m['mae_away']:>8.4f}{rho:>+9.4f}")
