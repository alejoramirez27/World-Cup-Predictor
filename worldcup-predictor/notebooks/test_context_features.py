"""Prueba de las features de contexto (altitud, descanso, knockout, viaje)
contra el baseline, con la batería completa de métricas (1X2 + marcador).
También chequeos de sanidad de las features nuevas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,
                   evaluate, prepare_data, train_models)

CONTEXT = ["home_rest_days", "away_rest_days", "home_alt_diff",
           "away_alt_diff", "home_travel_km", "away_travel_km", "is_knockout"]

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

# --- sanidad de las features nuevas ---
print("sanidad:")
print(f"  rest_days   mediana={df['home_rest_days'].median():.0f}  "
      f"NaN={df['home_rest_days'].isna().mean():.1%}")
print(f"  travel_km   mediana={df['home_travel_km'].median():.0f}  "
      f"NaN={df['home_travel_km'].isna().mean():.1%}")
print(f"  alt_diff    p95={df['home_alt_diff'].quantile(0.95):.0f}  "
      f"max={df['home_alt_diff'].max():.0f}")
ko = df[df["is_knockout"]]
print(f"  knockout    {len(ko)} partidos marcados | goles/partido: "
      f"{(ko['home_score'] + ko['away_score']).mean():.2f} vs "
      f"{(df.loc[~df['is_knockout'] & df['match_type'].isin(['world_cup', 'continental']), ['home_score', 'away_score']].sum(axis=1)).mean():.2f} en grupos")
boliv = df[(df["home_team"] == "Bolivia") & (df["match_type"] == "wc_qualifier")]
print(f"  Bolivia de local en eliminatorias: alt_diff promedio rival = "
      f"{boliv['away_alt_diff'].mean():.0f} m ({len(boliv)} partidos)")

# --- comparación con la prueba completa ---
variants = {"baseline": FEATURE_COLS, "contexto": FEATURE_COLS + CONTEXT}
KEYS = [("log_loss", "log-loss 1X2"), ("brier_score", "brier"),
        ("rps", "RPS"), ("accuracy_1x2", "acc 1X2"),
        ("scoreline_log_loss", "ll marcador"), ("top1_scoreline", "top-1"),
        ("top3_scoreline", "top-3")]

evs = {}
for tag, cols in variants.items():
    models = train_models(df, feature_cols=cols)
    rho = estimate_rho(models, train, cats)
    evs[tag] = evaluate(models, df, rho=rho)

print(f"\n{'métrica':<16}{'baseline':>12}{'contexto':>12}{'delta':>10}")
for key, label in KEYS:
    b, c = evs["baseline"][key], evs["contexto"][key]
    print(f"{label:<16}{b:>12.4f}{c:>12.4f}{c - b:>+10.4f}")
