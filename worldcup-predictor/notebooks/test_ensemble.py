"""Ensemble XGBoost + Dixon-Coles clásico.

- DC clásico: ataque/defensa por equipo, desde 2018, decaimiento temporal
  (se elige half-life 2/3/4 por log-loss de marcador en validación).
- XGBoost: configuración baseline actual (entrena 2010-2023).
- Ensemble: matriz_final = w*ML + (1-w)*DC, w optimizado en validación
  2024-2025 por log-loss de marcador exacto.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dixon_coles import DixonColesModel
from model import (TRAIN_END, TRAIN_START, VAL_END, VAL_START, estimate_rho,
                   make_xy, prepare_data, score_matrix_agg, train_models)

CUTOFF = pd.Timestamp(f"{VAL_START}-01-01")

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
val = df[df["year"].between(VAL_START, VAL_END)].reset_index(drop=True)
cats = df["match_type"].astype("category").cat.categories
print(f"train {len(train)} | val {len(val)}")

yh, ya = val["home_score"].values, val["away_score"].values
ih = np.minimum(yh.astype(int), 6)
ij = np.minimum(ya.astype(int), 6)


def mat_ll(mats: np.ndarray) -> float:
    p = mats[np.arange(len(mats)), ih, ij]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def outcome_metrics(mats: np.ndarray) -> tuple[float, float]:
    tril = np.tril(np.ones((7, 7)), -1)
    triu = np.triu(np.ones((7, 7)), 1)
    eye = np.eye(7)
    probs = np.stack([(mats * m).sum(axis=(1, 2)) for m in (tril, eye, triu)], axis=1)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.where(yh > ya, 0, np.where(yh == ya, 1, 2))
    ll = float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).mean())
    acc = float((probs.argmax(axis=1) == y).mean())
    return ll, acc


# --- XGBoost baseline ---
print("\nentrenando XGBoost baseline...")
models = train_models(df)
rho_ml = estimate_rho(models, train, cats)
X_val, _, _ = make_xy(val)
X_val["match_type"] = X_val["match_type"].cat.set_categories(cats)
lam_h = np.clip(models["model_home"].predict(X_val), 0.05, None)
lam_a = np.clip(models["model_away"].predict(X_val), 0.05, None)
M_ml = np.stack([score_matrix_agg(h, a, rho=rho_ml)
                 for h, a in zip(lam_h, lam_a)])
print(f"  XGB: ll marcador = {mat_ll(M_ml):.4f}")

# --- DC clásico: elegir half-life ---
print("\nDixon-Coles clásico (desde 2018, cutoff 2024):")
best_hl, best_dc, best_ll = None, None, np.inf
for hl in (2.0, 3.0, 4.0):
    dc = DixonColesModel(since=2018, half_life_years=hl).fit(df, cutoff=CUTOFF)
    M = np.stack([
        dc.score_matrix(r.home_team, r.away_team, bool(r.true_home_advantage))
        for r in val.itertuples()
    ])
    ll = mat_ll(M)
    print(f"  half-life {hl:.0f}a: ll={ll:.4f} (gamma={dc.gamma:.3f}, "
          f"rho={dc.rho:+.4f}, {dc.n_matches} partidos, "
          f"convergió={dc.converged})")
    if ll < best_ll:
        best_hl, best_dc, best_ll, M_dc = hl, dc, ll, M

print(f"  -> half-life elegido: {best_hl:.0f} años")
top = sorted(best_dc.attack.items(), key=lambda kv: kv[1], reverse=True)[:5]
print("  top ataques DC:", ", ".join(f"{t} {v:+.2f}" for t, v in top))

# --- ensemble ---
print(f"\n{'w (peso ML)':>12}{'ll marcador':>13}{'ll 1X2':>9}{'acc':>8}")
results = []
for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
    M = w * M_ml + (1 - w) * M_dc
    ll = mat_ll(M)
    ll1x2, acc = outcome_metrics(M)
    results.append((w, ll, ll1x2, acc))
    if w in (0.0, 0.5, 1.0):
        print(f"{w:>12.2f}{ll:>13.4f}{ll1x2:>9.4f}{acc:>8.4f}")

w_best, ll_best, ll1x2_best, acc_best = min(results, key=lambda t: t[1])
print(f"\nmejor w = {w_best:.2f}")
print(f"  ll marcador: XGB={mat_ll(M_ml):.4f} | DC={best_ll:.4f} | "
      f"ensemble={ll_best:.4f}")
print(f"  ll 1X2 ensemble={ll1x2_best:.4f}, acc={acc_best:.4f}")
mejora_xgb = mat_ll(M_ml) - ll_best
mejora_dc = best_ll - ll_best
print(f"  supera a XGB por {mejora_xgb:+.4f} y a DC por {mejora_dc:+.4f} "
      f"-> {'SI supera a ambos' if mejora_xgb > 0 and mejora_dc > 0 else 'NO supera a ambos'}")
