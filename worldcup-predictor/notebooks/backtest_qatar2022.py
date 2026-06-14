"""Backtest contra Qatar 2022: entrena SOLO con partidos anteriores al
20-nov-2022 y predice los 64 partidos del Mundial en orden cronológico
(el estado Elo/forma se actualiza con cada resultado real ya jugado, como
ocurriría prediciendo en vivo).

Compara Poisson puro vs Dixon-Coles, con foco en los empates 0-0 y 1-1,
y guarda la curva de calibración en reports/calibration_wc2022.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import add_basic_features, filter_since, load_results
from features import FeatureBuilder
from model import (FEATURE_COLS, _make_regressor, calibration_table,
                   estimate_rho, make_xy, outcome_probs, plot_calibration,
                   poisson_matrix, probs_metrics, REPORTS_DIR)

WC_START = pd.Timestamp("2022-11-20")

full = add_basic_features(filter_since(load_results(), 1990))
wc22 = full[(full["tournament_type"] == "world_cup")
            & (full["date"].dt.year == 2022)].sort_values("date")
pre = full[full["date"] < WC_START]
print(f"histórico pre-Mundial: {len(pre)} | partidos Qatar 2022: {len(wc22)}")

builder = FeatureBuilder()
builder.warm_up(pre[pre["date"] < "2010-01-01"])
hist = builder.transform(pre[pre["date"] >= "2010-01-01"].reset_index(drop=True))
# segundo transform: continúa desde el estado pre-Mundial y genera las
# features de cada partido del torneo solo con lo jugado hasta ese día
feat_wc = builder.transform(wc22.reset_index(drop=True))

cats = pd.concat([hist, feat_wc])["match_type"].astype("category").cat.categories
train = hist[hist["year"] <= 2021]
es_eval = hist[hist["year"] == 2022]  # 2022 pre-Mundial, para early stopping


def _xy(d):
    X, yh, ya = make_xy(d)
    X["match_type"] = X["match_type"].cat.set_categories(cats)
    return X, yh, ya


X_tr, yh_tr, ya_tr = _xy(train)
X_es, yh_es, ya_es = _xy(es_eval)
models = {"feature_cols": FEATURE_COLS}
for tag, y_tr, y_es in [("home", yh_tr, yh_es), ("away", ya_tr, ya_es)]:
    reg = _make_regressor()
    reg.fit(X_tr, y_tr, eval_set=[(X_es, {"home": yh_es, "away": ya_es}[tag])],
            verbose=False)
    models[f"model_{tag}"] = reg

rho = estimate_rho(models, train, cats)
print(f"entrenado con 2010-2021 ({len(train)} partidos), "
      f"early stopping con 2022 pre-Mundial ({len(es_eval)}), rho={rho:+.4f}")

X_wc, yh_wc, ya_wc = _xy(feat_wc)
lam_h = models["model_home"].predict(X_wc)
lam_a = models["model_away"].predict(X_wc)
y = np.where(yh_wc.values > ya_wc.values, 0,
             np.where(yh_wc.values == ya_wc.values, 1, 2))

results = {}
for tag, r in [("poisson", 0.0), ("dixon-coles", rho)]:
    probs = np.array([
        [op["home_win"], op["draw"], op["away_win"]]
        for op in (outcome_probs(h, a, rho=r) for h, a in zip(lam_h, lam_a))
    ])
    probs /= probs.sum(axis=1, keepdims=True)
    results[tag] = probs
    met = probs_metrics(probs, y)
    # masa esperada en los marcadores de empate bajos vs realidad
    e00 = sum(float(poisson_matrix(h, a, rho=r)[0, 0]) for h, a in zip(lam_h, lam_a))
    e11 = sum(float(poisson_matrix(h, a, rho=r)[1, 1]) for h, a in zip(lam_h, lam_a))
    print(f"\n[{tag}]  log-loss={met['log_loss']:.4f}  "
          f"brier={met['brier_score']:.4f}  acc 1X2={met['accuracy_1x2']:.4f}")
    print(f"  empates esperados: {probs[:, 1].sum():.1f}  |  "
          f"0-0 esperados: {e00:.1f}  |  1-1 esperados: {e11:.1f}")

a00 = int(((yh_wc == 0) & (ya_wc == 0)).sum())
a11 = int(((yh_wc == 1) & (ya_wc == 1)).sum())
print(f"\nrealidad: {int((y == 1).sum())} empates (90'), "
      f"{a00} fueron 0-0 y {a11} fueron 1-1")

table = calibration_table(results["dixon-coles"], np.eye(3)[y], n_bins=8)
png = REPORTS_DIR / "calibration_wc2022.png"
plot_calibration(table, png)
print(f"\nCalibración DC (8 bins, 64 partidos x 3 clases) -> {png}")
print(table.to_string(index=False))
