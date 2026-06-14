"""Entrena y CONGELA el modelo final para el Mundial 2026 (ensemble).

Pipeline:
 1. Con los hiperparámetros del tuning (notebooks/optuna_best_params.json),
    entrena XGB en 2010-2023 y un Dixon-Coles clásico con cutoff 2024, y
    re-optimiza el peso w del ensemble en la validación 2024-2025
    (matriz_final = w*ML + (1-w)*DC, log-loss de marcador exacto).
 2. Reentrena AMBOS modelos con todo el dataset jugado hasta la fecha y
    congela: models/model_final_wc2026/{xgb_home,xgb_away}.json + dc.json
    + meta.json (parámetros, rho, w, corte de datos, política).

Política de torneo: una vez congelado, update_data.py no reentrena (solo
--force-retrain lo rompe); las features se recalculan en cada corrida.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from dixon_coles import DixonColesModel
from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, VAL_END, VAL_START,
                   estimate_rho, make_xy, prepare_data, score_matrix_agg)

FINAL_DIR = Path(__file__).resolve().parent.parent / "models" / "model_final_wc2026"
PARAMS_JSON = (Path(__file__).resolve().parent.parent
               / "notebooks" / "optuna_best_params.json")
DC_SINCE, DC_HALF_LIFE = 2018, 4.0  # elegidos en notebooks/test_ensemble.py


def _fit_xgb(params: dict, n_estimators: int, X, y) -> XGBRegressor:
    reg = XGBRegressor(objective="count:poisson", n_estimators=n_estimators,
                       tree_method="hist", enable_categorical=True, **params)
    reg.fit(X, y, verbose=False)
    return reg


def main() -> None:
    if FINAL_DIR.exists() and "--overwrite" not in sys.argv:
        print(f"{FINAL_DIR} ya existe (modelo congelado). "
              f"Usa --overwrite si de verdad quieres regenerarlo.")
        return

    tuning = json.loads(PARAMS_JSON.read_text())
    params = tuning["best_params"]
    n_estimators = tuning["n_estimators_final"]
    print(f"hiperparámetros del tuning: {params}")
    print(f"n_estimators fijo: {n_estimators}")

    print(f"\nPreparando datos (warm-up Elo 1990, features desde {TRAIN_START})...")
    df, builder = prepare_data()
    cats = df["match_type"].astype("category").cat.categories

    # ---- paso 1: optimizar w en validación con los parámetros tuneados ----
    train = df[df["year"].between(TRAIN_START, TRAIN_END)]
    val = df[df["year"].between(VAL_START, VAL_END)].reset_index(drop=True)
    X_tr, yh_tr, ya_tr = make_xy(train)
    X_va, yh_va, ya_va = make_xy(val)
    X_tr["match_type"] = X_tr["match_type"].cat.set_categories(cats)
    X_va["match_type"] = X_va["match_type"].cat.set_categories(cats)

    print(f"entrenando XGB tuneado ({TRAIN_START}-{TRAIN_END}) para optimizar w...")
    mh = _fit_xgb(params, n_estimators, X_tr, yh_tr)
    ma = _fit_xgb(params, n_estimators, X_tr, ya_tr)
    models_tmp = {"model_home": mh, "model_away": ma,
                  "feature_cols": FEATURE_COLS}
    rho_ml = estimate_rho(models_tmp, train, cats)
    lam_h = np.clip(mh.predict(X_va), 0.05, None)
    lam_a = np.clip(ma.predict(X_va), 0.05, None)
    M_ml = np.stack([score_matrix_agg(h, a, rho=rho_ml)
                     for h, a in zip(lam_h, lam_a)])

    dc_val = DixonColesModel(since=DC_SINCE, half_life_years=DC_HALF_LIFE)
    dc_val.fit(df, cutoff=pd.Timestamp(f"{VAL_START}-01-01"))
    M_dc = np.stack([
        dc_val.score_matrix(r.home_team, r.away_team,
                            bool(r.true_home_advantage))
        for r in val.itertuples()
    ])

    ih = np.minimum(yh_va.values.astype(int), 6)
    ij = np.minimum(ya_va.values.astype(int), 6)

    def mat_ll(mats: np.ndarray) -> float:
        p = mats[np.arange(len(mats)), ih, ij]
        return float(-np.log(np.maximum(p, 1e-12)).mean())

    grid = np.round(np.arange(0.0, 1.001, 0.05), 2)
    lls = [mat_ll(w * M_ml + (1 - w) * M_dc) for w in grid]
    w = float(grid[int(np.argmin(lls))])
    print(f"w óptimo en validación: {w:.2f} "
          f"(ll ensemble {min(lls):.4f} | XGB {mat_ll(M_ml):.4f} | "
          f"DC {mat_ll(M_dc):.4f})")

    # ---- paso 2: reentrenar TODO y congelar ----
    print(f"\nreentrenando con TODO: {len(df)} partidos "
          f"({df['date'].min().date()} a {df['date'].max().date()})")
    X, yh, ya = make_xy(df)
    models = {
        "model_home": _fit_xgb(params, n_estimators, X, yh),
        "model_away": _fit_xgb(params, n_estimators, X, ya),
        "feature_cols": FEATURE_COLS,
    }
    rho = estimate_rho(models, df, cats)
    dc = DixonColesModel(since=DC_SINCE, half_life_years=DC_HALF_LIFE).fit(df)
    print(f"rho XGB {rho:+.4f} | DC: gamma={dc.gamma:.3f}, rho={dc.rho:+.4f}, "
          f"{dc.n_matches} partidos")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    models["model_home"].save_model(FINAL_DIR / "xgb_home.json")
    models["model_away"].save_model(FINAL_DIR / "xgb_away.json")
    (FINAL_DIR / "dc.json").write_text(json.dumps(dc.to_dict()))
    meta = {
        "version": "model_final_wc2026",
        "created": datetime.now().isoformat(timespec="seconds"),
        "trained_through": str(df["date"].max().date()),
        "n_matches": len(df),
        "params": params,
        "n_estimators": n_estimators,
        "rho": rho,
        "ensemble_w": w,
        "dc": {"since": DC_SINCE, "half_life_years": DC_HALF_LIFE,
               "gamma": dc.gamma, "rho": dc.rho},
        "feature_cols": FEATURE_COLS,
        "tuning": {"best_value": tuning["best_value"],
                   "default_value": tuning["default_value"]},
        "policy": ("CONGELADO durante el Mundial 2026: no reentrenar. "
                   "Las features (Elo, forma, descanso) se recalculan con "
                   "cada corrida a partir de data/raw actualizado."),
    }
    (FINAL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nModelo final (ensemble w={w:.2f}) congelado en {FINAL_DIR}")


if __name__ == "__main__":
    main()
