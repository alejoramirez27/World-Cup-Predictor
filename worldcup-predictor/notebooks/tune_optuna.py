"""Tuning de hiperparámetros con Optuna sobre ambos regresores.

- Validación temporal expandida, 3 folds: entrena 2010..X, valida X+1,
  con X = 2022, 2023, 2024 (validaciones: 2023, 2024, 2025).
- Objetivo: log-loss del marcador exacto (celda 6+ agregada), promedio de
  los 3 folds. Mismos hiperparámetros para ambos regresores.
- Pruning por mediana entre folds para acelerar.

Guarda los mejores parámetros en notebooks/optuna_best_params.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import _scoreline_metrics, make_xy, prepare_data

FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]  # (train_end, val_year)
N_TRIALS = 100
OUT = Path(__file__).parent / "optuna_best_params.json"

print("Preparando datos...")
df, builder = prepare_data()
cats = df["match_type"].astype("category").cat.categories

fold_data = []
for train_end, val_year in FOLDS:
    tr = df[df["year"].between(2010, train_end)]
    va = df[df["year"] == val_year]
    X_tr, yh_tr, ya_tr = make_xy(tr)
    X_va, yh_va, ya_va = make_xy(va)
    X_tr["match_type"] = X_tr["match_type"].cat.set_categories(cats)
    X_va["match_type"] = X_va["match_type"].cat.set_categories(cats)
    fold_data.append((X_tr, yh_tr, ya_tr, X_va, yh_va, ya_va))
    print(f"  fold {train_end}->{val_year}: {len(tr)} train / {len(va)} val")


def fit_predict(params: dict, X_tr, y_tr, X_va, y_va) -> tuple[np.ndarray, int]:
    reg = XGBRegressor(
        objective="count:poisson", n_estimators=2000,
        early_stopping_rounds=50, eval_metric="poisson-nloglik",
        tree_method="hist", enable_categorical=True, **params)
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return np.clip(reg.predict(X_va), 0.05, None), reg.best_iteration


def fold_scoreline_ll(params: dict, fold) -> tuple[float, list[int]]:
    X_tr, yh_tr, ya_tr, X_va, yh_va, ya_va = fold
    lam_h, it_h = fit_predict(params, X_tr, yh_tr, X_va, yh_va)
    lam_a, it_a = fit_predict(params, X_tr, ya_tr, X_va, ya_va)
    m = _scoreline_metrics(lam_h, lam_a, yh_va.values, ya_va.values, rho=0.0)
    return m["scoreline_log_loss"], [it_h, it_a]


DEFAULTS = {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8}

print("\nBaseline (parámetros default actuales):")
default_lls = []
for fold, (te, vy) in zip(fold_data, FOLDS):
    ll, _ = fold_scoreline_ll(DEFAULTS, fold)
    default_lls.append(ll)
    print(f"  fold ->{vy}: ll marcador = {ll:.4f}")
default_mean = float(np.mean(default_lls))
print(f"  promedio: {default_mean:.4f}")


def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 20, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    lls, iters = [], []
    for k, fold in enumerate(fold_data):
        ll, its = fold_scoreline_ll(params, fold)
        lls.append(ll)
        iters.extend(its)
        trial.report(float(np.mean(lls)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("best_iterations", iters)
    return float(np.mean(lls))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_trial
print(f"\ntrials completados: {len(study.trials)} "
      f"(podados: {sum(t.state.name == 'PRUNED' for t in study.trials)})")
print(f"mejor ll marcador (3 folds): {best.value:.4f} "
      f"(default: {default_mean:.4f}, delta {best.value - default_mean:+.4f})")
print("mejores parámetros:")
for k, v in best.params.items():
    print(f"  {k}: {v}")
iters = best.user_attrs["best_iterations"]
print(f"best_iteration por fold/modelo: {iters} (mediana {int(np.median(iters))})")

OUT.write_text(json.dumps({
    "best_params": best.params,
    "best_value": best.value,
    "default_value": default_mean,
    "best_iterations": iters,
    "n_estimators_final": int(np.median(iters) * 1.1),  # +10% por más datos
}, indent=2))
print(f"guardado -> {OUT}")
