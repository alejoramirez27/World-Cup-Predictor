"""Entrenamiento y predicción: dos regresores XGBoost (goles local / visitante)
con objetivo Poisson, validación con split temporal y marcadores vía matriz
de Poisson.

Split temporal:
  - warm-up Elo: 1990-2009 (solo ratings)
  - entrenamiento: 2002-2023
  - validación: 2024-2025
"""

from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from data_loader import add_basic_features, filter_since, load_results
from features import FeatureBuilder

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ELO_START = 1990
# Ventana de entrenamiento desde 2002 (era moderna, ~21k partidos). Probado en
# notebooks/test_train_window.py: más historia mejora log-loss/Brier de forma
# consistente; el grueso de la ganancia se captura en 2002 y luego se aplana.
TRAIN_START, TRAIN_END = 2002, 2023
VAL_START, VAL_END = 2024, 2025

MAX_GOALS = 6  # la matriz de Poisson cubre marcadores 0-6

# columnas de features por equipo generadas por features.py.
# Historial de selección (ver notebooks/):
# - played_qualifiers_cycle -> comp_share10 (diagnose_played_qualifiers.py):
#   el booleano era False en el 52% de las filas y casi no aportaba.
# - promedios crudos de goles + form5_weighted -> versiones ajustadas por
#   calidad del rival (test_adjusted_form.py): gf/ga_adj = goles relativos a
#   la expectativa según el Elo del rival, y form5_perf = (resultado -
#   esperado por Elo) ponderado. Mejor log-loss/Brier/accuracy con 18
#   features en vez de 26.
# - contexto (alt_diff, rest_days, travel_km, is_knockout) probado y NO
#   adoptado (test_context_features.py): empeora log-loss/Brier/top-3.
#   Las features se siguen calculando en features.py y se pueden activar
#   con feature_cols si se quiere reintentar con más datos.
_TEAM_FEATS = [
    "elo", "gf_adj5", "ga_adj5", "gf_adj10", "ga_adj10",
    "form5_perf", "comp_share10",
]
FEATURE_COLS = (
    [f"home_{c}" for c in _TEAM_FEATS]
    + [f"away_{c}" for c in _TEAM_FEATS]
    + ["elo_diff", "elo_expected_home", "true_home_advantage", "match_type"]
)


def prepare_data() -> tuple[pd.DataFrame, FeatureBuilder]:
    """Carga el dataset, corre el warm-up de Elo desde 1990 y genera las
    features para los partidos desde 2010."""
    full = add_basic_features(filter_since(load_results(), ELO_START))
    cutoff = pd.Timestamp(f"{TRAIN_START}-01-01")
    builder = FeatureBuilder()
    builder.warm_up(full[full["date"] < cutoff])
    df = builder.transform(full[full["date"] >= cutoff].reset_index(drop=True))
    return df, builder


def make_xy(df: pd.DataFrame,
            feature_cols: list[str] | None = None
            ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[feature_cols or FEATURE_COLS].copy()
    X["match_type"] = X["match_type"].astype("category")
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    return X, df["home_score"], df["away_score"]


def _make_regressor(objective: str = "count:poisson") -> XGBRegressor:
    return XGBRegressor(
        objective=objective,  # count:poisson garantiza lambdas positivas
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        eval_metric="mae",
    )


def train_models(df: pd.DataFrame,
                 feature_cols: list[str] | None = None,
                 half_life_years: float | None = None,
                 objective: str = "count:poisson") -> dict:
    """Entrena los dos regresores con split temporal y devuelve modelos
    y métricas de validación.

    `feature_cols`: subconjunto de features (ablación).
    `half_life_years`: decaimiento temporal exponencial — cada partido pesa
      0.5^(antigüedad_en_años / half_life), con la antigüedad medida desde
      el fin del período de entrenamiento. None = sin decaimiento.
    """
    feature_cols = feature_cols or FEATURE_COLS
    train = df[df["year"].between(TRAIN_START, TRAIN_END)]
    val = df[df["year"].between(VAL_START, VAL_END)]

    # categorías fijadas sobre todo el df para que train y val coincidan
    X_tr, yh_tr, ya_tr = make_xy(train, feature_cols)
    X_val, yh_val, ya_val = make_xy(val, feature_cols)
    cats = df["match_type"].astype("category").cat.categories
    X_tr["match_type"] = X_tr["match_type"].cat.set_categories(cats)
    X_val["match_type"] = X_val["match_type"].cat.set_categories(cats)

    weights = None
    if half_life_years:
        age_years = (
            pd.Timestamp(f"{TRAIN_END}-12-31") - train["date"]
        ).dt.days / 365.25
        weights = (0.5 ** (age_years / half_life_years)).values

    model_home = _make_regressor(objective)
    model_home.fit(X_tr, yh_tr, sample_weight=weights,
                   eval_set=[(X_val, yh_val)], verbose=False)
    model_away = _make_regressor(objective)
    model_away.fit(X_tr, ya_tr, sample_weight=weights,
                   eval_set=[(X_val, ya_val)], verbose=False)

    lam_home = np.clip(model_home.predict(X_val), 0.05, None)
    lam_away = np.clip(model_away.predict(X_val), 0.05, None)

    mae_home = mean_absolute_error(yh_val, lam_home)
    mae_away = mean_absolute_error(ya_val, lam_away)

    # resultado 1X2 derivado de la matriz de Poisson de cada partido
    pred_outcome = np.array([
        _most_likely_outcome(lh, la) for lh, la in zip(lam_home, lam_away)
    ])
    actual = np.where(
        yh_val > ya_val, "home_win", np.where(yh_val == ya_val, "draw", "away_win")
    )
    acc_1x2 = float((pred_outcome == actual).mean())

    return {
        "model_home": model_home,
        "model_away": model_away,
        "feature_cols": feature_cols,
        "metrics": {
            "mae_home": mae_home,
            "mae_away": mae_away,
            "accuracy_1x2": acc_1x2,
            "n_train": len(train),
            "n_val": len(val),
        },
    }


# ------------------------------------------------------------- evaluación ---

OUTCOME_LABELS = ["home_win", "draw", "away_win"]
REPORTS_DIR = MODELS_DIR.parent / "reports"


def evaluate(models: dict, df: pd.DataFrame, rho: float = 0.0,
             temperature: float = 1.0) -> dict:
    """Evaluación completa sobre el set de validación 2024-2025:
    log-loss, accuracy 1X2 (vs. baseline naive de Elo) y Brier score.
    Las probabilidades 1X2 salen de la matriz de Poisson de cada partido,
    con ajuste Dixon-Coles si rho != 0 y temperature scaling si T != 1."""
    from sklearn.metrics import log_loss

    val = df[df["year"].between(VAL_START, VAL_END)]
    X_val, yh, ya = make_xy(val, models.get("feature_cols"))
    cats = df["match_type"].astype("category").cat.categories
    X_val["match_type"] = X_val["match_type"].cat.set_categories(cats)

    lam_h = np.clip(models["model_home"].predict(X_val), 0.05, None)
    lam_a = np.clip(models["model_away"].predict(X_val), 0.05, None)
    probs = np.array([
        [op["home_win"], op["draw"], op["away_win"]]
        for op in (outcome_probs(h, a, rho=rho) for h, a in zip(lam_h, lam_a))
    ])
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs /= probs.sum(axis=1, keepdims=True)  # corrige redondeo flotante

    y = np.where(yh.values > ya.values, 0, np.where(yh.values == ya.values, 1, 2))
    y_onehot = np.eye(3)[y]

    acc = float((probs.argmax(axis=1) == y).mean())
    # baseline naive: siempre gana el equipo con mayor Elo pre-partido
    naive_pred = np.where(val["home_elo"].values > val["away_elo"].values, 0, 2)
    acc_naive = float((naive_pred == y).mean())

    # RPS (ranked probability score) sobre 1X2: penaliza según la "distancia"
    # ordinal del error (predecir local cuando gana visitante cuesta más que
    # predecir local cuando es empate)
    c_p = np.cumsum(probs, axis=1)[:, :2]
    c_o = np.cumsum(y_onehot, axis=1)[:, :2]
    rps = float(np.mean(np.sum((c_p - c_o) ** 2, axis=1) / 2))

    return {
        "n_val": len(val),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy_1x2": acc,
        "accuracy_naive_elo": acc_naive,
        "brier_score": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "rps": rps,
        **_scoreline_metrics(lam_h, lam_a, yh.values, ya.values, rho),
        "probs": probs,
        "y": y,
        "y_onehot": y_onehot,
        "years": val["year"].values,
    }


def probs_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """Log-loss, Brier y accuracy 1X2 a partir de probabilidades ya calculadas."""
    from sklearn.metrics import log_loss

    y_onehot = np.eye(3)[y]
    return {
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier_score": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "accuracy_1x2": float((probs.argmax(axis=1) == y).mean()),
    }


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling: T > 1 suaviza (corrige sobreconfianza)."""
    p = probs ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Ajusta T minimizando el log-loss sobre un set de calibración."""
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss

    def nll(t: float) -> float:
        return log_loss(y, apply_temperature(probs, t), labels=[0, 1, 2])

    res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    return float(res.x)


def calibration_table(probs: np.ndarray, y_onehot: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibración pooled one-vs-rest: cada probabilidad predicha
    (3 por partido) se agrupa en bins y se compara con la frecuencia real."""
    pred = probs.ravel()
    obs = y_onehot.ravel()
    bins = np.clip((pred * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"[{b / n_bins:.1f}, {(b + 1) / n_bins:.1f})",
            "mean_predicted": float(pred[mask].mean()),
            "observed_freq": float(obs[mask].mean()),
            "count": int(mask.sum()),
        })
    table = pd.DataFrame(rows)
    table["gap"] = table["mean_predicted"] - table["observed_freq"]
    return table


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    """Guarda el reliability diagram como PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibración perfecta")
    ax.plot(table["mean_predicted"], table["observed_freq"],
            "o-", color="tab:blue", label="modelo")
    for _, row in table.iterrows():
        ax.annotate(f"n={row['count']}", (row["mean_predicted"], row["observed_freq"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7,
                    color="gray")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title(f"Calibración 1X2 — validación {VAL_START}-{VAL_END}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Poisson ---

def _dc_tau(x: int, y: int, lam_home: float, lam_away: float,
            rho: float) -> float:
    """Factor tau de Dixon-Coles (1997) que corrige la dependencia entre
    goles en los marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1.0 - lam_home * lam_away * rho
    if x == 0 and y == 1:
        return 1.0 + lam_home * rho
    if x == 1 and y == 0:
        return 1.0 + lam_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def poisson_matrix(lam_home: float, lam_away: float,
                   max_goals: int = MAX_GOALS, rho: float = 0.0) -> np.ndarray:
    """Matriz (max_goals+1) x (max_goals+1) con P(local=i, visitante=j),
    normalizada para que la masa truncada sume 1. Con rho != 0 aplica el
    ajuste Dixon-Coles a las celdas (0,0), (1,0), (0,1) y (1,1)."""
    p_home = np.array([exp(-lam_home) * lam_home**i / factorial(i)
                       for i in range(max_goals + 1)])
    p_away = np.array([exp(-lam_away) * lam_away**j / factorial(j)
                       for j in range(max_goals + 1)])
    matrix = np.outer(p_home, p_away)
    if rho != 0.0:
        for x in (0, 1):
            for y in (0, 1):
                matrix[x, y] *= _dc_tau(x, y, lam_home, lam_away, rho)
    return matrix / matrix.sum()


def score_matrix_agg(lam_home: float, lam_away: float,
                     max_goals: int = MAX_GOALS,
                     rho: float = 0.0) -> np.ndarray:
    """Matriz de marcadores con cola agregada: la última fila/columna
    acumula P(goles >= max_goals), así la matriz suma 1 sin renormalizar
    por truncamiento. Es la versión correcta para log-loss de marcador
    exacto (la celda '6' significa '6 o más')."""

    def pvec(lam: float) -> np.ndarray:
        p = np.array([exp(-lam) * lam**i / factorial(i)
                      for i in range(max_goals)])
        return np.append(p, max(1.0 - p.sum(), 1e-12))

    m = np.outer(pvec(lam_home), pvec(lam_away))
    if rho != 0.0:
        for x in (0, 1):
            for y in (0, 1):
                m[x, y] *= _dc_tau(x, y, lam_home, lam_away, rho)
        m /= m.sum()
    return m


def _scoreline_metrics(lam_h: np.ndarray, lam_a: np.ndarray,
                       yh: np.ndarray, ya: np.ndarray,
                       rho: float, max_goals: int = MAX_GOALS) -> dict:
    """Log-loss del marcador exacto (celda 6+ agregada) y tasas de acierto
    del marcador más probable (top-1) y de estar entre los 3 primeros."""
    ll, top1, top3 = [], [], []
    for lh, la, gh, ga in zip(lam_h, lam_a, yh, ya):
        m = score_matrix_agg(lh, la, max_goals, rho)
        i, j = min(int(gh), max_goals), min(int(ga), max_goals)
        ll.append(-np.log(max(m[i, j], 1e-12)))
        order = np.argsort(m.ravel())[::-1]
        idx = i * (max_goals + 1) + j
        top1.append(idx == order[0])
        top3.append(idx in order[:3])
    return {
        "scoreline_log_loss": float(np.mean(ll)),
        "top1_scoreline": float(np.mean(top1)),
        "top3_scoreline": float(np.mean(top3)),
    }


def estimate_rho(models: dict, train_df: pd.DataFrame,
                 cats: pd.Index) -> float:
    """Estima rho maximizando la log-verosimilitud de los marcadores reales
    del set de entrenamiento. Solo los partidos con ambos marcadores <= 1
    aportan al término tau; el resto es constante en rho."""
    from scipy.optimize import minimize_scalar

    X, yh, ya = make_xy(train_df, models.get("feature_cols"))
    X["match_type"] = X["match_type"].cat.set_categories(cats)
    lam_h = np.clip(models["model_home"].predict(X), 0.05, None)
    lam_a = np.clip(models["model_away"].predict(X), 0.05, None)

    mask = (yh.values <= 1) & (ya.values <= 1)
    xs, ys = yh.values[mask], ya.values[mask]
    lh, la = lam_h[mask], lam_a[mask]

    def neg_loglik(rho: float) -> float:
        tau = np.ones_like(lh)
        c00 = (xs == 0) & (ys == 0)
        c01 = (xs == 0) & (ys == 1)
        c10 = (xs == 1) & (ys == 0)
        c11 = (xs == 1) & (ys == 1)
        tau[c00] = 1.0 - lh[c00] * la[c00] * rho
        tau[c01] = 1.0 + lh[c01] * rho
        tau[c10] = 1.0 + la[c10] * rho
        tau[c11] = 1.0 - rho
        if (tau <= 0).any():
            return np.inf
        return -float(np.log(tau).sum())

    res = minimize_scalar(neg_loglik, bounds=(-0.5, 0.5), method="bounded")
    return float(res.x)


def outcome_probs(lam_home: float, lam_away: float,
                  max_goals: int = MAX_GOALS,
                  rho: float = 0.0) -> dict[str, float]:
    m = poisson_matrix(lam_home, lam_away, max_goals, rho)
    return {
        "home_win": float(np.tril(m, -1).sum()),
        "draw": float(np.trace(m)),
        "away_win": float(np.triu(m, 1).sum()),
    }


def _most_likely_outcome(lam_home: float, lam_away: float,
                         rho: float = 0.0) -> str:
    probs = outcome_probs(lam_home, lam_away, rho=rho)
    return max(probs, key=probs.get)


def predict_scoreline(lam_home: float, lam_away: float,
                      max_goals: int = MAX_GOALS, top: int = 3,
                      rho: float = 0.0) -> list[tuple[str, float]]:
    """Los `top` marcadores más probables dado el par de lambdas, con
    ajuste Dixon-Coles si rho != 0.

    Devuelve [("2-1", 0.081), ...] ordenado de mayor a menor probabilidad.
    """
    m = poisson_matrix(lam_home, lam_away, max_goals, rho)
    flat = [(f"{i}-{j}", float(m[i, j]))
            for i in range(max_goals + 1) for j in range(max_goals + 1)]
    return sorted(flat, key=lambda t: t[1], reverse=True)[:top]


# ------------------------------------------------------------- predicción ---

def predict_match(models: dict, builder: FeatureBuilder, home_team: str,
                  away_team: str, date, match_type: str,
                  neutral: bool = True, rho: float = 0.0,
                  city: str | None = None, country: str | None = None,
                  knockout: bool = False) -> dict:
    """Predice un partido futuro: lambdas, probabilidades 1X2 y top marcadores.
    `rho`: coeficiente Dixon-Coles (usar el de models['rho'] si se estimó).
    `city`/`country`/`knockout`: contexto de la sede y fase del torneo."""
    feats = builder.match_features(home_team, away_team, date, match_type,
                                   neutral, city=city, country=country,
                                   knockout=knockout)
    cols = models.get("feature_cols") or FEATURE_COLS
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
    X["match_type"] = X["match_type"].astype("category")
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    lam_home = float(max(models["model_home"].predict(X)[0], 0.05))
    lam_away = float(max(models["model_away"].predict(X)[0], 0.05))
    return {
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "outcome_probs": outcome_probs(lam_home, lam_away, rho=rho),
        "top_scorelines": predict_scoreline(lam_home, lam_away, rho=rho),
    }


def save_models(models: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    models["model_home"].save_model(MODELS_DIR / "xgb_home.json")
    models["model_away"].save_model(MODELS_DIR / "xgb_away.json")


if __name__ == "__main__":
    print(f"Preparando datos (warm-up Elo {ELO_START}, features desde {TRAIN_START})...")
    df, builder = prepare_data()

    print(f"Entrenando: {TRAIN_START}-{TRAIN_END} | validando: {VAL_START}-{VAL_END}")
    models = train_models(df)
    m = models["metrics"]
    print(f"\n  partidos train/val: {m['n_train']} / {m['n_val']}")
    print(f"  MAE goles local:     {m['mae_home']:.4f}")
    print(f"  MAE goles visitante: {m['mae_away']:.4f}")
    print(f"  Accuracy 1X2:        {m['accuracy_1x2']:.4f}")

    save_models(models)
    print(f"\nModelos guardados en {MODELS_DIR}")

    # --- Dixon-Coles: estimar rho en el set de entrenamiento ---
    train_df = df[df["year"].between(TRAIN_START, TRAIN_END)]
    cats = df["match_type"].astype("category").cat.categories
    rho = estimate_rho(models, train_df, cats)
    aviso = "" if -0.15 <= rho <= -0.05 else "  <-- FUERA DEL RANGO TIPICO [-0.15, -0.05]"
    print(f"\nrho Dixon-Coles estimado: {rho:+.4f}{aviso}")

    print("\n--- Evaluación comparativa (validación 2024-2025) ---")
    ev_base = evaluate(models, df, rho=0.0)
    ev_dc = evaluate(models, df, rho=rho)
    print(f"  {'métrica':<22}{'Poisson':>10}{'Poisson+DC':>12}{'delta':>10}")
    for key, label in [("log_loss", "Log-loss 1X2"),
                       ("brier_score", "Brier score"),
                       ("rps", "RPS 1X2"),
                       ("accuracy_1x2", "Accuracy 1X2"),
                       ("scoreline_log_loss", "Log-loss marcador"),
                       ("top1_scoreline", "Top-1 marcador"),
                       ("top3_scoreline", "Top-3 marcador")]:
        d = ev_dc[key] - ev_base[key]
        print(f"  {label:<22}{ev_base[key]:>10.4f}{ev_dc[key]:>12.4f}{d:>+10.4f}")
    print(f"  {'Baseline naive Elo':<22}{ev_base['accuracy_naive_elo']:>10.4f}")

    tables = {}
    for tag, ev in [("poisson", ev_base), ("dc", ev_dc)]:
        tables[tag] = calibration_table(ev["probs"], ev["y_onehot"])
        png = REPORTS_DIR / f"calibration_2024_2025_{tag}.png"
        plot_calibration(tables[tag], png)
        print(f"\nCalibración [{tag}] -> {png.name}")
        print(tables[tag].to_string(index=False))

    # --- Paso 4: temperature scaling si DC no eliminó la sobreconfianza ---
    high_bins = tables["dc"][tables["dc"]["mean_predicted"] >= 0.6]
    overconf = float(high_bins["gap"].mean())
    print(f"\nGap medio en bins >=0.6 tras Dixon-Coles: {overconf:+.4f}")
    if overconf > 0.02:
        print("Sigue sobreconfiado -> temperature scaling "
              "(T ajustada en 2024, evaluada en 2025 para no contaminar la métrica)")
        y, probs, years = ev_dc["y"], ev_dc["probs"], ev_dc["years"]
        m24, m25 = years == 2024, years == 2025
        temp = fit_temperature(probs[m24], y[m24])
        print(f"  T = {temp:.3f}")

        met_dc25 = probs_metrics(probs[m25], y[m25])
        probs_t25 = apply_temperature(probs[m25], temp)
        met_t25 = probs_metrics(probs_t25, y[m25])
        print(f"\n  Solo 2025 ({int(m25.sum())} partidos):")
        print(f"  {'métrica':<22}{'DC':>10}{'DC+temp':>12}{'delta':>10}")
        for key, label in [("log_loss", "Log-loss 1X2"),
                           ("brier_score", "Brier score"),
                           ("accuracy_1x2", "Accuracy 1X2")]:
            d = met_t25[key] - met_dc25[key]
            print(f"  {label:<22}{met_dc25[key]:>10.4f}{met_t25[key]:>12.4f}{d:>+10.4f}")

        table_t = calibration_table(probs_t25, np.eye(3)[y[m25]])
        png_t = REPORTS_DIR / "calibration_2025_dc_temp.png"
        plot_calibration(table_t, png_t)
        print(f"\nCalibración [dc+temp, solo 2025] -> {png_t.name}")
        print(table_t.to_string(index=False))
    else:
        print("La sobreconfianza quedó corregida con Dixon-Coles; "
              "no se aplica temperature scaling.")

    pred = predict_match(models, builder, "Mexico", "Argentina",
                         "2026-06-20", "world_cup", rho=rho)
    print("\nEjemplo México vs Argentina (Mundial 2026):")
    print(f"  lambdas: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
    probs = pred["outcome_probs"]
    print(f"  1X2: local {probs['home_win']:.3f} | empate {probs['draw']:.3f} "
          f"| visitante {probs['away_win']:.3f}")
    print("  marcadores más probables:")
    for score, p in pred["top_scorelines"]:
        print(f"    {score}  ({p:.3f})")
