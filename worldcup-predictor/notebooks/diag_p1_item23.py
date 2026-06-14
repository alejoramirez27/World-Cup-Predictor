"""Diagnóstico P1, items 2 (peso de amistosos de preparación) y 3
(calibración del bonus de localía). Entrena modelos TEMPORALES; NO toca
el modelo congelado (models/model_final_wc2026)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features
from data_loader import TOURNAMENT_TYPES, add_basic_features, filter_since, load_results
from features import FeatureBuilder
from model import (calibration_table, estimate_rho, evaluate, outcome_probs,
                   predict_match, train_models)

PAIRS = [("Brazil", "Argentina"), ("Spain", "France"), ("England", "Netherlands"),
         ("Mexico", "United States"), ("Germany", "Belgium"),
         ("Portugal", "Croatia"), ("Morocco", "Senegal"), ("Japan", "South Korea"),
         ("Colombia", "Ecuador"), ("Uruguay", "Paraguay")]


def prep_schedule() -> dict:
    """Fechas de torneos mayores (world_cup/continental) por equipo, incl.
    fixtures futuros 2026 (la agenda es conocida -> sin leakage)."""
    raw = load_results()
    raw["tt"] = raw["tournament"].map(TOURNAMENT_TYPES).fillna("other")
    maj = raw[raw["tt"].isin(["world_cup", "continental"])]
    d: dict[str, list] = {}
    for col in ("home_team", "away_team"):
        for t, dt in zip(maj[col], maj["date"]):
            d.setdefault(t, []).append(np.datetime64(dt))
    return {t: np.sort(np.array(v)) for t, v in d.items()}


def build(prep_weight, prep_dates):
    full = add_basic_features(filter_since(load_results(), 1990))
    cutoff = pd.Timestamp("2010-01-01")
    b = FeatureBuilder()
    b.prep_friendly_weight = prep_weight
    b._prep_tourn_dates = prep_dates
    b.warm_up(full[full["date"] < cutoff])
    df = b.transform(full[full["date"] >= cutoff].reset_index(drop=True))
    return df, b


def fit_eval(df):
    models = train_models(df)
    train = df[df["year"].between(2010, 2023)]
    cats = df["match_type"].astype("category").cat.categories
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    ct = calibration_table(ev["probs"], ev["y_onehot"])
    ece = float((ct["count"] * ct["gap"].abs()).sum() / ct["count"].sum())
    return models, rho, ev, ece


# =================================================== ITEM 2
print("=" * 70)
print("ITEM 2 — peso de amistosos de preparación (ventana 60d pre-torneo)")
sched = prep_schedule()

for label, pw in [("baseline (todos 0.40)", None), ("prep -> 0.30", 0.30),
                  ("prep -> 0.25", 0.25)]:
    df, b = build(pw, sched)
    models, rho, ev, ece = fit_eval(df)
    # USA: cuántos de sus últimos 10 son amistosos de preparación + predicción
    usa = b.states["United States"]
    n_prep = sum(1 for m in usa.all_matches if (not m[3]) and m[7])
    snap = b.match_features("United States", "Paraguay", "2026-07-01",
                            "world_cup", country="United States")
    pred = predict_match(models, b, "United States", "Paraguay", "2026-07-01",
                         "world_cup", rho=rho, country="United States")
    op = pred["outcome_probs"]
    print(f"\n  [{label}]")
    print(f"    val 2024-25: ll_1x2={ev['log_loss']:.4f}  brier={ev['brier_score']:.4f}"
          f"  ll_marcador={ev['scoreline_log_loss']:.4f}  ECE={ece:.4f}")
    print(f"    USA form5_perf={snap['home_form5_perf']:+.3f} "
          f"({n_prep}/10 últimos son amistosos de preparación)  "
          f"USA-PAR: {op['home_win']:.0%}/{op['draw']:.0%}/{op['away_win']:.0%}")

# =================================================== ITEM 3
print("\n" + "=" * 70)
print("ITEM 3 — calibración del bonus de localía (objetivo ~0.30-0.40 goles)")
orig = features.HOME_ELO_BONUS
for bonus in (100, 140, 180):
    features.HOME_ELO_BONUS = bonus
    df, b = build(None, {})
    models, rho, ev, ece = fit_eval(df)
    deltas = []
    for h, a in PAIRS:
        radv = predict_match(models, b, h, a, "2026-07-01", "wc_qualifier",
                             neutral=False, rho=rho)
        rno = predict_match(models, b, h, a, "2026-07-01", "wc_qualifier",
                            neutral=True, rho=rho)
        deltas.append(radv["lambda_home"] - rno["lambda_home"])
    print(f"  bonus={bonus:>3}: XGB home-adv = {np.mean(deltas):+.3f} goles | "
          f"ll_1x2={ev['log_loss']:.4f} ll_marcador={ev['scoreline_log_loss']:.4f} "
          f"ECE={ece:.4f}")
features.HOME_ELO_BONUS = orig
