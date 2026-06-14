"""Diagnóstico de la feature played_qualifiers_cycle:
1) importancia en ambos regresores, 2) soporte en entrenamiento,
3) ablación comparando predicciones para los partidos de grupo de los
   3 anfitriones del Mundial 2026 (con vs sin la feature)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,
                   predict_match, prepare_data, train_models)
from worldcup_2026 import load_group_fixtures

pd.set_option("display.width", 200)

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

# ---------------------------------------------------------------- 1) importancia
print("=" * 70)
print("1) FEATURE IMPORTANCE (gain)")
models_full = train_models(df)
for side in ("home", "away"):
    model = models_full[f"model_{side}"]
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print(f"\n  Regresor goles_{side} — top 10:")
    for name, val in imp.head(10).items():
        print(f"    {val:.4f}  {name}")
    for feat in ("home_played_qualifiers_cycle", "away_played_qualifiers_cycle"):
        rank = list(imp.index).index(feat) + 1
        print(f"    -> {feat}: importancia={imp[feat]:.4f}, "
              f"rank {rank}/{len(imp)}")

# ---------------------------------------------------------------- 2) soporte
print("\n" + "=" * 70)
print("2) SOPORTE DE played_qualifiers_cycle=False EN ENTRENAMIENTO")
n = len(train)
h_false = ~train["home_played_qualifiers_cycle"]
a_false = ~train["away_played_qualifiers_cycle"]
print(f"  filas de entrenamiento: {n}")
print(f"  home=False: {h_false.sum()} ({h_false.mean():.1%})")
print(f"  away=False: {a_false.sum()} ({a_false.mean():.1%})")
print("\n  Desglose por tipo de partido (filas con algún False):")
any_false = train[h_false | a_false]
print(any_false["match_type"].value_counts().to_string())

wc_false = train[(train["match_type"] == "world_cup") & (h_false | a_false)]
print(f"\n  Partidos de MUNDIAL con algún equipo False: {len(wc_false)}")
print(f"  ...de los cuales asimétricos (uno False, otro True) — la señal "
      f"'anfitrión': {len(train[(train['match_type'] == 'world_cup') & (h_false ^ a_false)])}")
cols = ["date", "home_team", "away_team", "home_score", "away_score",
        "home_played_qualifiers_cycle", "away_played_qualifiers_cycle"]
asym = train[(train["match_type"] == "world_cup") & (h_false ^ a_false)]
print("\n  Lista de partidos de Mundial asimétricos:")
print(asym[cols].to_string(index=False))
out_csv = Path(__file__).parent / "played_qualifiers_false_rows.csv"
any_false[cols + ["match_type"]].to_csv(out_csv, index=False)
print(f"\n  (lista completa de {len(any_false)} filas con algún False -> {out_csv.name})")

# ---------------------------------------------------------------- 3) ablación
print("\n" + "=" * 70)
print("3) ABLACION: predicciones para los anfitriones, con vs sin la feature")
cols_abl = [c for c in FEATURE_COLS if "played_qualifiers" not in c]
models_abl = train_models(df, feature_cols=cols_abl)
print(f"  métricas  con: {models_full['metrics']['accuracy_1x2']:.4f} acc | "
      f"sin: {models_abl['metrics']['accuracy_1x2']:.4f} acc")

rho_full = estimate_rho(models_full, train, cats)
rho_abl = estimate_rho(models_abl, train, cats)

fixtures = load_group_fixtures()
hosts = ["Mexico", "United States", "Canada"]
host_fx = fixtures[
    fixtures["home_team"].isin(hosts) | fixtures["away_team"].isin(hosts)
].sort_values("date")

print(f"\n  {'partido':<42}{'con feature':>22}{'sin feature':>22}")
for row in host_fx.itertuples():
    preds = {}
    for tag, mods, rho in [("con", models_full, rho_full),
                           ("sin", models_abl, rho_abl)]:
        p = predict_match(mods, builder, row.home_team, row.away_team,
                          row.date, "world_cup", neutral=bool(row.neutral),
                          rho=rho)
        o = p["outcome_probs"]
        preds[tag] = (f"{o['home_win']:.0%}/{o['draw']:.0%}/{o['away_win']:.0%} "
                      f"({p['lambda_home']:.2f}-{p['lambda_away']:.2f})")
    label = f"{row.home_team} vs {row.away_team}"
    print(f"  {label:<42}{preds['con']:>22}{preds['sin']:>22}")
