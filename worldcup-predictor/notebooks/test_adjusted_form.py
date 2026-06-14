"""Prueba a tres bandas de las features ajustadas por calidad del rival:
  A_raw:  features actuales (promedios crudos de goles + form5_weighted)
  B_adj:  goles/expectativa-Elo del rival (gf_adj, ga_adj) + form5_perf
  C_both: todas juntas
Compara log-loss/Brier/accuracy en validación 2024-2025 y las predicciones
para los partidos de grupos de los 3 anfitriones del Mundial 2026."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,
                   evaluate, predict_match, prepare_data, train_models)
from worldcup_2026 import load_group_fixtures

df, builder = prepare_data()
gm = builder.goal_model
print(f"curva de expectativa (ajustada en warm-up 1990-2009): "
      f"E[g] = exp({gm.a:.4f} + {gm.b:.6f}*dr)")
for dr in (-500, -250, 0, 250, 500):
    print(f"  dr={dr:+5d} -> {gm.expected(dr):.2f} goles esperados")

train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

PAIR = ["elo_diff", "elo_expected_home", "true_home_advantage", "match_type"]
raw_team = ["elo", "gf_avg5", "ga_avg5", "gf_avg10", "ga_avg10",
            "gf_avg5_comp", "ga_avg5_comp", "gf_avg10_comp", "ga_avg10_comp",
            "form5_weighted", "comp_share10"]
adj_team = ["elo", "gf_adj5", "ga_adj5", "gf_adj10", "ga_adj10",
            "form5_perf", "comp_share10"]
both_team = raw_team + ["gf_adj5", "ga_adj5", "gf_adj10", "ga_adj10", "form5_perf"]


def cols(team_feats):
    return ([f"home_{c}" for c in team_feats]
            + [f"away_{c}" for c in team_feats] + PAIR)


variants = {"A_raw": cols(raw_team), "B_adj": cols(adj_team),
            "C_both": cols(both_team)}

results = {}
for tag, fc in variants.items():
    models = train_models(df, feature_cols=fc)
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    results[tag] = (models, rho, ev)

print(f"\n{'variante':<10}{'#feats':>8}{'log-loss':>10}{'brier':>10}{'acc 1X2':>10}{'rho':>9}")
for tag, (models, rho, ev) in results.items():
    print(f"{tag:<10}{len(variants[tag]):>8}{ev['log_loss']:>10.4f}"
          f"{ev['brier_score']:>10.4f}{ev['accuracy_1x2']:>10.4f}{rho:>+9.4f}")

fixtures = load_group_fixtures()
hosts = ["Mexico", "United States", "Canada"]
host_fx = fixtures[
    fixtures["home_team"].isin(hosts) | fixtures["away_team"].isin(hosts)
].sort_values("date")

print(f"\n{'partido':<36}" + "".join(f"{t:>24}" for t in variants))
for row in host_fx.itertuples():
    cells = []
    for tag, (models, rho, _) in results.items():
        p = predict_match(models, builder, row.home_team, row.away_team,
                          row.date, "world_cup", neutral=bool(row.neutral),
                          rho=rho)
        o = p["outcome_probs"]
        cells.append(f"{o['home_win']:.0%}/{o['draw']:.0%}/{o['away_win']:.0%}")
    print(f"{row.home_team + ' vs ' + row.away_team:<36}"
          + "".join(f"{c:>24}" for c in cells))
