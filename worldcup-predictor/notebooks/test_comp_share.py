"""Compara tres variantes de features sobre la validación 2024-2025 y los
partidos de grupos de los anfitriones del Mundial 2026:
  A) original: flag played_qualifiers_cycle
  B) sin el flag
  C) flag reemplazado por comp_share10 (proporción de competitivos en últimos 10)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,
                   evaluate, predict_match, prepare_data, train_models)
from worldcup_2026 import load_group_fixtures

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

base_no_flag = [c for c in FEATURE_COLS if "played_qualifiers" not in c]
variants = {
    "A_flag": FEATURE_COLS,
    "B_sin": base_no_flag,
    "C_comp_share": base_no_flag + ["home_comp_share10", "away_comp_share10"],
}

results = {}
for tag, cols in variants.items():
    models = train_models(df, feature_cols=cols)
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    results[tag] = (models, rho, ev)

print(f"{'variante':<16}{'log-loss':>10}{'brier':>10}{'acc 1X2':>10}{'rho':>9}")
for tag, (_, rho, ev) in results.items():
    print(f"{tag:<16}{ev['log_loss']:>10.4f}{ev['brier_score']:>10.4f}"
          f"{ev['accuracy_1x2']:>10.4f}{rho:>+9.4f}")

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
