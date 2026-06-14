"""Diagnóstico P1, items 1 (half-life DC) y 3 (sensibilidad de localía).
Solo lectura: NO toca el modelo congelado."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dixon_coles import DixonColesModel
from model import predict_match
from worldcup_2026 import WorldCup2026Predictor

p = WorldCup2026Predictor()
df = p.df
val = df[df["year"].between(2024, 2025)].reset_index(drop=True)
yh, ya = val["home_score"].values, val["away_score"].values
ih, ij = np.minimum(yh.astype(int), 6), np.minimum(ya.astype(int), 6)


def mat_ll(M):
    return float(-np.log(np.maximum(M[np.arange(len(M)), ih, ij], 1e-12)).mean())


# ---------------------------------------------------------- item 1
print("=" * 64)
print("ITEM 1 — half-life Dixon-Coles (cutoff 2024, eval val 2024-2025)")
cutoff = pd.Timestamp("2024-01-01")
d = df[(df["date"].dt.year >= 2018) & (df["date"] < cutoff)]
print(f"  (partidos base 2018-2023: {len(d)})\n")
print(f"  {'half-life':>10}{'ll_marcador':>13}{'efec/equipo med':>17}"
      f"{'p10':>7}{'min':>7}")
for hl in (1.5, 2.0, 3.0, 4.0):
    dc = DixonColesModel(since=2018, half_life_years=hl).fit(df, cutoff=cutoff)
    M = np.stack([dc.score_matrix(r.home_team, r.away_team,
                                  bool(r.true_home_advantage))
                  for r in val.itertuples()])
    ll = mat_ll(M)
    age = (cutoff - d["date"]).dt.days / 365.25
    w = (0.5 ** (age / hl)).values
    eff = defaultdict(float)
    for tt, ww in zip(d["home_team"], w):
        eff[tt] += ww
    for tt, ww in zip(d["away_team"], w):
        eff[tt] += ww
    e = np.array(list(eff.values()))
    print(f"  {hl:>10.1f}{ll:>13.4f}{np.median(e):>17.1f}"
          f"{np.percentile(e, 10):>7.1f}{e.min():>7.1f}")

# ---------------------------------------------------------- item 3
print("\n" + "=" * 64)
print("ITEM 3 — sensibilidad de localía (goles que suma la ventaja local)")
dc_full = DixonColesModel(since=2018, half_life_years=4.0).fit(df)
g = dc_full.gamma
bases = [dc_full.lambdas(r.home_team, r.away_team, False)[0]
         for r in val.itertuples()]
base = float(np.mean(bases))
print(f"  DC: gamma={g:.4f}  exp(gamma)={np.exp(g):.3f}  "
      f"-> efecto aditivo medio = {base * (np.exp(g) - 1):.3f} goles "
      f"(lambda base medio {base:.2f})")

# XGB / ensemble: toggle de localía en pares parejos (wc_qualifier, fecha
# futura para evitar leakage). En WC2026 el override de anfitrión impide
# togglear, por eso se usa wc_qualifier.
pairs = [("Brazil", "Argentina"), ("Spain", "France"), ("England", "Netherlands"),
         ("Mexico", "United States"), ("Germany", "Belgium"),
         ("Portugal", "Croatia"), ("Morocco", "Senegal"), ("Japan", "South Korea"),
         ("Colombia", "Ecuador"), ("Uruguay", "Paraguay")]
dml, dens = [], []
for h, a in pairs:
    radv = predict_match(p.models, p.builder, h, a, "2026-07-01",
                         "wc_qualifier", neutral=False, rho=p.rho)
    rno = predict_match(p.models, p.builder, h, a, "2026-07-01",
                        "wc_qualifier", neutral=True, rho=p.rho)
    dml.append(radv["lambda_home"] - rno["lambda_home"])
print(f"  XGB: lambda_home con localía - sin localía (media {len(pairs)} pares) "
      f"= {np.mean(dml):+.3f} goles")
print(f"  HOME_ELO_BONUS actual = 100 pts Elo")
print(f"\n  benchmark empírico mundiales: +0.30 a +0.40 goles")
