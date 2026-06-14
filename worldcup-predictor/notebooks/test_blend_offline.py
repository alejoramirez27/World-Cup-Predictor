"""Test offline del blend y el detector de valor con filas sintéticas
(en memoria — no guarda nada en el tracking real)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live_tracking import BLEND_W_DEFAULT, LiveTracker


def fake_row(home, away, pm, pk, outcome="", scores=(np.nan, np.nan)):
    p_actual = dict(zip(["home_win", "draw", "away_win"], pm)).get(outcome)
    return {
        "date": "2026-06-15", "home_team": home, "away_team": away,
        "lambda_home": 1.5, "lambda_away": 1.0,
        "p_home": pm[0], "p_draw": pm[1], "p_away": pm[2],
        "top_scorelines": "1-0:0.12",
        "odds_home": 1 / pk[0] * 1.05, "odds_draw": 1 / pk[1] * 1.05,
        "odds_away": 1 / pk[2] * 1.05,
        "p_mkt_home": pk[0], "p_mkt_draw": pk[1], "p_mkt_away": pk[2],
        "odds_ts": "2026-06-15T00:00:00+00:00", "n_bookmakers": 5,
        "home_score": scores[0], "away_score": scores[1], "outcome": outcome,
        "logloss_model": -np.log(p_actual) if outcome else np.nan,
        "logloss_market": np.nan,
    }


tracker = LiveTracker()
rng = np.random.default_rng(7)

# --- caso 1: pocos partidos jugados -> w default 0.3 ---
rows = [fake_row("A", "B", (0.5, 0.3, 0.2), (0.45, 0.3, 0.25),
                 "home_win", (2, 0)) for _ in range(3)]
tracker.df = pd.DataFrame(rows)
w, src = tracker.blend_weight()
assert w == BLEND_W_DEFAULT, (w, src)
print(f"caso 1 (3 jugados): w={w} -> '{src}'  OK")

# --- caso 2: 12 jugados donde el mercado es mejor -> w bajo ---
rows = []
for _ in range(12):
    # el resultado se sortea con las probs del mercado: el mercado "sabe"
    pk = rng.dirichlet([5, 3, 3])
    pm = rng.dirichlet([3, 3, 3])  # modelo ruidoso
    out = ["home_win", "draw", "away_win"][rng.choice(3, p=pk)]
    rows.append(fake_row("A", "B", pm, pk, out, (1, 0)))
tracker.df = pd.DataFrame(rows)
w, src = tracker.blend_weight()
m = tracker.metrics()
print(f"caso 2 (mercado mejor): w={w} ({src})")
print(f"  ll modelo={m['ll_model_subset']:.4f} mercado={m['ll_market']:.4f} "
      f"blend={m['ll_blend']:.4f}")
assert m["ll_blend"] <= min(m["ll_model_subset"], m["ll_market"]) + 1e-9

# --- caso 3: detector de valor con warning >10pp ---
up = [
    fake_row("United States", "Paraguay", (0.27, 0.27, 0.46), (0.49, 0.27, 0.24)),
    fake_row("Spain", "Uruguay", (0.55, 0.25, 0.20), (0.52, 0.26, 0.22)),
]
tracker.df = pd.DataFrame(rows + up)
t = tracker.value_table()
print(f"\ncaso 3: {len(t)} filas de valor, warnings={int(t['warn'].sum())}")
assert t.iloc[0]["match"] == "United States vs Paraguay"  # mayor discrepancia primero
assert t.iloc[0]["warn"], "la discrepancia de 22pp debe disparar warning"
assert not t[t["match"] == "Spain vs Uruguay"]["warn"].any()
tracker.print_value()

# --- caso 4: export web a directorio temporal ---
with tempfile.TemporaryDirectory() as tmp:
    out = tracker.export_web(Path(tmp) / "web")
    files = sorted(p.name for p in out.iterdir())
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "data.json" in files and "index.html" in files
    assert "Detector de valor" in html and "United States" in html
    print(f"\ncaso 4: export OK -> {files}")

print("\nTODOS LOS CASOS OK")
