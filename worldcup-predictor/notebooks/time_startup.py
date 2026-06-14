"""Cronometra el arranque (prepare_data) y confirma que las predicciones
no cambiaron tras optimizar warm_up/transform."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

t0 = time.perf_counter()
from worldcup_2026 import WorldCup2026Predictor

p = WorldCup2026Predictor()
print(f"startup: {time.perf_counter() - t0:.1f}s")

top = sorted(p.builder.states.items(), key=lambda kv: kv[1].elo, reverse=True)[:3]
print("top Elo:", ", ".join(f"{t} {s.elo:.0f}" for t, s in top))

pr = p.predict("United States", "Paraguay")
o = pr["outcome_probs"]
print(f"USA-PAR: {o['home_win']:.0%}/{o['draw']:.0%}/{o['away_win']:.0%} "
      f"(esperado 36/28/36)")
