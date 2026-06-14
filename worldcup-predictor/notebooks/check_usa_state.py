"""¿Por qué el modelo da a Paraguay favorito sobre USA? Estado pre-Mundial."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import prepare_data

df, builder = prepare_data()
for team in ["United States", "Paraguay", "Australia", "Turkey", "Mexico", "Canada"]:
    st = builder.states[team]
    snap = st.snapshot(pd.Timestamp("2022-12-18"))
    print(f"{team:<15} elo={st.elo:7.1f}  form5_w={snap['form5_weighted']:5.2f}  "
          f"comp_share10={snap['comp_share10']:.1f}  "
          f"gf5={snap['gf_avg5']:.1f} ga5={snap['ga_avg5']:.1f}")

# últimos 10 partidos de USA en el dataset
m = df[(df["home_team"] == "United States") | (df["away_team"] == "United States")]
print("\nÚltimos 10 de USA:")
print(m.tail(10)[["date", "home_team", "away_team", "home_score",
                  "away_score", "tournament_type"]].to_string(index=False))
