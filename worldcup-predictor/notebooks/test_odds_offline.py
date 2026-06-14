"""Test offline de odds_fetcher: mediana entre casas, de-vig, mapeo de
nombres de la API y actualización del tracking (sin llamar a la API)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from live_tracking import LiveTracker
from odds_fetcher import _norm, median_devig, team_resolver

# evento sintético estilo The Odds API (3 casas, una sin mercado completo)
event = {
    "home_team": "USA", "away_team": "Paraguay",
    "bookmakers": [
        {"markets": [{"key": "h2h", "outcomes": [
            {"name": "USA", "price": 1.95},
            {"name": "Draw", "price": 3.50},
            {"name": "Paraguay", "price": 4.20}]}]},
        {"markets": [{"key": "h2h", "outcomes": [
            {"name": "USA", "price": 2.00},
            {"name": "Draw", "price": 3.40},
            {"name": "Paraguay", "price": 4.00}]}]},
        {"markets": [{"key": "h2h", "outcomes": [
            {"name": "USA", "price": 1.90}]}]},  # incompleta: se descarta
    ],
}
m = median_devig(event)
print(f"casas válidas: {m['n_bookmakers']} (esperado 2)")
print(f"odds mediana: {m['odds']}")
print(f"margen de la casa: {m['margin']:+.4f}")
print(f"p de-vig: {np.round(m['p_mkt'], 4)} (suma={m['p_mkt'].sum():.6f})")

resolver = team_resolver()
for api_name in ("USA", "Czechia", "Bosnia-Herzegovina", "Korea Republic",
                 "Türkiye", "Mexico"):
    print(f"  '{api_name}' -> {resolver.get(_norm(api_name))}")

tracker = LiveTracker()
ok = tracker.update_market(
    "Mexico", "South Africa",
    odds=(1.45, 4.60, 8.50), p_mkt=(0.6622, 0.2088, 0.1130),
    ts="2026-06-11T18:00:00+00:00", n_bookmakers=2)
print(f"\nupdate_market sobre partido existente: {ok}")
row = tracker.df.iloc[0]
print(f"  odds={row['odds_home']}/{row['odds_draw']}/{row['odds_away']}  "
      f"p_mkt={row['p_mkt_home']}/{row['p_mkt_draw']}/{row['p_mkt_away']}  "
      f"ts={row['odds_ts']}")
print(f"update_market sobre partido inexistente: "
      f"{tracker.update_market('Spain', 'Ghana', (2, 3, 4), (0.4, 0.3, 0.3), 'x', 1)}")
