"""Ingesta de cuotas 1X2 desde The Odds API para el Mundial 2026.

Por partido del fixture: cuotas de todas las casas disponibles, mediana
entre casas, probabilidades implícitas y de-vig por normalización. Guarda:
 - data/processed/odds_snapshots.csv: TODOS los snapshots (histórico, se
   appendea en cada corrida — permite reconstruir la línea de cierre)
 - tracking (live_tracking_2026.csv): el último snapshot por partido
   (cuotas crudas mediana, probabilidades de-vigged, timestamp)

Uso (correr manualmente ~2 veces al día; el snapshot justo antes del
partido es la línea de cierre, la que vale para evaluación):
    python src/odds_fetcher.py              # snapshot + actualizar tracking
    python src/odds_fetcher.py --register   # además congela predicción del
                                            # modelo para partidos aún no
                                            # registrados en el tracking

Requiere ODDS_API_KEY en .env (https://the-odds-api.com, plan free alcanza:
cada corrida gasta ~2 créditos de 500/mes).

Para recordatorio dos veces al día (opcional, sigue siendo corrida manual
la primera vez que pide permisos):
    schtasks /create /tn wc26_odds_am /sc daily /st 09:00 /tr "C:\\Users\\alejo\\worldcup-predictor\\.venv\\Scripts\\python.exe C:\\Users\\alejo\\worldcup-predictor\\src\\odds_fetcher.py"
    schtasks /create /tn wc26_odds_pm /sc daily /st 17:00 /tr "..."
"""

import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import truststore

truststore.inject_into_ssl()  # certificados del sistema (Windows)

import requests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
SNAPSHOTS_CSV = ROOT / "data" / "processed" / "odds_snapshots.csv"
RAW_RESULTS = ROOT / "data" / "raw" / "results.csv"

BASE_URL = "https://api.the-odds-api.com/v4"
REGIONS = "eu,uk"   # costo API = mercados x regiones (2 créditos por corrida)
MARKETS = "h2h"

# variantes de nombre que usa la API -> nombre del dataset
API_ALIASES = {
    "usa": "United States", "united states of america": "United States",
    "south korea republic": "South Korea", "korea republic": "South Korea",
    "ir iran": "Iran", "iran islamic republic of": "Iran",
    "turkiye": "Turkey", "czechia": "Czech Republic",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "cote divoire": "Ivory Coast", "cote d'ivoire": "Ivory Coast",
    "cabo verde": "Cape Verde", "dr congo": "DR Congo",
    "congo dr": "DR Congo", "ireland": "Republic of Ireland",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def load_api_key() -> str:
    import os
    key = os.environ.get("ODDS_API_KEY", "")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit(
            "Falta la API key: pega tu key de https://the-odds-api.com en "
            f"{ENV_FILE} (ODDS_API_KEY=...) y vuelve a correr."
        )
    return key


def find_sport_key(api_key: str) -> str:
    r = requests.get(f"{BASE_URL}/sports",
                     params={"apiKey": api_key}, timeout=30)
    r.raise_for_status()
    for s in r.json():
        if "world cup" in s["title"].lower() and s["group"] == "Soccer":
            return s["key"]
    return "soccer_fifa_world_cup"  # fallback al key histórico


def fetch_events(api_key: str, sport_key: str) -> tuple[list, str]:
    r = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS,
                "oddsFormat": "decimal"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining", "?")


def median_devig(event: dict) -> dict | None:
    """Mediana de cuotas entre casas + probabilidades sin margen."""
    prices: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market["key"] != "h2h":
                continue
            row = {}
            for out in market["outcomes"]:
                if out["name"] == event["home_team"]:
                    row["home"] = out["price"]
                elif out["name"] == event["away_team"]:
                    row["away"] = out["price"]
                elif out["name"] == "Draw":
                    row["draw"] = out["price"]
            if len(row) == 3:  # solo casas con el mercado completo
                for k, v in row.items():
                    prices[k].append(v)
    n = len(prices["home"])
    if n == 0:
        return None
    odds = {k: float(np.median(v)) for k, v in prices.items()}
    implied = np.array([1.0 / odds[k] for k in ("home", "draw", "away")])
    devig = implied / implied.sum()
    return {"odds": odds, "p_mkt": devig, "n_bookmakers": n,
            "margin": float(implied.sum() - 1.0)}


def team_resolver() -> dict[str, str]:
    """Mapa nombre-normalizado -> nombre dataset, para los 48 clasificados
    (lee el fixture directo del CSV, sin cargar el modelo)."""
    df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"].dt.year == 2026)]
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    res = {_norm(t): t for t in teams}
    res.update({_norm(k): v for k, v in API_ALIASES.items()})
    return res


def main() -> None:
    register = "--register" in sys.argv
    api_key = load_api_key()
    sport_key = find_sport_key(api_key)
    events, remaining = fetch_events(api_key, sport_key)
    print(f"sport: {sport_key} | eventos con odds: {len(events)} | "
          f"créditos restantes: {remaining}")
    if not events:
        print("Sin eventos (¿mercados aún no abiertos?). Nada que guardar.")
        return

    resolver = team_resolver()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    from live_tracking import LiveTracker
    tracker = LiveTracker()

    snapshots, updated, missing = [], 0, []
    for ev in events:
        home = resolver.get(_norm(ev["home_team"]))
        away = resolver.get(_norm(ev["away_team"]))
        if not home or not away:
            print(f"  AVISO: no pude mapear '{ev['home_team']}' vs "
                  f"'{ev['away_team']}' a equipos del dataset")
            continue
        m = median_devig(ev)
        if m is None:
            continue
        snapshots.append({
            "ts": ts, "commence": ev["commence_time"],
            "home_team": home, "away_team": away,
            "odds_home": m["odds"]["home"], "odds_draw": m["odds"]["draw"],
            "odds_away": m["odds"]["away"],
            "p_mkt_home": round(m["p_mkt"][0], 4),
            "p_mkt_draw": round(m["p_mkt"][1], 4),
            "p_mkt_away": round(m["p_mkt"][2], 4),
            "n_bookmakers": m["n_bookmakers"],
            "margin": round(m["margin"], 4),
        })
        odds3 = (m["odds"]["home"], m["odds"]["draw"], m["odds"]["away"])
        pmkt3 = tuple(round(p, 4) for p in m["p_mkt"])
        ok = tracker.update_market(home, away, odds3, pmkt3, ts,
                                   m["n_bookmakers"])
        if ok:
            updated += 1
        else:
            missing.append((home, away, odds3, pmkt3, m["n_bookmakers"]))

    if snapshots:
        snap_df = pd.DataFrame(snapshots)
        header = not SNAPSHOTS_CSV.exists()
        SNAPSHOTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        snap_df.to_csv(SNAPSHOTS_CSV, mode="a", header=header, index=False)
        print(f"snapshot guardado: {len(snapshots)} partidos -> "
              f"{SNAPSHOTS_CSV.name}")

    print(f"tracking actualizado: {updated} partidos con odds")
    if missing:
        if register:
            print(f"congelando predicción para {len(missing)} partidos "
                  f"nuevos (carga el modelo, tarda un poco)...")
            for home, away, odds3, pmkt3, n in missing:
                tracker.record_prediction(home, away, odds3)
                tracker.update_market(home, away, odds3, pmkt3, ts, n)
        else:
            print(f"{len(missing)} partidos con odds aún sin predicción "
                  f"congelada en el tracking (corre con --register para "
                  f"congelarlas ahora):")
            for home, away, *_ in missing:
                print(f"    {home} vs {away}")


if __name__ == "__main__":
    main()
