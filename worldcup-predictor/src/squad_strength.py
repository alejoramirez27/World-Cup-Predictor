"""Fuerza de plantilla por selección y año, desde ratings FIFA/FC.

strength[(equipo, año)] = media del overall de los 23 mejores jugadores de esa
nacionalidad. Cubre 2015-2025 (FIFA 15-22 + FC24 + FC25); con forward-fill se
aplica a cualquier año (los anteriores a 2015 quedan sin dato = NaN).

`build_table()` descarga los datasets (una vez) y cachea a CSV; el resto del
pipeline solo lee el CSV, así la predicción no depende de la descarga.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path(__file__).resolve().parent.parent / "data" / "processed" / "squad_strength.csv"

# FIFA -> nombre del dataset de resultados (todo normalizado)
ALIAS = {
    "korea republic": "south korea", "korea dpr": "north korea",
    "cote d'ivoire": "ivory coast", "czechia": "czech republic",
    "cabo verde": "cape verde", "congo dr": "dr congo",
    "ir iran": "iran", "china pr": "china pr",
    "united states": "united states", "republic of ireland": "republic of ireland",
}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def _cols(df: pd.DataFrame):
    ovr = next((c for c in df.columns if c.lower() in ("overall", "ovr", "rating")), None)
    nat = next((c for c in df.columns if c.lower() in
                ("nationality_name", "nationality", "nation", "country")), None)
    return ovr, nat


def _strength_from(df: pd.DataFrame) -> pd.Series:
    ovr, nat = _cols(df)
    df = df.rename(columns={ovr: "overall"})
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df[df["overall"].notna()]
    top = df.sort_values("overall", ascending=False).groupby(nat).head(23)
    s = top.groupby(nat)["overall"].mean()
    s.index = [ALIAS.get(norm(n), norm(n)) for n in s.index]
    return s.groupby(level=0).mean()


def build_table() -> pd.DataFrame:
    """Descarga FIFA/FC y construye la tabla larga (equipo, año, strength)."""
    import truststore

    truststore.inject_into_ssl()
    import kagglehub

    rows = []
    d1 = Path(kagglehub.dataset_download("stefanoleone992/fifa-22-complete-player-dataset"))
    for f in d1.glob("players_*.csv"):
        yr = 2000 + int(re.search(r"players_(\d+)", f.name).group(1))
        for team, val in _strength_from(pd.read_csv(f, low_memory=False)).items():
            rows.append((team, yr, round(float(val), 2)))
    for slug, yr in [("stefanoleone992/ea-sports-fc-24-complete-player-dataset", 2024),
                     ("nyagami/ea-sports-fc-25-database-ratings-and-stats", 2025)]:
        d = Path(kagglehub.dataset_download(slug))
        f = next(d.glob("male_players.csv"))
        for team, val in _strength_from(pd.read_csv(f, low_memory=False)).items():
            rows.append((team, yr, round(float(val), 2)))
    df = pd.DataFrame(rows, columns=["team", "year", "strength"])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE, index=False)
    return df


_CACHE: dict | None = None


def _get() -> dict:
    global _CACHE
    if _CACHE is None:
        df = pd.read_csv(TABLE) if TABLE.exists() else build_table()
        by_year: dict[int, dict[str, float]] = {}
        for t, y, s in zip(df["team"], df["year"], df["strength"]):
            by_year.setdefault(int(y), {})[t] = float(s)
        _CACHE = {"by_year": by_year, "years": sorted(by_year)}
    return _CACHE


def squad_for(team: str, year: int) -> float:
    """Fuerza de plantilla con forward-fill: usa el año FIFA más reciente <= año
    del partido. NaN si no hay (equipo sin match o año < 2015)."""
    c = _get()
    cands = [y for y in c["years"] if y <= year]
    if not cands:
        return np.nan
    return c["by_year"][cands[-1]].get(norm(team), np.nan)


def add_squad_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega home_squad, away_squad y squad_diff a un dataframe de partidos."""
    df = df.copy()
    df["home_squad"] = [squad_for(t, y) for t, y in zip(df["home_team"], df["year"])]
    df["away_squad"] = [squad_for(t, y) for t, y in zip(df["away_team"], df["year"])]
    df["squad_diff"] = df["home_squad"] - df["away_squad"]
    return df


if __name__ == "__main__":
    t = build_table()
    print(f"tabla construida: {len(t)} filas, años {sorted(t['year'].unique())}")
    for team in ("Brazil", "France", "Spain", "Argentina", "Mexico", "South Korea", "Morocco"):
        print(f"  {team} 2025: {squad_for(team, 2025)}")
