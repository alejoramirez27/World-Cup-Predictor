"""Carga y preparación básica del dataset de resultados internacionales.

Dataset: martj42/international-football-results-from-1872-to-2017 (Kaggle),
actualizado hasta 2025/2026. CSVs esperados en data/raw/.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Agrupación de torneos en categorías útiles para el modelo
TOURNAMENT_TYPES = {
    "FIFA World Cup": "world_cup",
    "FIFA World Cup qualification": "wc_qualifier",
    "UEFA Euro": "continental",
    "UEFA Euro qualification": "continental_qualifier",
    "Copa América": "continental",
    "Copa América qualification": "continental_qualifier",
    "African Cup of Nations": "continental",
    "African Cup of Nations qualification": "continental_qualifier",
    "AFC Asian Cup": "continental",
    "AFC Asian Cup qualification": "continental_qualifier",
    "Gold Cup": "continental",
    "Gold Cup qualification": "continental_qualifier",
    "UEFA Nations League": "nations_league",
    "CONCACAF Nations League": "nations_league",
    "Friendly": "friendly",
}


def load_results(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga results.csv con fechas parseadas."""
    df = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    return df


def load_shootouts(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga shootouts.csv (definiciones por penales)."""
    return pd.read_csv(raw_dir / "shootouts.csv", parse_dates=["date"])


def load_goalscorers(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga goalscorers.csv (goleadores por partido)."""
    return pd.read_csv(raw_dir / "goalscorers.csv", parse_dates=["date"])


def filter_since(df: pd.DataFrame, year: int = 2010) -> pd.DataFrame:
    """Filtra partidos desde el año indicado (inclusive) y descarta
    partidos sin marcador (fixtures futuros aún no jugados)."""
    out = df[df["date"].dt.year >= year].copy()
    out = out.dropna(subset=["home_score", "away_score"])
    out[["home_score", "away_score"]] = out[["home_score", "away_score"]].astype(int)
    return out.reset_index(drop=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features básicas:

    - neutral como bool explícito y home_advantage (local real, no sede neutral)
    - tournament_type: categoría del torneo (world_cup, friendly, etc.)
    - is_competitive: todo lo que no sea amistoso
    - goal_diff y outcome desde la perspectiva del equipo local
    - year para splits temporales
    """
    out = df.copy()

    out["neutral"] = out["neutral"].astype(bool)
    out["home_advantage"] = ~out["neutral"]

    out["tournament_type"] = (
        out["tournament"].map(TOURNAMENT_TYPES).fillna("other")
    )
    out["is_competitive"] = out["tournament_type"] != "friendly"

    out["goal_diff"] = out["home_score"] - out["away_score"]
    out["outcome"] = pd.cut(
        out["goal_diff"], bins=[-100, -1, 0, 100], labels=["away_win", "draw", "home_win"]
    )

    out["year"] = out["date"].dt.year
    return out


def load_dataset(since: int = 2010, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Pipeline completo: carga, filtra desde `since` y agrega features."""
    return add_basic_features(filter_since(load_results(raw_dir), since))


if __name__ == "__main__":
    df = load_dataset()
    print(f"{len(df)} partidos desde {df['year'].min()} hasta {df['year'].max()}")
    print(df["tournament_type"].value_counts())
    print(df["outcome"].value_counts(normalize=True).round(3))
    print(df.head())
