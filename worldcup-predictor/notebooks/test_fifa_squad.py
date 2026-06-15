"""¿La fuerza de plantilla (rating FIFA por nacionalidad) afina el modelo?
Construye strength[(nacionalidad, año)] = media overall de los 23 mejores,
la une a los partidos (con forward-fill de año) y mide en validación. NO toca
el modelo congelado."""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import truststore

truststore.inject_into_ssl()
import kagglehub  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,  # noqa: E402
                   evaluate, prepare_data, train_models)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


# FIFA -> nombre del dataset (normalizados)
ALIAS = {"korea republic": "south korea", "korea dpr": "north korea",
         "cote d'ivoire": "ivory coast", "czechia": "czech republic",
         "cabo verde": "cape verde", "congo dr": "dr congo",
         "china pr": "china pr", "ir iran": "iran"}


def _cols(df: pd.DataFrame):
    ovr = next((c for c in df.columns if c.lower() in ("overall", "ovr", "rating")), None)
    nat = next((c for c in df.columns if c.lower() in
                ("nationality_name", "nationality", "nation", "country")), None)
    return ovr, nat


def top23(df: pd.DataFrame, natcol: str) -> pd.Series:
    df = df[df["overall"].notna()]
    top = df.sort_values("overall", ascending=False).groupby(natcol).head(23)
    s = top.groupby(natcol)["overall"].mean()
    s.index = [ALIAS.get(norm(n), norm(n)) for n in s.index]
    return s.groupby(level=0).mean()


def _strength_from(df: pd.DataFrame) -> pd.Series:
    ovr, nat = _cols(df)
    df = df.rename(columns={ovr: "overall"})
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    return top23(df, nat)


def load_strength() -> dict[int, pd.Series]:
    out = {}
    d1 = Path(kagglehub.dataset_download("stefanoleone992/fifa-22-complete-player-dataset"))
    for f in d1.glob("players_*.csv"):  # solo masculino
        yr = 2000 + int(re.search(r"players_(\d+)", f.name).group(1))
        out[yr] = _strength_from(pd.read_csv(f, low_memory=False))
    for slug, yr in [("stefanoleone992/ea-sports-fc-24-complete-player-dataset", 2024),
                     ("nyagami/ea-sports-fc-25-database-ratings-and-stats", 2025)]:
        d = Path(kagglehub.dataset_download(slug))
        f = next(d.glob("male_players.csv"))
        out[yr] = _strength_from(pd.read_csv(f, low_memory=False))
    return out


def main():
    strength = load_strength()
    avail = sorted(strength)
    print(f"años FIFA: {avail}")

    def pick(y):
        c = [a for a in avail if a <= y]
        return c[-1] if c else None

    def squad(team, year):
        yy = pick(year)
        if yy is None:
            return float("nan")
        return float(strength[yy].get(norm(team), float("nan")))

    df, _ = prepare_data()
    df["home_squad"] = [squad(t, y) for t, y in zip(df["home_team"], df["year"])]
    df["away_squad"] = [squad(t, y) for t, y in zip(df["away_team"], df["year"])]
    df["squad_diff"] = df["home_squad"] - df["away_squad"]

    val = df[df["year"].between(2024, 2025)]
    cov = val[["home_squad", "away_squad"]].notna().all(axis=1).mean()
    print(f"cobertura plantilla en validación 2024-25: {cov:.1%}")

    cats = df["match_type"].astype("category").cat.categories
    train = df[df["year"].between(TRAIN_START, TRAIN_END)]
    SQUAD = ["home_squad", "away_squad", "squad_diff"]
    print(f"\n{'variante':<14}{'log-loss':>10}{'brier':>9}{'acc':>8}{'ll_marc':>9}")
    for tag, cols in [("baseline", FEATURE_COLS), ("+plantilla", FEATURE_COLS + SQUAD)]:
        models = train_models(df, feature_cols=cols)
        rho = estimate_rho(models, train, cats)
        ev = evaluate(models, df, rho=rho)
        print(f"{tag:<14}{ev['log_loss']:>10.4f}{ev['brier_score']:>9.4f}"
              f"{ev['accuracy_1x2']:>8.4f}{ev['scoreline_log_loss']:>9.4f}")


if __name__ == "__main__":
    main()
