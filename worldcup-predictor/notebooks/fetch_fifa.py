"""Sondeo de factibilidad: ¿conseguimos ratings FIFA por nacionalidad y por
año, incluyendo años recientes? Construye fuerza de plantilla = media del
overall de los 23 mejores jugadores por nacionalidad."""
import pathlib
import sys

import truststore

truststore.inject_into_ssl()
import kagglehub  # noqa: E402
import pandas as pd  # noqa: E402

# candidatos: multi-año 15-22, y datasets FC recientes
SLUGS = [
    "stefanoleone992/fifa-22-complete-player-dataset",
    "stefanoleone992/ea-sports-fc-24-complete-player-dataset",
    "nyagami/ea-sports-fc-25-database-ratings-and-stats",
]


def strength_table(df: pd.DataFrame, natcol: str) -> pd.Series:
    top = df.sort_values("overall", ascending=False).groupby(natcol).head(23)
    return top.groupby(natcol)["overall"].mean().sort_values(ascending=False)


for slug in SLUGS:
    print(f"\n=== {slug} ===")
    try:
        p = kagglehub.dataset_download(slug)
        csvs = sorted(pathlib.Path(p).glob("*.csv"))
        print(f"  {len(csvs)} csv:", [f.name for f in csvs][:12])
        # elige un csv con 'overall' y nacionalidad
        for f in csvs:
            try:
                df = pd.read_csv(f, nrows=5)
            except Exception:
                continue
            cols = [c.lower() for c in df.columns]
            natcol = next((c for c in df.columns if c.lower() in
                          ("nationality_name", "nationality", "nation")), None)
            ovrcol = next((c for c in df.columns if c.lower() == "overall"), None)
            if natcol and ovrcol:
                full = pd.read_csv(f)
                full = full.rename(columns={ovrcol: "overall"})
                st = strength_table(full, natcol)
                print(f"  [{f.name}] nat='{natcol}'  equipos={st.size}")
                for team in ("Brazil", "France", "Spain", "Argentina", "Mexico", "Morocco"):
                    if team in st.index:
                        print(f"     {team}: {st[team]:.1f}")
                break
    except Exception as e:
        print(f"  FALLO: {str(e)[:160]}")
