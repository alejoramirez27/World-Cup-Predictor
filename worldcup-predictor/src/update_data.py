"""Actualiza data/raw con la última versión del dataset de Kaggle (que el
autor actualiza durante el Mundial con los resultados nuevos) y reentrena
los modelos.

Uso:
    python src/update_data.py            # descarga + reentrena
    python src/update_data.py --no-train # solo descarga
"""

import shutil
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # certificados del sistema (evita error SSL en Windows)

import kagglehub  # noqa: E402  (necesita el inject antes del import)
import pandas as pd  # noqa: E402

DATASET = "martj42/international-football-results-from-1872-to-2017"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _snapshot() -> dict:
    """Resumen del estado actual de results.csv para reportar el delta."""
    path = RAW_DIR / "results.csv"
    if not path.exists():
        return {"n": 0, "last_played": None, "wc26_played": 0}
    df = pd.read_csv(path, parse_dates=["date"])
    played = df.dropna(subset=["home_score"])
    wc26 = played[(played["tournament"] == "FIFA World Cup")
                  & (played["date"].dt.year == 2026)]
    return {
        "n": len(df),
        "last_played": played["date"].max(),
        "wc26_played": len(wc26),
    }


def update_dataset() -> None:
    before = _snapshot()
    print("Descargando última versión del dataset...")
    path = kagglehub.dataset_download(DATASET, force_download=True)
    for f in Path(path).glob("*.csv"):
        shutil.copy2(f, RAW_DIR)
    after = _snapshot()
    print(f"  filas: {before['n']} -> {after['n']} "
          f"(+{after['n'] - before['n']})")
    print(f"  último partido jugado: {before['last_played']} -> "
          f"{after['last_played']}")
    print(f"  partidos del Mundial 2026 con resultado: "
          f"{before['wc26_played']} -> {after['wc26_played']}")


def retrain() -> None:
    from model import (TRAIN_END, TRAIN_START, estimate_rho, prepare_data,
                       save_models, train_models)

    print(f"\nReentrenando modelos (warm-up Elo 1990, features desde {TRAIN_START})...")
    df, builder = prepare_data()
    models = train_models(df)
    save_models(models)
    cats = df["match_type"].astype("category").cat.categories
    rho = estimate_rho(models, df[df["year"].between(TRAIN_START, TRAIN_END)], cats)
    m = models["metrics"]
    print(f"  validación: MAE {m['mae_home']:.3f}/{m['mae_away']:.3f}, "
          f"accuracy 1X2 {m['accuracy_1x2']:.4f}, rho DC {rho:+.4f}")
    print("Modelos guardados. worldcup_2026.py usará automáticamente los "
          "resultados nuevos (Elo y forma se recalculan en cada corrida).")


if __name__ == "__main__":
    update_dataset()
    final_dir = Path(__file__).resolve().parent.parent / "models" / "model_final_wc2026"
    if final_dir.exists() and "--force-retrain" not in sys.argv:
        print("\nPOLITICA DE TORNEO: existe el modelo final congelado "
              "(models/model_final_wc2026), no se reentrena durante el "
              "Mundial. Las features (Elo, forma, descanso) se recalculan "
              "automáticamente con los resultados nuevos en cada corrida. "
              "Usa --force-retrain solo si de verdad quieres romper el "
              "congelamiento.")
    elif "--no-train" not in sys.argv:
        retrain()
