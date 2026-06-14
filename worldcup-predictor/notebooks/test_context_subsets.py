"""Sub-variantes de contexto: ¿alguna parte del paquete aporta sola?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (FEATURE_COLS, TRAIN_END, TRAIN_START, estimate_rho,
                   evaluate, prepare_data, train_models)

df, builder = prepare_data()
train = df[df["year"].between(TRAIN_START, TRAIN_END)]
cats = df["match_type"].astype("category").cat.categories

variants = {
    "solo_altitud": FEATURE_COLS + ["home_alt_diff", "away_alt_diff"],
    "alt+descanso": FEATURE_COLS + ["home_alt_diff", "away_alt_diff",
                                    "home_rest_days", "away_rest_days"],
}
for tag, cols in variants.items():
    models = train_models(df, feature_cols=cols)
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    print(f"{tag:<14} ll={ev['log_loss']:.4f} brier={ev['brier_score']:.4f} "
          f"acc={ev['accuracy_1x2']:.4f} ll_marc={ev['scoreline_log_loss']:.4f} "
          f"top3={ev['top3_scoreline']:.4f}")
