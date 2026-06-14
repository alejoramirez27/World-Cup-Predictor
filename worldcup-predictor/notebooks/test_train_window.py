"""¿Entrenar con más años de historia fortalece el modelo? Compara la ventana
de entrenamiento (start year) midiendo en validación 2024-2025. Warm-up de Elo
siempre desde 1990; validación fija. NO toca el modelo congelado."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model
from data_loader import add_basic_features, filter_since, load_results
from features import FeatureBuilder
from model import estimate_rho, evaluate, train_models


def prep(start: int):
    full = add_basic_features(filter_since(load_results(), 1990))
    cutoff = pd.Timestamp(f"{start}-01-01")
    b = FeatureBuilder()
    b.warm_up(full[full["date"] < cutoff])
    df = b.transform(full[full["date"] >= cutoff].reset_index(drop=True))
    return df


print(f"{'start':>6}{'n_train':>9}{'log-loss':>10}{'brier':>9}{'acc':>8}{'ll_marc':>9}")
for start in (2010, 2006, 2002, 1998):
    model.TRAIN_START = start          # train_models lee este global
    df = prep(start)
    models = train_models(df)
    train = df[df["year"].between(start, 2023)]
    cats = df["match_type"].astype("category").cat.categories
    rho = estimate_rho(models, train, cats)
    ev = evaluate(models, df, rho=rho)
    print(f"{start:>6}{len(train):>9}{ev['log_loss']:>10.4f}{ev['brier_score']:>9.4f}"
          f"{ev['accuracy_1x2']:>8.4f}{ev['scoreline_log_loss']:>9.4f}")
