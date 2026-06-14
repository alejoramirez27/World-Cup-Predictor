"""Chequeo puntual: ¿hace falta temperature scaling después de Dixon-Coles?
Ajusta T en 2024 y mide el efecto en 2025. Si T ~ 1, la calibración está bien."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (TRAIN_END, TRAIN_START, apply_temperature, estimate_rho,
                   evaluate, fit_temperature, prepare_data, probs_metrics,
                   train_models)

df, builder = prepare_data()
models = train_models(df)
cats = df["match_type"].astype("category").cat.categories
rho = estimate_rho(models, df[df["year"].between(TRAIN_START, TRAIN_END)], cats)
ev = evaluate(models, df, rho=rho)

y, probs, years = ev["y"], ev["probs"], ev["years"]
m24, m25 = years == 2024, years == 2025
T = fit_temperature(probs[m24], y[m24])
print(f"rho = {rho:+.4f} | T ajustada en 2024: {T:.4f}")

met_dc = probs_metrics(probs[m25], y[m25])
met_t = probs_metrics(apply_temperature(probs[m25], T), y[m25])
print(f"2025 ({int(m25.sum())} partidos)")
print(f"  DC:    log-loss={met_dc['log_loss']:.4f}  brier={met_dc['brier_score']:.4f}  acc={met_dc['accuracy_1x2']:.4f}")
print(f"  DC+T:  log-loss={met_t['log_loss']:.4f}  brier={met_t['brier_score']:.4f}  acc={met_t['accuracy_1x2']:.4f}")
