"""Sincroniza predicciones, resultados y métricas a Supabase (supabase-py).

La web pública lee con la ANON key (read-only vía RLS); este script escribe
con la SERVICE key (bypassa RLS). Esquema y políticas: sql/supabase_schema.sql.

Uso:
    python src/sync_supabase.py            # upsert real a Supabase
    python src/sync_supabase.py --dry-run  # arma payloads y reporta sin red
                                           # (vuelca un sample a reports/)

Requiere SUPABASE_URL y SUPABASE_SERVICE_KEY en .env (salvo --dry-run).
"""

import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:  # usa el almacén de certificados del sistema (evita errores SSL en Windows)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
META = ROOT / "models" / "model_final_wc2026" / "meta.json"
SAMPLE_OUT = ROOT / "reports" / "supabase_payload_sample.json"


def _env(key: str) -> str:
    import os
    val = os.environ.get(key, "")
    if not val and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                val = line.split("=", 1)[1].strip()
    return val


def make_match_id(d, home: str, away: str) -> str:
    """ID estable y determinista, compartido por predictions y results."""
    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return f"{pd.Timestamp(d).date().isoformat()}__{slug(home)}__vs__{slug(away)}"


def model_version() -> str:
    if META.exists():
        m = json.loads(META.read_text())
        return f"{m.get('version', 'final')}@{m.get('trained_through', '?')}"
    return "dev"


def _assemble(match_id, fecha, fase, home, away, probs, lam_h, lam_a, m, mv):
    """Arma una fila de predictions a partir de una matriz de marcadores."""
    from worldcup_2026 import match_personality, top_scorelines_from_matrix
    n = m.shape[0]
    over = 1.0 - float(sum(m[i, j] for i in range(n) for j in range(n)
                          if i + j < 2.5))
    pers = match_personality(m, probs)
    top5 = [{"score": s, "prob": round(p, 4)}
            for s, p in top_scorelines_from_matrix(m, 5)]
    return {
        "match_id": match_id, "fecha": fecha, "fase_grupo": fase,
        "equipo_home": home, "equipo_away": away,
        "prob_home": round(probs["home_win"], 4),
        "prob_draw": round(probs["draw"], 4),
        "prob_away": round(probs["away_win"], 4),
        "top_scorelines": top5,
        "poisson_matrix": np.round(m, 5).tolist(),
        "over25_prob": round(over, 4),
        "prob_goleada": round(pers["p_goleada_fav"], 4),
        "lambda_home": round(float(lam_h), 4),
        "lambda_away": round(float(lam_a), 4),
        "model_version": mv,
    }


def build_predictions(predictor, tracker) -> list[dict]:
    """Una fila por partido del fixture.

    Partidos YA JUGADOS: se usa la predicción CONGELADA del tracking (las
    prob/lambda pre-partido que evaluó results.log_loss); re-predecirlos
    sería leakage (el estado del builder ya incluye su resultado). La
    matriz/over25/goleada se reconstruyen de los lambdas congelados (Poisson
    +rho), por eso en jugados pueden diferir levemente del ensemble.
    Partidos PRÓXIMOS: predicción fresca completa del modelo.
    """
    from model import score_matrix_agg
    team_group = {t: g for g, teams in predictor.groups.items() for t in teams}
    mv = model_version()
    frozen = {make_match_id(r["date"], r["home_team"], r["away_team"]): r
              for _, r in tracker.df.iterrows() if r["outcome"]}
    rows = []
    for fx in predictor.fixtures.itertuples():
        mid = make_match_id(fx.date, fx.home_team, fx.away_team)
        fecha = pd.Timestamp(fx.date).date().isoformat()
        fase = team_group.get(fx.home_team)
        if mid in frozen:
            r = frozen[mid]
            probs = {"home_win": float(r["p_home"]), "draw": float(r["p_draw"]),
                     "away_win": float(r["p_away"])}
            lam_h, lam_a = float(r["lambda_home"]), float(r["lambda_away"])
            m = score_matrix_agg(lam_h, lam_a, rho=predictor.rho)
            rows.append(_assemble(mid, fecha, fase, r["home_team"],
                                  r["away_team"], probs, lam_h, lam_a, m, mv))
        else:
            p = predictor.predict(fx.home_team, fx.away_team)
            rows.append(_assemble(mid, fecha, fase, p["home"], p["away"],
                                  p["outcome_probs"], p["lambda_home"],
                                  p["lambda_away"], np.asarray(p["score_matrix"]),
                                  mv))
    return rows


def build_results(tracker) -> list[dict]:
    """Una fila por partido ya jugado en el tracking."""
    from live_tracking import hit_1x2
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    played = tracker.df[tracker.df["outcome"] != ""]
    rows = []
    for _, r in played.iterrows():
        rows.append({
            "match_id": make_match_id(r["date"], r["home_team"], r["away_team"]),
            "goles_home": int(r["home_score"]),
            "goles_away": int(r["away_score"]),
            "log_loss_partido": round(float(r["logloss_model"]), 4),
            "acierto_1x2": bool(hit_1x2(r)),
            "registrado_at": ts,
        })
    return rows


def build_metrics(tracker) -> dict | None:
    """Snapshot acumulado del modelo (un registro por día)."""
    played = tracker.df[tracker.df["outcome"] != ""]
    if played.empty:
        return None
    y = played["outcome"].map({"home_win": 0, "draw": 1, "away_win": 2}).values
    p = played[["p_home", "p_draw", "p_away"]].values
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    acc = float((p.argmax(axis=1) == y).mean())
    return {
        "fecha": date.today().isoformat(),
        "log_loss_acumulado": round(float(played["logloss_model"].mean()), 4),
        "brier_acumulado": round(brier, 4),
        "accuracy_1x2": round(acc, 4),
        "partidos_evaluados": int(len(played)),
        "log_loss_azar": round(float(np.log(3)), 4),
    }


def main() -> None:
    dry = "--dry-run" in sys.argv

    print("Cargando modelo y tracking...")
    from live_tracking import LiveTracker
    from worldcup_2026 import WorldCup2026Predictor
    predictor = WorldCup2026Predictor()
    tracker = LiveTracker()

    preds = build_predictions(predictor, tracker)
    results = build_results(tracker)
    metrics = build_metrics(tracker)
    print(f"payloads: {len(preds)} predicciones | {len(results)} resultados | "
          f"{1 if metrics else 0} snapshot de métricas")

    if dry:
        SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_OUT.write_text(json.dumps(
            {"predictions_sample": preds[:2], "results": results,
             "metrics": metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[DRY-RUN] sin red. Sample -> {SAMPLE_OUT}")
        if metrics:
            print(f"[DRY-RUN] métricas: ll_acum={metrics['log_loss_acumulado']} "
                  f"brier={metrics['brier_acumulado']} acc={metrics['accuracy_1x2']} "
                  f"n={metrics['partidos_evaluados']}")
        return

    url, key = _env("SUPABASE_URL"), _env("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise SystemExit(
            "Faltan credenciales: pega SUPABASE_URL y SUPABASE_SERVICE_KEY en "
            f"{ENV_FILE}. (Usa --dry-run para probar sin red.)")

    from supabase import create_client
    client = create_client(url, key)
    try:  # verificar conexión
        client.table("predictions").select("match_id").limit(1).execute()
    except Exception as exc:
        raise SystemExit(
            f"No pude conectar / tabla 'predictions' inexistente ({exc}). "
            f"¿Ejecutaste sql/supabase_schema.sql en el proyecto?") from exc
    print(f"conectado a {url}")

    n_pred = len(client.table("predictions").upsert(
        preds, on_conflict="match_id").execute().data)
    n_res = (len(client.table("results").upsert(
        results, on_conflict="match_id").execute().data) if results else 0)
    n_met = (len(client.table("model_metrics").upsert(
        [metrics], on_conflict="fecha").execute().data) if metrics else 0)

    print(f"\nupsert OK:")
    print(f"  predictions:   {n_pred} filas")
    print(f"  results:       {n_res} filas")
    print(f"  model_metrics: {n_met} filas")


if __name__ == "__main__":
    main()
