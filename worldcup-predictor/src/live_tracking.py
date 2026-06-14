"""Registro en vivo del Mundial 2026: predicción congelada del modelo,
odds de mercado (opcionales) y resultado real por partido, con log-loss
acumulado del torneo.

La predicción se genera con el estado actual del dataset y queda CONGELADA
en el CSV — registra el partido ANTES de correr update_data.py, para que la
predicción sea genuinamente pre-partido.

Uso CLI:
    python src/live_tracking.py predict "Mexico" "South Africa"
    python src/live_tracking.py predict "Mexico" "South Africa" --odds 1.45 4.4 8.0
    python src/live_tracking.py result "Mexico" "South Africa" 2 0
    python src/live_tracking.py report
"""

import sys
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_CSV = (Path(__file__).resolve().parent.parent
             / "data" / "processed" / "live_tracking_2026.csv")

COLUMNS = [
    "date", "home_team", "away_team",
    "lambda_home", "lambda_away", "p_home", "p_draw", "p_away",
    "top_scorelines",                       # "2-0:0.157|1-0:0.133|3-0:0.121"
    "odds_home", "odds_draw", "odds_away",  # mediana entre casas (crudas)
    "p_mkt_home", "p_mkt_draw", "p_mkt_away",  # de-vigged (sin margen)
    "odds_ts", "n_bookmakers",              # timestamp y # de casas
    "home_score", "away_score", "outcome",
    "logloss_model", "logloss_market",
]


BLEND_W_DEFAULT = 0.3       # peso del modelo mientras no haya datos
MIN_MATCHES_FOR_W = 8       # mínimo de partidos con odds+resultado para optimizar w
VALUE_WARN_PP = 0.10        # discrepancia que dispara el warning


def _outcome(hs: float, as_: float) -> str:
    if hs > as_:
        return "home_win"
    if hs == as_:
        return "draw"
    return "away_win"


def _market_probs(row: pd.Series) -> np.ndarray | None:
    """Probabilidades implícitas del mercado, sin el margen de la casa."""
    odds = [row["odds_home"], row["odds_draw"], row["odds_away"]]
    if any(pd.isna(o) for o in odds):
        return None
    inv = np.array([1.0 / o for o in odds])
    return inv / inv.sum()


DRAW_LL = log(3)  # 1.0986 — log-loss del azar uniforme 1X2
GRADE_GOOD_LL = 0.70


def grade_match(logloss) -> dict:
    """Grado de tres niveles según el log-loss del partido (la métrica
    honesta, no el acierto binario):
      ACIERTO (verde)  log-loss < 0.70      -> convicción correcta
      FLOJO   (amarillo) 0.70 <= ll <= 1.0986 -> poca convicción
      FALLO   (rojo)   log-loss > 1.0986    -> peor que el azar
    """
    if logloss is None or pd.isna(logloss):
        return {"label": "", "level": "none", "color": "dim"}
    if logloss < GRADE_GOOD_LL:
        return {"label": "ACIERTO", "level": "good", "color": "green"}
    if logloss <= DRAW_LL:
        return {"label": "FLOJO", "level": "weak", "color": "yellow"}
    return {"label": "FALLO", "level": "bad", "color": "red"}


def hit_1x2(row: pd.Series) -> bool | None:
    """Acierto 1X2 puro: ¿el resultado más probable coincidió? (None si el
    partido no se ha jugado)."""
    if not row["outcome"]:
        return None
    pred = max([("home_win", row["p_home"]), ("draw", row["p_draw"]),
                ("away_win", row["p_away"])], key=lambda t: t[1])[0]
    return pred == row["outcome"]


class LiveTracker:
    def __init__(self) -> None:
        self.df = pd.DataFrame(columns=COLUMNS)
        if TRACK_CSV.exists():
            try:
                self.df = pd.read_csv(TRACK_CSV).fillna(
                    {"outcome": "", "top_scorelines": ""})
            except Exception as exc:  # archivo corrupto: no pisarlo en silencio
                raise RuntimeError(
                    f"No se pudo leer {TRACK_CSV} ({exc}). Revísalo o "
                    f"bórralo manualmente antes de continuar."
                ) from exc
            for col in COLUMNS:  # compatibilidad con CSVs previos
                if col not in self.df.columns:
                    self.df[col] = np.nan
            self.df = self.df[COLUMNS]
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from worldcup_2026 import WorldCup2026Predictor
            self._predictor = WorldCup2026Predictor()
        return self._predictor

    def _save(self) -> None:
        """Escritura atómica: temporal + replace, para que una interrupción
        a mitad de escritura no deje el registro corrupto."""
        TRACK_CSV.parent.mkdir(parents=True, exist_ok=True)
        tmp = TRACK_CSV.with_suffix(".csv.tmp")
        self.df.to_csv(tmp, index=False)
        tmp.replace(TRACK_CSV)

    def _find(self, home: str, away: str) -> int | None:
        hit = self.df[(self.df["home_team"] == home)
                      & (self.df["away_team"] == away)]
        return int(hit.index[0]) if len(hit) else None

    def record_prediction(self, team_a: str, team_b: str,
                          odds: tuple[float, float, float] | None = None) -> pd.Series:
        """Genera y congela la predicción del modelo para un partido.
        Si ya estaba registrada, no la regenera (solo actualiza odds)."""
        pred = self.predictor.predict(team_a, team_b)
        home, away = pred["home"], pred["away"]
        idx = self._find(home, away)
        if idx is not None:
            if odds:
                self.df.loc[idx, ["odds_home", "odds_draw", "odds_away"]] = odds
                self._save()
                print(f"{home} vs {away}: ya registrado, odds actualizadas.")
            else:
                print(f"{home} vs {away}: ya registrado, predicción intacta.")
            return self.df.loc[idx]

    # fecha del fixture si existe; si no, hoy
        fixture = self.predictor.find_fixture(home, away)
        date = (fixture["date"].date() if fixture is not None
                else pd.Timestamp.today().date())
        probs = pred["outcome_probs"]
        top = "|".join(f"{s}:{p:.3f}" for s, p in pred["top_scorelines"])
        row = {
            "date": str(date), "home_team": home, "away_team": away,
            "lambda_home": round(pred["lambda_home"], 3),
            "lambda_away": round(pred["lambda_away"], 3),
            "p_home": round(probs["home_win"], 4),
            "p_draw": round(probs["draw"], 4),
            "p_away": round(probs["away_win"], 4),
            "top_scorelines": top,
            "odds_home": odds[0] if odds else np.nan,
            "odds_draw": odds[1] if odds else np.nan,
            "odds_away": odds[2] if odds else np.nan,
            "home_score": np.nan, "away_score": np.nan, "outcome": "",
            "logloss_model": np.nan, "logloss_market": np.nan,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        print(f"Predicción congelada: {home} {probs['home_win']:.0%} / "
              f"empate {probs['draw']:.0%} / {away} {probs['away_win']:.0%}  "
              f"[{top}]")
        return self.df.iloc[-1]

    def update_market(self, home: str, away: str,
                      odds: tuple[float, float, float],
                      p_mkt: tuple[float, float, float],
                      ts: str, n_bookmakers: int) -> bool:
        """Actualiza las columnas de mercado de un partido ya registrado
        (nombres en formato dataset, orden local/visitante del fixture).
        Cada corrida pisa la anterior: el último snapshot antes del partido
        son las cuotas de cierre. Devuelve False si el partido no está."""
        idx = self._find(home, away)
        if idx is None:
            return False
        self.df.loc[idx, ["odds_home", "odds_draw", "odds_away"]] = odds
        self.df.loc[idx, ["p_mkt_home", "p_mkt_draw", "p_mkt_away"]] = p_mkt
        if self.df["odds_ts"].dtype != object:  # creada como float (NaN)
            self.df["odds_ts"] = self.df["odds_ts"].astype(object)
        self.df.loc[idx, "odds_ts"] = ts
        self.df.loc[idx, "n_bookmakers"] = float(n_bookmakers)
        self._save()
        return True

    def add_result(self, team_a: str, team_b: str, score_a: int, score_b: int,
                   odds: tuple[float, float, float] | None = None) -> None:
        """Registra el resultado real (marcador en el orden de los equipos
        tal como se pasan). Si no había predicción congelada, la genera
        primero — válido solo si data/raw aún no incluye este resultado."""
        a = self.predictor.resolve_team(team_a)
        b = self.predictor.resolve_team(team_b)
        # normalizar al orden local/visitante del fixture
        fixture = self.predictor.find_fixture(a, b)
        if fixture is not None and fixture["home_team"] == b:
            a, b = b, a
            score_a, score_b = score_b, score_a

        if self._find(a, b) is None:
            self.record_prediction(a, b, odds)
        idx = self._find(a, b)

        row = self.df.loc[idx]
        out = _outcome(score_a, score_b)
        p_model = {"home_win": row["p_home"], "draw": row["p_draw"],
                   "away_win": row["p_away"]}[out]
        self.df.loc[idx, ["home_score", "away_score", "outcome"]] = \
            [score_a, score_b, out]
        self.df.loc[idx, "logloss_model"] = -log(max(p_model, 1e-12))
        market = _market_probs(row)
        if market is not None:
            p_mkt = market[["home_win", "draw", "away_win"].index(out)]
            self.df.loc[idx, "logloss_market"] = -log(max(p_mkt, 1e-12))
        self._save()
        print(f"Resultado registrado: {a} {score_a}-{score_b} {b} ({out}). "
              f"Log-loss del partido: {self.df.loc[idx, 'logloss_model']:.4f}")

    # ------------------------------------------------- blend modelo+mercado ---

    def _played_with_market(self) -> pd.DataFrame:
        return self.df[(self.df["outcome"] != "")
                       & self.df["p_mkt_home"].notna()]

    def blend_weight(self) -> tuple[float, str]:
        """w del blend prob = w*modelo + (1-w)*mercado, optimizado por
        log-loss sobre los partidos del torneo ya jugados con odds.
        Con menos de MIN_MATCHES_FOR_W usa el default 0.3."""
        d = self._played_with_market()
        if len(d) < MIN_MATCHES_FOR_W:
            return BLEND_W_DEFAULT, (f"default (solo {len(d)} partidos con "
                                     f"odds y resultado; se optimiza con "
                                     f">={MIN_MATCHES_FOR_W})")
        pm = d[["p_home", "p_draw", "p_away"]].values
        pk = d[["p_mkt_home", "p_mkt_draw", "p_mkt_away"]].values
        y = d["outcome"].map({"home_win": 0, "draw": 1, "away_win": 2}).values
        grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
        lls = [
            -np.log(np.maximum(
                (w * pm + (1 - w) * pk)[np.arange(len(y)), y], 1e-12)).mean()
            for w in grid
        ]
        w = float(grid[int(np.argmin(lls))])
        return w, f"optimizado sobre {len(d)} partidos"

    def metrics(self) -> dict:
        """Log-loss acumulado: modelo (todos los jugados) y, sobre el
        subconjunto con odds, modelo vs mercado vs blend."""
        played = self.df[self.df["outcome"] != ""]
        out = {"n_played": len(played),
               "ll_model": (float(played["logloss_model"].mean())
                            if len(played) else None)}
        d = self._played_with_market()
        w, w_src = self.blend_weight()
        out.update({"blend_w": w, "blend_w_source": w_src, "n_market": len(d)})
        if len(d):
            pm = d[["p_home", "p_draw", "p_away"]].values
            pk = d[["p_mkt_home", "p_mkt_draw", "p_mkt_away"]].values
            y = d["outcome"].map({"home_win": 0, "draw": 1, "away_win": 2}).values
            ix = np.arange(len(y))

            def ll(p):
                return float(-np.log(np.maximum(p[ix, y], 1e-12)).mean())

            out.update({"ll_model_subset": ll(pm), "ll_market": ll(pk),
                        "ll_blend": ll(w * pm + (1 - w) * pk)})
        return out

    # ------------------------------------------------- detector de valor ---

    def value_table(self) -> pd.DataFrame:
        """Partidos próximos (sin resultado, con odds): modelo vs mercado
        por resultado, EV de apostar cada lado a las odds de mercado, y
        warning si la discrepancia supera VALUE_WARN_PP."""
        up = self.df[(self.df["outcome"] == "") & self.df["p_mkt_home"].notna()]
        rows = []
        for _, r in up.iterrows():
            sides = [("local", r["home_team"], "p_home", "p_mkt_home", "odds_home"),
                     ("empate", "empate", "p_draw", "p_mkt_draw", "odds_draw"),
                     ("visita", r["away_team"], "p_away", "p_mkt_away", "odds_away")]
            for side, label, pm_c, pk_c, od_c in sides:
                pm, pk, od = r[pm_c], r[pk_c], r[od_c]
                rows.append({
                    "date": r["date"],
                    "match": f"{r['home_team']} vs {r['away_team']}",
                    "side": side, "label": label,
                    "p_model": pm, "p_market": pk,
                    "diff_pp": (pm - pk) * 100,
                    "odds": od,
                    "ev": pm * od - 1.0,
                })
        t = pd.DataFrame(rows)
        if t.empty:
            return t
        t["abs_diff"] = t["diff_pp"].abs()
        t["warn"] = t["abs_diff"] > VALUE_WARN_PP * 100
        # ordenar partidos por su discrepancia máxima, lados por |diff|
        order = t.groupby("match")["abs_diff"].transform("max")
        return (t.assign(_o=order)
                 .sort_values(["_o", "match", "abs_diff"],
                              ascending=[False, True, False])
                 .drop(columns="_o").reset_index(drop=True))

    def print_value(self) -> None:
        t = self.value_table()
        if t.empty:
            print("Sin partidos próximos con odds en el tracking. "
                  "Corre primero: python src/odds_fetcher.py --register")
            return
        w, _ = self.blend_weight()
        print(f"\n=== Detector de valor (modelo vs mercado) ===")
        cur = None
        for _, r in t.iterrows():
            if r["match"] != cur:
                cur = r["match"]
                print(f"\n  {r['date']}  {r['match']}")
            warn = "  <!> VERIFICAR ALINEACIONES" if r["warn"] else ""
            print(f"    {r['side']:<8} modelo {r['p_model']:5.1%} | "
                  f"mercado {r['p_market']:5.1%} | diff {r['diff_pp']:+5.1f}pp | "
                  f"odds {r['odds']:.2f} | EV {r['ev']:+.3f}{warn}")
        if t["warn"].any():
            print("\n  <!> Discrepancias >10pp: antes de confiar en el modelo, "
                  "verifica noticias de alineación/lesiones — el modelo no "
                  "las ve, el mercado sí.")

    # --------------------------------------------------------- export web ---

    def export_web(self, out_dir: Path | None = None) -> Path:
        """Exporta tracking + métricas + detector de valor a JSON y un
        index.html estático autocontenido (sin servidor)."""
        import json
        from datetime import datetime, timezone

        out_dir = out_dir or (TRACK_CSV.parent.parent.parent
                              / "reports" / "web")
        out_dir.mkdir(parents=True, exist_ok=True)
        t = self.value_table()
        matches = json.loads(
            self.df.where(pd.notna(self.df), None).to_json(orient="records"))
        for rec, (_, row) in zip(matches, self.df.iterrows()):
            g = grade_match(row["logloss_model"])
            rec["grade"] = g["label"]
            rec["grade_level"] = g["level"]
            h = hit_1x2(row)
            rec["hit_1x2"] = None if h is None else bool(h)
        data = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": self.metrics(),
            "matches": matches,
            "value": (json.loads(
                t.where(pd.notna(t), None).to_json(orient="records"))
                if not t.empty else []),
        }
        (out_dir / "data.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "index.html").write_text(
            _WEB_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)),
            encoding="utf-8")
        return out_dir

    def report(self) -> None:
        """Tabla del torneo + log-loss acumulado (modelo vs mercado vs azar)."""
        if self.df.empty:
            print("Sin partidos registrados todavía.")
            return
        print(f"\n=== Tracking Mundial 2026 — {len(self.df)} partidos "
              f"registrados, {self.df['outcome'].ne('').sum()} con resultado ===")
        for _, r in self.df.iterrows():
            res = (f"{int(r['home_score'])}-{int(r['away_score'])}"
                   if r["outcome"] else "pendiente")
            tag = ""
            if r["outcome"]:
                g = grade_match(r["logloss_model"])
                hit = "1X2 ok" if hit_1x2(r) else "1X2 x"
                tag = f"  [{g['label']:<7}] {hit:<7} (ll {r['logloss_model']:.2f})"
            print(f"  {r['date']}  {r['home_team']} vs {r['away_team']}: "
                  f"{r['p_home']:.0%}/{r['p_draw']:.0%}/{r['p_away']:.0%} "
                  f"-> {res}{tag}")

        m = self.metrics()
        if m["n_played"]:
            played = self.df[self.df["outcome"] != ""]
            grades = [grade_match(ll)["label"]
                      for ll in played["logloss_model"]]
            print(f"\n  grado por log-loss (ACIERTO <0.70 · FLOJO <1.0986 · "
                  f"FALLO >1.0986): "
                  f"{grades.count('ACIERTO')} ACIERTO · "
                  f"{grades.count('FLOJO')} FLOJO · {grades.count('FALLO')} FALLO")
            print(f"\n  log-loss modelo (todos):  {m['ll_model']:.4f} "
                  f"({m['n_played']} partidos)")
            print(f"  referencia azar uniforme: {log(3):.4f}")
            if m["n_market"]:
                print(f"\n  comparación en los {m['n_market']} partidos con odds:")
                print(f"    modelo:  {m['ll_model_subset']:.4f}")
                print(f"    mercado: {m['ll_market']:.4f}")
                print(f"    blend:   {m['ll_blend']:.4f} "
                      f"(w={m['blend_w']:.2f}, {m['blend_w_source']})")
            else:
                print(f"  blend: w={m['blend_w']:.2f} ({m['blend_w_source']})")


_WEB_TEMPLATE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>worldcup-predictor — Mundial 2026</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:960px;
      background:#0e1117;color:#e6e6e6;padding:0 1rem}
 h1,h2{color:#58a6ff} table{border-collapse:collapse;width:100%;margin:1rem 0}
 th,td{padding:.4rem .6rem;border-bottom:1px solid #30363d;text-align:left;
       font-size:.9rem} th{color:#8b949e}
 .ok{color:#3fb950}.bad{color:#f85149}.warn{background:#3d2c00}
 .pill{display:inline-block;padding:.1rem .5rem;border-radius:1rem;
       background:#21262d;margin-right:.5rem}
 small{color:#8b949e}
 .g-good{color:#3fb950;font-weight:600}.g-weak{color:#d29922;font-weight:600}
 .g-bad{color:#f85149;font-weight:600}.g-none{color:#8b949e}
</style></head><body>
<h1>worldcup-predictor — Mundial 2026</h1>
<p id="meta"></p><div id="metrics"></div>
<h2>Detector de valor</h2><div id="value"></div>
<h2>Partidos</h2><div id="matches"></div>
<script>
const D = __DATA__;
const pct = x => x==null ? "—" : (100*x).toFixed(1)+"%";
const f = (x,n=4) => x==null ? "—" : (+x).toFixed(n);
document.getElementById("meta").innerHTML =
  `<small>generado ${D.generated} (UTC)</small>`;
const M = D.metrics;
let mh = `<span class="pill">jugados: ${M.n_played}</span>
 <span class="pill">log-loss modelo: ${f(M.ll_model)}</span>
 <span class="pill">w blend: ${M.blend_w}</span>`;
if (M.n_market) mh += `<br><span class="pill">con odds: ${M.n_market}</span>
 <span class="pill">modelo: ${f(M.ll_model_subset)}</span>
 <span class="pill">mercado: ${f(M.ll_market)}</span>
 <span class="pill">blend: ${f(M.ll_blend)}</span>`;
document.getElementById("metrics").innerHTML = mh;
if (D.value.length) {
  let h = `<table><tr><th>partido</th><th>lado</th><th>modelo</th>
   <th>mercado</th><th>diff</th><th>odds</th><th>EV</th></tr>`;
  for (const v of D.value)
    h += `<tr class="${v.warn ? "warn" : ""}"><td>${v.match}</td>
     <td>${v.side}</td><td>${pct(v.p_model)}</td><td>${pct(v.p_market)}</td>
     <td>${v.diff_pp.toFixed(1)}pp</td><td>${f(v.odds,2)}</td>
     <td class="${v.ev>0?"ok":"bad"}">${f(v.ev,3)}</td></tr>`;
  h += "</table><small>filas resaltadas: discrepancia &gt;10pp — verificar " +
       "noticias de alineación antes de confiar en el modelo</small>";
  document.getElementById("value").innerHTML = h;
} else document.getElementById("value").innerHTML =
  "<small>sin partidos próximos con odds</small>";
let h = `<table><tr><th>fecha</th><th>partido</th><th>predicción 1X2</th>
 <th>mercado</th><th>resultado</th><th>grado</th><th>1X2</th>
 <th>log-loss</th></tr>`;
for (const r of D.matches) {
  const res = r.outcome ? `${r.home_score}-${r.away_score}` : "pendiente";
  const mkt = r.p_mkt_home!=null ?
    `${pct(r.p_mkt_home)}/${pct(r.p_mkt_draw)}/${pct(r.p_mkt_away)}` : "—";
  const grade = r.grade ?
    `<span class="g-${r.grade_level}">${r.grade}</span>` : "—";
  const x12 = r.hit_1x2==null ? "—" :
    (r.hit_1x2 ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>');
  h += `<tr><td>${r.date}</td><td>${r.home_team} vs ${r.away_team}</td>
   <td>${pct(r.p_home)}/${pct(r.p_draw)}/${pct(r.p_away)}</td><td>${mkt}</td>
   <td>${res}</td><td>${grade}</td><td>${x12}</td>
   <td>${f(r.logloss_model,3)}</td></tr>`;
}
h += `</table><small>grado por log-loss del partido: ` +
  `<span class="g-good">ACIERTO</span> &lt;0.70 · ` +
  `<span class="g-weak">FLOJO</span> 0.70–1.0986 · ` +
  `<span class="g-bad">FALLO</span> &gt;1.0986 (azar). ` +
  `La columna 1X2 es el acierto direccional puro.</small>`;
document.getElementById("matches").innerHTML = h;
</script></body></html>
"""


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in ("predict", "result", "report", "value",
                                   "export"):
        print(__doc__)
        return
    tracker = LiveTracker()
    if args[0] == "report":
        tracker.report()
        return
    if args[0] == "value":
        tracker.print_value()
        return
    if args[0] == "export":
        out = tracker.export_web()
        print(f"export web -> {out / 'index.html'}")
        return

    odds = None
    if "--odds" in args:
        i = args.index("--odds")
        odds = tuple(float(x) for x in args[i + 1:i + 4])
        args = args[:i]

    if args[0] == "predict":
        tracker.record_prediction(args[1], args[2], odds)
    elif args[0] == "result":
        tracker.add_result(args[1], args[2], int(args[3]), int(args[4]), odds)


if __name__ == "__main__":
    main()
