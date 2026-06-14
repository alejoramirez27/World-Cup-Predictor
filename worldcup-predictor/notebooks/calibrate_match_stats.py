"""Calibra la estimación de remates y corners con datos REALES de torneos
internacionales (StatsBomb Open Data, gratis en GitHub). Cuenta remates y
corners por equipo a partir de los eventos, y ajusta por mínimos cuadrados:
    remates_equipo ≈ a + b · goles_equipo
    corners_equipo ≈ a + b · goles_equipo
La relación se aplica luego a los goles esperados (λ) del modelo.
"""
import json
import sys

import numpy as np
import requests
import truststore

truststore.inject_into_ssl()

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
MAX_MATCHES = 150  # suficiente para ajustar 2 coeficientes lineales


def get(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    comps = get(f"{BASE}/competitions.json")
    # torneos internacionales de selecciones masculinas (Mundial, Euro)
    wanted = {}
    for c in comps:
        name = c["competition_name"]
        if c.get("competition_gender") == "male" and (
            "World Cup" in name or "Euro" in name
        ):
            wanted.setdefault((c["competition_id"], c["season_id"]),
                              f"{name} {c['season_name']}")
    print(f"competiciones internacionales encontradas: {len(wanted)}")
    for k, v in wanted.items():
        print(f"  {k} -> {v}")

    rows = []  # (goles, remates, corners) por equipo-partido
    n_matches = 0
    for (cid, sid), label in wanted.items():
        if n_matches >= MAX_MATCHES:
            break
        try:
            matches = get(f"{BASE}/matches/{cid}/{sid}.json")
        except Exception as exc:
            print(f"  (sin matches {label}: {exc})")
            continue
        print(f"\n{label}: {len(matches)} partidos")
        for m in matches:
            if n_matches >= MAX_MATCHES:
                break
            mid = m["match_id"]
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            gh, ga = m["home_score"], m["away_score"]
            try:
                events = get(f"{BASE}/events/{mid}.json")
            except Exception:
                continue
            shots = {home: 0, away: 0}
            corners = {home: 0, away: 0}
            for e in events:
                t = e.get("team", {}).get("name")
                if t not in shots:
                    continue
                tn = e.get("type", {}).get("name")
                if tn == "Shot":
                    shots[t] += 1
                elif tn == "Pass" and e.get("pass", {}).get("type", {}).get("name") == "Corner":
                    corners[t] += 1
            rows.append((gh, shots[home], corners[home]))
            rows.append((ga, shots[away], corners[away]))
            n_matches += 1
        print(f"  acumulado: {n_matches} partidos")

    arr = np.array(rows, dtype=float)
    goals, shots_v, corners_v = arr[:, 0], arr[:, 1], arr[:, 2]
    print(f"\n=== muestra: {len(arr)} filas equipo-partido ({n_matches} partidos) ===")
    print(f"promedios reales: goles={goals.mean():.2f}  remates={shots_v.mean():.2f}  "
          f"corners={corners_v.mean():.2f}")

    def fit(y, x):
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return a, b, 1 - ss_res / ss_tot

    a_s, b_s, r2_s = fit(shots_v, goals)
    a_c, b_c, r2_c = fit(corners_v, goals)
    print(f"\nremates  = {a_s:.3f} + {b_s:.3f}·goles   (R²={r2_s:.3f})")
    print(f"corners  = {a_c:.3f} + {b_c:.3f}·goles   (R²={r2_c:.3f})")

    out = {
        "n_matches": n_matches, "n_rows": len(arr),
        "shots": {"a": round(a_s, 4), "b": round(b_s, 4), "r2": round(r2_s, 4)},
        "corners": {"a": round(a_c, 4), "b": round(b_c, 4), "r2": round(r2_c, 4)},
        "avg": {"goals": round(float(goals.mean()), 3),
                "shots": round(float(shots_v.mean()), 3),
                "corners": round(float(corners_v.mean()), 3)},
    }
    from pathlib import Path
    p = Path(__file__).parent / "match_stats_calibration.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nguardado -> {p}")


if __name__ == "__main__":
    main()
