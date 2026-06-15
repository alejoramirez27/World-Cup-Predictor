"""Genera metadata por selección (rating de plantilla + ranking FIFA oficial)
para las 48 del Mundial 2026, y la escribe como JSON estático en la web.
No toca Supabase: la web lee este JSON directo."""
import json
import sys
import unicodedata
from pathlib import Path

import re

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from squad_strength import squad_for  # noqa: E402

RAW = ROOT / "data" / "raw" / "results.csv"
OUT = ROOT.parent / "mundialista-26" / "src" / "lib" / "teamMeta.json"


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


RANK_ALIAS = {
    "usa": "united states", "korea republic": "south korea",
    "korea dpr": "north korea", "ir iran": "iran", "turkiye": "turkey",
    "cote d'ivoire": "ivory coast", "czechia": "czech republic",
    "cabo verde": "cape verde", "congo dr": "dr congo", "china pr": "china pr",
}

# equipos del Mundial 2026
df = pd.read_csv(RAW, parse_dates=["date"])
wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))

# ranking FIFA oficial y ACTUAL desde la API interna de inside.fifa.com.
# El dateId más reciente se descubre en la propia página.
def fetch_fifa_ranking() -> tuple[dict, str]:
    H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    page = requests.get("https://inside.fifa.com/es/fifa-world-ranking/men",
                        headers=H, timeout=30).text
    ids = sorted(set(re.findall(r"id\d{4,7}", page)), key=lambda s: int(s[2:]), reverse=True)
    for did in ids[:5]:
        d = requests.get(f"https://inside.fifa.com/api/ranking-overview?locale=en&dateId={did}",
                         headers=H, timeout=30).json()
        rows = d.get("rankings", [])
        if rows:
            rb = {}
            for it in rows:
                ri = it["rankingItem"]
                if ri.get("rank") is None or not ri.get("name"):
                    continue
                rb[RANK_ALIAS.get(norm(ri["name"]), norm(ri["name"]))] = int(ri["rank"])
            date = str(rows[0].get("lastUpdateDate", ""))[:10]
            return rb, date
    return {}, ""


rank_by, fifa_date = fetch_fifa_ranking()
print(f"ranking FIFA a la fecha: {fifa_date} ({len(rank_by)} selecciones)")

# ranking por Elo actual (nuestro modelo, junio 2026) sobre todas las selecciones
from model import prepare_data  # noqa: E402

_, builder = prepare_data()
elo_sorted = sorted(builder.states.items(), key=lambda kv: kv[1].elo, reverse=True)
elo_rank = {norm(t): i + 1 for i, (t, _) in enumerate(elo_sorted)}

meta = {}
miss_sq, miss_rk = [], []
for t in teams:
    sq = squad_for(t, 2025)
    rk_ = rank_by.get(norm(t))
    meta[t] = {
        "squad": None if pd.isna(sq) else round(float(sq), 1),
        "fifaRank": rk_,
        "eloRank": elo_rank.get(norm(t)),
    }
    if pd.isna(sq):
        miss_sq.append(t)
    if rk_ is None:
        miss_rk.append(t)

out = {"fifaRankDate": fifa_date, "teams": meta}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"escrito {OUT} ({len(meta)} equipos)")
print(f"sin plantilla: {miss_sq}")
print(f"sin ranking FIFA: {miss_rk}")
print("ejemplos:", {k: meta[k] for k in list(meta)[:4]})
