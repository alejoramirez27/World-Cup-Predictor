"""Replica las queries de la web con la key PÚBLICA (anon) para ver cuáles
fallan bajo RLS y por qué."""
import datetime
import truststore

truststore.inject_into_ssl()
from supabase import create_client

URL = "https://xtkgvhgavwsarbsiebjn.supabase.co"
ANON = "sb_publishable_9accttJSYa5LroVsutkyPg_ftYR1Lzw"
db = create_client(URL, ANON)

print("hoy (máquina):", datetime.date.today().isoformat())


def run(name, fn):
    try:
        res = fn()
        n = len(res.data) if res.data is not None else 0
        print(f"  {name}: {n} filas")
        if n:
            print("     ej:", str(res.data[0])[:90])
    except Exception as exc:
        print(f"  {name}: ERROR {type(exc).__name__}: {str(exc)[:140]}")


run("predictions (getTracking)", lambda: db.table("predictions").select("*, result:results(*)").limit(2).execute())
run("predictions gte hoy (getUpcoming)", lambda: db.table("predictions").select("*").gte("fecha", datetime.date.today().isoformat()).limit(2).execute())
run("results reverse-embed (getRecentResults)", lambda: db.table("results").select("match_id, goles_home, prediction:predictions(*)").limit(2).execute())
run("model_metrics (getMetrics)", lambda: db.table("model_metrics").select("*").execute())
