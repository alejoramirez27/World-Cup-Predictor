"""Probe rápido: credenciales + existencia de tablas (sin cargar el modelo)."""
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sync_supabase import _env  # noqa: E402

from supabase import create_client  # noqa: E402

url, key = _env("SUPABASE_URL"), _env("SUPABASE_SERVICE_KEY")
print(f"url: {url}")
client = create_client(url, key)
for table in ("predictions", "results", "model_metrics"):
    try:
        res = client.table(table).select("*", count="exact").limit(1).execute()
        print(f"  OK  {table}: {res.count} filas")
    except Exception as exc:
        print(f"  FALTA  {table}: {type(exc).__name__}: {str(exc)[:120]}")
