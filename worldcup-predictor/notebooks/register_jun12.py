"""Registra Canadá 1-1 Bosnia (final) y CONGELA la predicción de USA-Paraguay
antes de que termine (para que sea genuinamente pre-partido). El resultado de
USA-Paraguay se agrega después con su marcador."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live_tracking import LiveTracker

t = LiveTracker()
print("--- Canadá 1-1 Bosnia ---")
t.add_result("Canada", "Bosnia and Herzegovina", 1, 1)

print("\n--- congelando USA vs Paraguay (pre-partido) ---")
if t._find("United States", "Paraguay") is None:
    t.record_prediction("United States", "Paraguay")
else:
    print("USA-Paraguay ya estaba registrado, no se regenera.")

print()
t.report()
