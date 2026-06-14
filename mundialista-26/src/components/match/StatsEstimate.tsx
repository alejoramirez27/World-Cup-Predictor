import { CALIBRATION, matchStatsEstimate } from "@/lib/stats";

// Remates y corners ESTIMADOS desde los goles esperados, calibrados con
// datos reales. Se muestran con su caveat de baja precisión.
export function StatsEstimate({
  lambdaHome,
  lambdaAway,
  home,
  away,
}: {
  lambdaHome: number;
  lambdaAway: number;
  home: string;
  away: string;
}) {
  const s = matchStatsEstimate(lambdaHome, lambdaAway);
  const rows = [
    { k: "Remates", h: s.shots.home, a: s.shots.away },
    { k: "Corners", h: s.corners.home, a: s.corners.away },
  ];

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="grid grid-cols-[1fr_auto_1fr] items-center text-xs text-faint mb-2">
        <span className="truncate">{home}</span>
        <span />
        <span className="text-right truncate">{away}</span>
      </div>
      <ul className="space-y-3">
        {rows.map((r) => {
          const total = r.h + r.a;
          const hPct = total > 0 ? (r.h / total) * 100 : 50;
          return (
            <li key={r.k}>
              <div className="flex items-center justify-between text-sm">
                <span className="tnum font-medium">{r.h.toFixed(1)}</span>
                <span className="text-muted">{r.k}</span>
                <span className="tnum font-medium">{r.a.toFixed(1)}</span>
              </div>
              <div className="mt-1 flex h-1.5 rounded-full overflow-hidden bg-surface-2">
                <span className="anim-grow-x h-full bg-home" style={{ width: `${hPct}%` }} />
                <span className="h-full bg-away" style={{ width: `${100 - hPct}%` }} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs text-faint">
        Estimación derivada de los goles esperados, calibrada con {CALIBRATION.nMatches}{" "}
        partidos reales ({CALIBRATION.source}). Relación débil (R²{" "}
        {CALIBRATION.shots.r2} remates / {CALIBRATION.corners.r2} corners): es
        aproximada, dominada por el promedio del torneo.
      </p>
    </div>
  );
}
