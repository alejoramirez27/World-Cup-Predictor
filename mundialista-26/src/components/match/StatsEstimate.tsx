import { pct } from "@/lib/format";
import { CALIBRATION, cornerShotMarkets, type Line } from "@/lib/stats";

// Probabilidades de líneas de corners y remates, modeladas como Poisson sobre
// las medias calibradas con datos reales. Orientativas (ver caveat).
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
  const m = cornerShotMarkets(lambdaHome, lambdaAway);
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <MarketCard title="Corners" data={m.corners} home={home} away={away} />
      <MarketCard title="Remates" data={m.shots} home={home} away={away} />
      <p className="sm:col-span-2 text-xs text-faint">
        Líneas modeladas como Poisson sobre medias calibradas con{" "}
        {CALIBRATION.nMatches} partidos reales ({CALIBRATION.source}). La media
        se predice débilmente desde los goles esperados (R² {CALIBRATION.shots.r2}{" "}
        remates / {CALIBRATION.corners.r2} corners), así que son probabilidades
        orientativas, cercanas al promedio del torneo.
      </p>
    </div>
  );
}

function MarketCard({
  title,
  data,
  home,
  away,
}: {
  title: string;
  data: { home: Line[]; away: Line[]; total: Line[]; expTotal: number };
  home: string;
  away: string;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="tnum text-xs text-faint">
          ≈ {data.expTotal.toFixed(1)} totales
        </span>
      </div>
      <LineGroup label="Total" lines={data.total} />
      <LineGroup label={home} lines={data.home} />
      <LineGroup label={away} lines={data.away} />
    </div>
  );
}

function LineGroup({ label, lines }: { label: string; lines: Line[] }) {
  return (
    <div className="mb-2 last:mb-0">
      <div className="text-xs text-faint mb-1 truncate">{label}</div>
      <div className="flex flex-wrap gap-2">
        {lines.map((l) => (
          <span
            key={l.line}
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2.5 py-1 text-xs"
          >
            <span className="text-muted">+{l.line}</span>
            <span className="tnum text-fg">{pct(l.over)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
