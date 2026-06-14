import type { MetricRow } from "@/lib/types";

// Chart SVG propio: log-loss acumulado del modelo vs la línea del azar.
// Estático + animación de trazado por CSS (respeta prefers-reduced-motion).
export function LogLossChart({ metrics }: { metrics: MetricRow[] }) {
  if (metrics.length === 0) {
    return (
      <div className="rounded-card border border-border bg-surface px-4 py-10 text-center text-sm text-muted">
        Sin snapshots de métricas todavía.
      </div>
    );
  }

  const W = 720;
  const H = 240;
  const pad = { top: 16, right: 16, bottom: 28, left: 40 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const azar = metrics[0].log_loss_azar ?? Math.log(3);
  const vals = metrics.map((m) => m.log_loss_acumulado);
  const yMax = Math.max(azar, ...vals) * 1.15;

  const n = metrics.length;
  const x = (i: number) => pad.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const y = (v: number) => pad.top + plotH - (v / yMax) * plotH;

  const pts = metrics.map((m, i) => `${x(i)},${y(m.log_loss_acumulado)}`);
  const line = pts.join(" ");
  const area = `M${x(0)},${y(0)} L${pts.join(" L")} L${x(n - 1)},${y(0)} Z`;
  const azarY = y(azar);

  const fmtDate = (s: string) => s.slice(5); // MM-DD

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="mb-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
        <Legend color="var(--color-accent)" label="log-loss acumulado del modelo" />
        <Legend color="var(--color-faint)" dashed label={`azar uniforme (${azar.toFixed(4)})`} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Log-loss acumulado del modelo a lo largo del torneo">
        {/* grid horizontal */}
        {[0, 0.5, 1].map((f) => {
          const v = yMax * f;
          return (
            <g key={f}>
              <line x1={pad.left} x2={W - pad.right} y1={y(v)} y2={y(v)} stroke="var(--color-border-soft)" strokeWidth={1} />
              <text x={pad.left - 6} y={y(v) + 3} textAnchor="end" className="tnum" fontSize={10} fill="var(--color-faint)">
                {v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* línea del azar */}
        <line x1={pad.left} x2={W - pad.right} y1={azarY} y2={azarY} stroke="var(--color-faint)" strokeWidth={1.5} strokeDasharray="5 4" />

        {/* área + línea del modelo */}
        <path d={area} fill="color-mix(in oklab, var(--color-accent) 12%, transparent)" />
        {n > 1 && (
          <polyline
            className="anim-draw"
            style={{ "--dash": "2200" } as React.CSSProperties}
            points={line}
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        {metrics.map((m, i) => (
          <circle key={i} cx={x(i)} cy={y(m.log_loss_acumulado)} r={3} fill="var(--color-accent)" />
        ))}

        {/* eje x: primera y última fecha */}
        <text x={pad.left} y={H - 8} textAnchor="start" className="tnum" fontSize={10} fill="var(--color-faint)">
          {fmtDate(metrics[0].fecha)}
        </text>
        {n > 1 && (
          <text x={W - pad.right} y={H - 8} textAnchor="end" className="tnum" fontSize={10} fill="var(--color-faint)">
            {fmtDate(metrics[n - 1].fecha)}
          </text>
        )}
      </svg>
      <p className="mt-1 text-xs text-faint">
        Por debajo de la línea de azar = el modelo aporta información.
      </p>
    </div>
  );
}

function Legend({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-muted">
      <svg width="18" height="8" aria-hidden>
        <line x1="0" y1="4" x2="18" y2="4" stroke={color} strokeWidth="2.5" strokeDasharray={dashed ? "4 3" : undefined} />
      </svg>
      {label}
    </span>
  );
}
