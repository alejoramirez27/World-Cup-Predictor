import { pct } from "@/lib/format";
import { buildMarketGroups, totalGoalsDist } from "@/lib/markets";

// Todos los mercados de goles derivados de la matriz de marcadores.
export function MarketsPanel({
  matrix,
  home,
  away,
}: {
  matrix: number[][];
  home: string;
  away: string;
}) {
  const groups = buildMarketGroups(matrix, home, away);
  const totals = totalGoalsDist(matrix);
  const maxTotal = Math.max(...totals.map((t) => t.prob));

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {groups.map((g) => (
          <div key={g.title} className="rounded-card border border-border bg-surface p-4">
            <h3 className="text-sm font-semibold mb-3">{g.title}</h3>
            <ul className="space-y-2">
              {g.items.map((it) => (
                <li key={it.label} className="flex items-center gap-3">
                  <span className="text-sm text-muted w-40 shrink-0 truncate" title={it.label}>
                    {it.label}
                  </span>
                  <span className="flex-1 h-1.5 rounded-full bg-surface-2 overflow-hidden">
                    <span
                      className="anim-grow-x block h-full bg-accent"
                      style={{ width: `${Math.min(it.prob * 100, 100)}%` }}
                    />
                  </span>
                  <span className="tnum text-sm w-12 text-right">{pct(it.prob)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}

        {/* distribución de goles totales */}
        <div className="rounded-card border border-border bg-surface p-4 sm:col-span-2">
          <h3 className="text-sm font-semibold mb-3">Goles totales del partido</h3>
          <div className="flex items-end gap-2 h-28">
            {totals.map((t) => (
              <div key={t.label} className="flex-1 flex flex-col items-center justify-end gap-1">
                <span className="tnum text-[11px] text-muted">{pct(t.prob)}</span>
                <span
                  className="anim-grow-y w-full rounded-t bg-accent/70"
                  style={{ height: `${maxTotal > 0 ? (t.prob / maxTotal) * 100 : 0}%`, minHeight: "2px" }}
                />
                <span className="tnum text-[11px] text-faint">{t.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
