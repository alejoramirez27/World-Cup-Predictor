import { heatColor, pct, scoreLabel } from "@/lib/format";

// Heatmap 7x7 de la matriz de marcadores (índice 6 = "6+"). La diagonal
// (empates) se marca con un anillo sutil. Tooltip nativo por celda.
export function PoissonHeatmap({
  matrix,
  homeTeam,
  awayTeam,
}: {
  matrix: number[][];
  homeTeam: string;
  awayTeam: string;
}) {
  const n = matrix.length;
  const pmax = Math.max(...matrix.flat());

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between text-xs text-faint">
        <span>
          filas: <span className="text-muted">{homeTeam}</span> ↓
        </span>
        <span>
          columnas: <span className="text-muted">{awayTeam}</span> →
        </span>
      </div>

      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: `1.5rem repeat(${n}, minmax(0, 1fr))` }}
      >
        <div aria-hidden />
        {Array.from({ length: n }, (_, j) => (
          <div key={`h${j}`} className="tnum text-center text-[11px] text-faint">
            {j === n - 1 ? `${j}+` : j}
          </div>
        ))}

        {matrix.map((row, i) => (
          <Row key={i}>
            <div className="tnum flex items-center justify-center text-[11px] text-faint">
              {i === n - 1 ? `${i}+` : i}
            </div>
            {row.map((p, j) => (
              <div
                key={j}
                title={`${scoreLabel(i, j, n)} · ${pct(p, 1)}`}
                className={`tnum aspect-square rounded-[5px] flex items-center justify-center text-[10px] transition-transform hover:scale-[1.08] ${
                  i === j ? "ring-1 ring-inset ring-fg/15" : ""
                }`}
                style={{
                  background: heatColor(p, pmax),
                  color: p / pmax > 0.45 ? "#06121f" : "var(--color-muted)",
                }}
              >
                {p >= 0.05 ? Math.round(p * 100) : ""}
              </div>
            ))}
          </Row>
        ))}
      </div>
      <p className="mt-3 text-xs text-faint">
        Celdas en % de probabilidad. El anillo marca los empates (diagonal).
      </p>
    </div>
  );
}

// Subgrid por fila para mantener el grid del padre.
function Row({ children }: { children: React.ReactNode }) {
  return <div className="contents">{children}</div>;
}
