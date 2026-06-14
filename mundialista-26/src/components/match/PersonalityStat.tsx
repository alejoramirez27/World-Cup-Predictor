import { pct } from "@/lib/format";

// Indicadores de "personalidad" del partido (abierto vs cerrado).
export function PersonalityStat({
  over25,
  goleada,
  empate,
  favLabel,
}: {
  over25: number;
  goleada: number;
  empate: number;
  favLabel: string;
}) {
  const rows = [
    { k: "Over 2.5", v: over25, hint: "goles del partido" },
    { k: "Goleada", v: goleada, hint: `${favLabel} gana por 2+` },
    { k: "Empate", v: empate, hint: "reparto de puntos" },
  ];
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {rows.map((r) => (
        <div key={r.k} className="rounded-card border border-border bg-surface px-4 py-3">
          <div className="flex items-baseline justify-between">
            <span className="text-sm text-muted">{r.k}</span>
            <span className="tnum text-xl font-semibold">{pct(r.v)}</span>
          </div>
          <div className="text-xs text-faint mt-0.5">{r.hint}</div>
        </div>
      ))}
    </div>
  );
}
