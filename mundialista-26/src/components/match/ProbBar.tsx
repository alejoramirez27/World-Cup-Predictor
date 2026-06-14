import { pct } from "@/lib/format";

// Barra apilada 1X2. Colores = serie de datos (home/draw/away), no decoración.
export function ProbBar({
  home,
  draw,
  away,
  homeLabel,
  awayLabel,
  compact = false,
}: {
  home: number;
  draw: number;
  away: number;
  homeLabel: string;
  awayLabel: string;
  compact?: boolean;
}) {
  const segs = [
    { v: home, color: "var(--color-home)" },
    { v: draw, color: "var(--color-draw)" },
    { v: away, color: "var(--color-away)" },
  ];
  return (
    <div>
      <div
        className="flex h-2.5 w-full overflow-hidden rounded-full bg-surface-2"
        role="img"
        aria-label={`Probabilidad: ${homeLabel} ${pct(home)}, empate ${pct(draw)}, ${awayLabel} ${pct(away)}`}
      >
        {segs.map((s, i) => (
          <span
            key={i}
            className="anim-grow-x h-full"
            style={{
              width: `${s.v * 100}%`,
              background: s.color,
              animationDelay: `${i * 90}ms`,
            }}
          />
        ))}
      </div>
      {!compact && (
        <div className="mt-2 flex justify-between text-xs">
          <span className="flex items-center gap-1.5">
            <Dot color="var(--color-home)" />
            <span className="text-muted">{homeLabel}</span>
            <span className="tnum text-fg">{pct(home)}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <Dot color="var(--color-draw)" />
            <span className="text-muted">X</span>
            <span className="tnum text-fg">{pct(draw)}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <Dot color="var(--color-away)" />
            <span className="text-muted">{awayLabel}</span>
            <span className="tnum text-fg">{pct(away)}</span>
          </span>
        </div>
      )}
    </div>
  );
}

function Dot({ color }: { color: string }) {
  return (
    <span
      className="inline-block h-2 w-2 rounded-full"
      style={{ background: color }}
      aria-hidden
    />
  );
}
