import { gradeOf } from "@/lib/format";

// Semáforo por log-loss del partido. El color lo manda el log-loss.
export function GradeBadge({ logloss }: { logloss: number | null | undefined }) {
  const g = gradeOf(logloss);
  if (g.level === "none") return null;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium"
      style={{
        color: g.color,
        background: `color-mix(in oklab, ${g.color} 14%, transparent)`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: g.color }} aria-hidden />
      {g.label}
    </span>
  );
}
