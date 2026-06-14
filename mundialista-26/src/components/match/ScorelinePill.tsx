import { pct } from "@/lib/format";
import type { Scoreline } from "@/lib/types";

export function ScorelinePill({ top }: { top: Scoreline | null }) {
  if (!top) return null;
  return (
    <span className="inline-flex items-center gap-1.5 text-xs">
      <span className="text-faint">marcador</span>
      <span className="tnum rounded-card bg-surface-2 px-2 py-1 text-fg">{top.score}</span>
      <span className="tnum text-muted">{pct(top.prob)}</span>
    </span>
  );
}
