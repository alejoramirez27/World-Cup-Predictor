import { pct } from "@/lib/format";

export function OverUnderBadge({ over }: { over: number }) {
  const open = over >= 0.5;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2.5 py-1 text-xs">
      <span className="text-faint">{open ? "Over" : "Under"} 2.5</span>
      <span className="tnum text-fg">{pct(open ? over : 1 - over)}</span>
    </span>
  );
}
