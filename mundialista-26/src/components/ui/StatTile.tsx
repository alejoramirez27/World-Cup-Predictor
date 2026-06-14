export function StatTile({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="rounded-card border border-border bg-surface px-4 py-4">
      <div className="text-xs uppercase tracking-wide text-faint">{label}</div>
      <div
        className="tnum text-2xl sm:text-3xl mt-1 font-semibold"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      {sub ? <div className="text-xs text-muted mt-1">{sub}</div> : null}
    </div>
  );
}
