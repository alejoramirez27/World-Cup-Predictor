"use client";

import { useMemo, useState } from "react";
import { chipDate, cn } from "@/lib/format";

export type ExplorerItem = {
  id: string;
  date: string;
  group: string | null;
  phase: string;
  node: React.ReactNode;
};

// Filtros de la home (fecha + grupo + fase) en cliente: instantáneo y permite
// que la página sea ISR (sin searchParams que la fuercen a dinámica).
export function MatchExplorer({ items, today }: { items: ExplorerItem[]; today: string }) {
  const dates = useMemo(() => Array.from(new Set(items.map((i) => i.date))).sort(), [items]);
  const groups = useMemo(
    () => Array.from(new Set(items.map((i) => i.group).filter((g): g is string => !!g))).sort(),
    [items]
  );
  const phases = useMemo(() => Array.from(new Set(items.map((i) => i.phase))), [items]);

  const defaultDate =
    dates.includes(today) ? today : (dates.find((d) => d >= today) ?? dates[dates.length - 1] ?? "all");
  const [date, setDate] = useState<string>(defaultDate);
  const [group, setGroup] = useState<string>("all");
  const [phase, setPhase] = useState<string>("all");

  const shown = items.filter(
    (i) =>
      (date === "all" || i.date === date) &&
      (group === "all" || i.group === group) &&
      (phase === "all" || i.phase === phase)
  );

  if (items.length === 0) {
    return (
      <div className="rounded-card border border-border bg-surface px-4 py-10 text-center text-sm text-muted">
        No hay partidos cargados todavía.
      </div>
    );
  }

  return (
    <div>
      {/* fecha */}
      <div className="-mx-4 px-4 overflow-x-auto">
        <div className="flex gap-2 w-max pb-1">
          <DayChip active={date === "all"} onClick={() => setDate("all")}>
            <div className="text-[11px] uppercase">todas</div>
            <div className="tnum text-base font-semibold leading-tight">·</div>
            <div className="text-[11px]">fechas</div>
          </DayChip>
          {dates.map((d) => {
            const c = chipDate(d);
            return (
              <DayChip key={d} active={date === d} onClick={() => setDate(d)}>
                <div className="text-[11px] uppercase">{c.dow}</div>
                <div className="tnum text-base font-semibold leading-tight">{c.day}</div>
                <div className="text-[11px]">{d === today ? "hoy" : c.mon}</div>
              </DayChip>
            );
          })}
        </div>
      </div>

      {/* grupo */}
      <div className="mt-3 flex flex-wrap gap-2">
        <Pill active={group === "all"} onClick={() => setGroup("all")}>Todos los grupos</Pill>
        {groups.map((g) => (
          <Pill key={g} active={group === g} onClick={() => setGroup(g)}>Grupo {g}</Pill>
        ))}
      </div>

      {/* fase (aparece sola cuando haya eliminatorias) */}
      {phases.length > 1 && (
        <div className="mt-2 flex flex-wrap gap-2">
          <Pill active={phase === "all"} onClick={() => setPhase("all")}>Todas las fases</Pill>
          {phases.map((p) => (
            <Pill key={p} active={phase === p} onClick={() => setPhase(p)}>{p}</Pill>
          ))}
        </div>
      )}

      <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {shown.length ? (
          shown.map((i) => <div key={i.id}>{i.node}</div>)
        ) : (
          <div className="sm:col-span-2 lg:col-span-3 rounded-card border border-border bg-surface px-4 py-8 text-center text-sm text-muted">
            No hay partidos con esos filtros.
          </div>
        )}
      </div>
    </div>
  );
}

function DayChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "shrink-0 rounded-card border px-3 py-2 text-center transition-colors",
        active ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface text-muted hover:text-fg"
      )}
    >
      {children}
    </button>
  );
}

function Pill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-full border px-3 py-1 text-xs transition-colors",
        active ? "border-accent bg-accent-soft text-accent" : "border-border bg-surface text-muted hover:text-fg"
      )}
    >
      {children}
    </button>
  );
}
