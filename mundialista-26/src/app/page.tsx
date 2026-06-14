import Link from "next/link";
import { getAllMatches, getLatestMetric, todayISO } from "@/lib/queries";
import { supabaseConfigured } from "@/lib/supabase";
import { MatchCard } from "@/components/match/MatchCard";
import { StatTile } from "@/components/ui/StatTile";
import { chipDate, cn, pct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{ fecha?: string }>;
}) {
  const { fecha } = await searchParams;
  const [all, metric] = await Promise.all([getAllMatches(), getLatestMetric()]);

  const dates = [...new Set(all.map((m) => m.fecha))].sort();
  const today = todayISO();
  const selected =
    fecha && dates.includes(fecha)
      ? fecha
      : dates.includes(today)
        ? today
        : (dates.find((d) => d >= today) ?? dates[dates.length - 1] ?? today);

  const dayMatches = all.filter((m) => m.fecha === selected);

  return (
    <>
      <section className="mb-10">
        <p className="text-sm text-accent font-medium">Mundial 2026</p>
        <h1 className="mt-2 max-w-2xl text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
          Probabilidades honestas para cada partido, con el modelo a la vista.
        </h1>

        {metric && (
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 gap-3 max-w-xl">
            <StatTile
              label="Log-loss acumulado"
              value={metric.log_loss_acumulado.toFixed(3)}
              sub={`azar: ${metric.log_loss_azar.toFixed(3)}`}
              accent={metric.log_loss_acumulado < metric.log_loss_azar ? "var(--color-good)" : "var(--color-bad)"}
            />
            <StatTile label="Acierto 1X2" value={pct(metric.accuracy_1x2)} />
            <StatTile label="Partidos" value={String(metric.partidos_evaluados)} />
          </div>
        )}
      </section>

      {!supabaseConfigured() && (
        <div className="mb-8 rounded-card border border-border bg-surface px-4 py-3 text-sm text-muted">
          Conecta Supabase (variables <span className="tnum">NEXT_PUBLIC_SUPABASE_*</span>) para cargar los datos.
        </div>
      )}

      {dates.length > 0 ? (
        <>
          {/* filtro por fecha */}
          <div className="mb-5 -mx-4 px-4 overflow-x-auto">
            <div className="flex gap-2 w-max pb-1">
              {dates.map((d) => {
                const c = chipDate(d);
                const isSel = d === selected;
                const isToday = d === today;
                const count = all.filter((m) => m.fecha === d).length;
                return (
                  <Link
                    key={d}
                    href={`/?fecha=${d}`}
                    scroll={false}
                    className={cn(
                      "shrink-0 rounded-card border px-3 py-2 text-center transition-colors",
                      isSel
                        ? "border-accent bg-accent-soft text-accent"
                        : "border-border bg-surface text-muted hover:text-fg"
                    )}
                  >
                    <div className="text-[11px] uppercase">{c.dow}</div>
                    <div className="tnum text-lg font-semibold leading-tight">{c.day}</div>
                    <div className="text-[11px]">
                      {isToday ? "hoy" : c.mon} · {count}
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* partidos de la fecha seleccionada */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {dayMatches.map((m) => (
              <MatchCard key={m.match_id} match={m} />
            ))}
          </div>
        </>
      ) : (
        <div className="rounded-card border border-border bg-surface px-4 py-10 text-center text-sm text-muted">
          No hay partidos cargados todavía.
        </div>
      )}

      <Link href="/tracking" className="mt-8 inline-block text-sm text-accent hover:underline">
        Ver seguimiento completo del modelo
      </Link>
    </>
  );
}
