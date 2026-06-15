import Link from "next/link";
import { getAllMatches, getLatestMetric, todayISO } from "@/lib/queries";
import { supabaseConfigured } from "@/lib/supabase";
import { MatchCard } from "@/components/match/MatchCard";
import { MatchExplorer, type ExplorerItem } from "@/components/home/MatchExplorer";
import { StatTile } from "@/components/ui/StatTile";
import { pct } from "@/lib/format";

// ISR: se regenera cada hora (refresca datos de Supabase sin redeploy).
export const revalidate = 3600;

function phaseOf(group: string | null): string {
  return group && /^[A-L]$/.test(group) ? "Fase de grupos" : "Eliminatorias";
}

export default async function HomePage() {
  const [all, metric] = await Promise.all([getAllMatches(), getLatestMetric()]);

  const items: ExplorerItem[] = all.map((m) => ({
    id: m.match_id,
    date: m.fecha,
    group: m.fase_grupo,
    phase: phaseOf(m.fase_grupo),
    node: <MatchCard match={m} />,
  }));

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

      <MatchExplorer items={items} today={todayISO()} />

      <Link href="/tracking" className="mt-8 inline-block text-sm text-accent hover:underline">
        Ver seguimiento completo del modelo
      </Link>
    </>
  );
}
