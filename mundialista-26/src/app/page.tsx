import { ArrowRight } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { getUpcoming, getRecentResults, getLatestMetric } from "@/lib/queries";
import { supabaseConfigured } from "@/lib/supabase";
import { MatchCard } from "@/components/match/MatchCard";
import { Section } from "@/components/ui/Section";
import { StatTile } from "@/components/ui/StatTile";
import { pct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [upcoming, recent, metric] = await Promise.all([
    getUpcoming(9),
    getRecentResults(6),
    getLatestMetric(),
  ]);

  return (
    <>
      {/* hero compacto: el dato es el protagonista */}
      <section className="mb-12">
        <p className="text-sm text-accent font-medium">Mundial 2026</p>
        <h1 className="mt-2 max-w-2xl text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
          Probabilidades honestas para cada partido, con el modelo a la vista.
        </h1>
        <p className="mt-3 max-w-xl text-muted">
          Cada predicción se congela antes del partido y se evalúa contra el
          resultado real. Sin retoques a posteriori.
        </p>

        {metric && (
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 gap-3 max-w-xl">
            <StatTile
              label="Log-loss acumulado"
              value={metric.log_loss_acumulado.toFixed(3)}
              sub={`azar: ${metric.log_loss_azar.toFixed(3)}`}
              accent={
                metric.log_loss_acumulado < metric.log_loss_azar
                  ? "var(--color-good)"
                  : "var(--color-bad)"
              }
            />
            <StatTile label="Acierto 1X2" value={pct(metric.accuracy_1x2)} />
            <StatTile label="Partidos" value={String(metric.partidos_evaluados)} />
          </div>
        )}
      </section>

      {!supabaseConfigured() && (
        <div className="mb-10 rounded-card border border-border bg-surface px-4 py-3 text-sm text-muted">
          Conecta Supabase (variables <span className="tnum">NEXT_PUBLIC_SUPABASE_*</span> en{" "}
          <span className="tnum">.env.local</span>) para cargar los datos.
        </div>
      )}

      <Section title="Hoy y próximos" hint={`${upcoming.length} partidos`}>
        {upcoming.length ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {upcoming.map((m) => (
              <MatchCard key={m.match_id} match={m} />
            ))}
          </div>
        ) : (
          <EmptyState text="No hay próximos partidos cargados." />
        )}
      </Section>

      {recent.length > 0 && (
        <Section title="Últimos resultados">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {recent.map((m) => (
              <MatchCard key={m.match_id} match={m} />
            ))}
          </div>
          <Link
            href="/tracking"
            className="mt-5 inline-flex items-center gap-1.5 text-sm text-accent hover:gap-2.5 transition-all"
          >
            Ver seguimiento completo <ArrowRight size={15} />
          </Link>
        </Section>
      )}
    </>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-card border border-border bg-surface px-4 py-10 text-center text-sm text-muted">
      {text}
    </div>
  );
}
