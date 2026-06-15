import { getTracking, getMetrics, getLatestMetric } from "@/lib/queries";
import { supabaseConfigured } from "@/lib/supabase";
import { LogLossChart } from "@/components/tracking/LogLossChart";
import { TrackingTable } from "@/components/tracking/TrackingTable";
import { StatTile } from "@/components/ui/StatTile";
import { Section } from "@/components/ui/Section";
import { pct } from "@/lib/format";

export const revalidate = 3600; // ISR: refresca datos cada hora sin redeploy
export const metadata = { title: "Tracking · mundialista·26" };

export default async function TrackingPage() {
  const [matches, metrics, latest] = await Promise.all([
    getTracking(),
    getMetrics(),
    getLatestMetric(),
  ]);

  return (
    <>
      <header className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">Seguimiento del modelo</h1>
        <p className="mt-2 max-w-xl text-muted">
          Cada predicción congelada contra su resultado real. El color de cada
          fila lo manda el log-loss del partido, la métrica honesta.
        </p>
      </header>

      {!supabaseConfigured() && (
        <div className="mb-8 rounded-card border border-border bg-surface px-4 py-3 text-sm text-muted">
          Conecta Supabase en <span className="tnum">.env.local</span> para cargar el seguimiento.
        </div>
      )}

      {latest && (
        <div className="mb-8 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatTile
            label="Log-loss acumulado"
            value={latest.log_loss_acumulado.toFixed(3)}
            sub={`azar: ${latest.log_loss_azar.toFixed(3)}`}
            accent={
              latest.log_loss_acumulado < latest.log_loss_azar
                ? "var(--color-good)"
                : "var(--color-bad)"
            }
          />
          <StatTile label="Brier" value={latest.brier_acumulado.toFixed(3)} />
          <StatTile label="Acierto 1X2" value={pct(latest.accuracy_1x2)} />
          <StatTile label="Partidos" value={String(latest.partidos_evaluados)} />
        </div>
      )}

      <Section title="Log-loss acumulado">
        <LogLossChart metrics={metrics} />
      </Section>

      <Section title="Predicción vs resultado" hint={`${matches.length} jugados`}>
        <TrackingTable matches={matches} />
        <p className="mt-3 text-xs text-faint">
          Semáforo por log-loss: <span className="text-good">ACIERTO</span> &lt;0.70 ·{" "}
          <span className="text-weak">FLOJO</span> 0.70-1.0986 ·{" "}
          <span className="text-bad">FALLO</span> &gt;1.0986 (azar). La columna 1X2 es el
          acierto direccional puro.
        </p>
      </Section>
    </>
  );
}
