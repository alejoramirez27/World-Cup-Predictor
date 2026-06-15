import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "@phosphor-icons/react/dist/ssr";
import { getMatch } from "@/lib/queries";
import { supabaseConfigured } from "@/lib/supabase";
import { favorite, pct } from "@/lib/format";
import { ProbBar } from "@/components/match/ProbBar";
import { PoissonHeatmap } from "@/components/match/PoissonHeatmap";
import { PersonalityStat } from "@/components/match/PersonalityStat";
import { MarketsPanel } from "@/components/match/MarketsPanel";
import { StatsEstimate } from "@/components/match/StatsEstimate";
import { OverUnderBadge } from "@/components/match/OverUnderBadge";
import { GradeBadge } from "@/components/tracking/GradeBadge";
import { teamMeta, FIFA_RANK_DATE, type TeamMeta } from "@/lib/teamMeta";
import { matchContext } from "@/lib/context";

export const revalidate = 3600; // ISR: refresca datos cada hora sin redeploy

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const match = await getMatch(decodeURIComponent(id));
  if (!match) return { title: "Partido · mundialista·26" };
  return { title: `${match.equipo_home} vs ${match.equipo_away} · mundialista·26` };
}

export default async function MatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const match = await getMatch(decodeURIComponent(id));

  if (!match) {
    if (!supabaseConfigured()) {
      return (
        <p className="text-sm text-muted">
          Conecta Supabase en <span className="tnum">.env.local</span> para cargar este partido.
        </p>
      );
    }
    notFound();
  }

  const played = match.result !== null;
  const fav = favorite(match);
  const favLabel =
    fav === "home" ? match.equipo_home : fav === "away" ? match.equipo_away : "el favorito";

  const hm = teamMeta(match.equipo_home);
  const am = teamMeta(match.equipo_away);
  const ctx = matchContext(match, hm, am);

  return (
    <article className="anim-rise">
      <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-fg transition-colors">
        <ArrowLeft size={15} /> Volver
      </Link>

      <header className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-faint">
        <span className="tnum">{match.fecha}</span>
        {match.fase_grupo && <span>· Grupo {match.fase_grupo}</span>}
        <span>· {match.model_version}</span>
      </header>

      <div className="mt-2 flex items-end justify-between gap-4">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
          {match.equipo_home}
          {hm.fifaRank != null && (
            <span className="tnum text-sm text-faint font-normal" title="Ranking FIFA"> #{hm.fifaRank}</span>
          )}{" "}
          <span className="text-faint font-normal">vs</span> {match.equipo_away}
          {am.fifaRank != null && (
            <span className="tnum text-sm text-faint font-normal" title="Ranking FIFA"> #{am.fifaRank}</span>
          )}
        </h1>
        {played && (
          <div className="flex items-center gap-3">
            <span className="tnum text-3xl font-semibold">
              {match.result!.goles_home}-{match.result!.goles_away}
            </span>
            <GradeBadge logloss={match.result!.log_loss_partido} />
          </div>
        )}
      </div>

      <div className="mt-6 rounded-card border border-border bg-surface p-5">
        <div className="flex items-center justify-between text-sm text-muted mb-3">
          <span>Resultado esperado</span>
          <span className="tnum">
            λ {match.lambda_home.toFixed(2)} - {match.lambda_away.toFixed(2)}
          </span>
        </div>
        <ProbBar
          home={match.prob_home}
          draw={match.prob_draw}
          away={match.prob_away}
          homeLabel={match.equipo_home}
          awayLabel={match.equipo_away}
        />
      </div>

      <div className="mt-6 rounded-card border border-border bg-surface p-5">
        <h2 className="text-sm font-semibold mb-2">Contexto del partido</h2>
        <p className="text-muted leading-relaxed">{ctx}</p>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <TeamMetaBlock name={match.equipo_home} meta={hm} />
          <TeamMetaBlock name={match.equipo_away} meta={am} align="right" />
        </div>
        <p className="mt-3 text-xs text-faint">
          Plantilla = rating medio de los 23 mejores (EA FC 25). Ranking FIFA a{" "}
          {FIFA_RANK_DATE}; el Elo es actual. La descripción se genera desde las
          probabilidades del modelo, no de análisis táctico.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-6">
        {match.poisson_matrix && (
          <PoissonHeatmap
            matrix={match.poisson_matrix}
            homeTeam={match.equipo_home}
            awayTeam={match.equipo_away}
          />
        )}

        <div>
          <h2 className="text-sm font-semibold mb-3">Marcadores más probables</h2>
          <ul className="rounded-card border border-border divide-y divide-border-soft">
            {(match.top_scorelines ?? []).slice(0, 5).map((s, i) => (
              <li key={s.score} className="flex items-center gap-3 px-4 py-2.5">
                <span className="tnum w-4 text-xs text-faint">{i + 1}</span>
                <span className="tnum text-fg w-12">{s.score}</span>
                <span className="flex-1 h-1.5 rounded-full bg-surface-2 overflow-hidden">
                  <span
                    className="anim-grow-x block h-full bg-accent"
                    style={{ width: `${(s.prob / (match.top_scorelines?.[0]?.prob ?? 1)) * 100}%` }}
                  />
                </span>
                <span className="tnum text-sm text-muted w-12 text-right">{pct(s.prob)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mt-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Personalidad del partido</h2>
          <OverUnderBadge over={match.over25_prob} />
        </div>
        <PersonalityStat
          over25={match.over25_prob}
          goleada={match.prob_goleada}
          empate={match.prob_draw}
          favLabel={favLabel}
        />
      </div>

      <div className="mt-8">
        <h2 className="text-sm font-semibold mb-3">Remates y corners (estimado)</h2>
        <StatsEstimate
          lambdaHome={match.lambda_home}
          lambdaAway={match.lambda_away}
          home={match.equipo_home}
          away={match.equipo_away}
        />
      </div>

      {match.poisson_matrix && (
        <div className="mt-8">
          <h2 className="text-sm font-semibold mb-3">Todos los mercados</h2>
          <MarketsPanel
            matrix={match.poisson_matrix}
            home={match.equipo_home}
            away={match.equipo_away}
          />
          <p className="mt-3 text-xs text-faint">
            Todos los mercados se derivan de la distribución de goles del modelo.
            Corners, remates y goles por tiempo no están disponibles: el dataset
            solo contiene resultados finales.
          </p>
        </div>
      )}
    </article>
  );
}

function TeamMetaBlock({
  name,
  meta,
  align = "left",
}: {
  name: string;
  meta: TeamMeta;
  align?: "left" | "right";
}) {
  const cls = align === "right" ? "text-right" : "";
  return (
    <div className={cls}>
      <div className="font-medium">{name}</div>
      <div className="mt-1 text-sm text-muted tnum">
        Plantilla {meta.squad ?? "—"}
      </div>
      <div className="text-xs text-faint tnum">
        FIFA #{meta.fifaRank ?? "—"} · Elo #{meta.eloRank ?? "—"}
      </div>
    </div>
  );
}
