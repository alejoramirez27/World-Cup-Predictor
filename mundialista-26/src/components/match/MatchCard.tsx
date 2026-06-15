import Link from "next/link";
import { CaretRight } from "@phosphor-icons/react/dist/ssr";
import type { MatchWithResult } from "@/lib/types";
import { todayISO } from "@/lib/queries";
import { ProbBar } from "./ProbBar";
import { ScorelinePill } from "./ScorelinePill";
import { OverUnderBadge } from "./OverUnderBadge";
import { GradeBadge } from "@/components/tracking/GradeBadge";
import { teamMeta } from "@/lib/teamMeta";

function Rank({ team }: { team: string }) {
  const r = teamMeta(team).eloRank;
  if (r == null) return null;
  return (
    <span className="tnum text-[11px] text-faint" title="Ranking Elo (actual)">
      #{r}
    </span>
  );
}

export function MatchCard({ match }: { match: MatchWithResult }) {
  const played = match.result !== null;
  const isToday = match.fecha === todayISO();
  const top = match.top_scorelines?.[0] ?? null;

  return (
    <Link
      href={`/partido/${encodeURIComponent(match.match_id)}`}
      className="anim-rise group block rounded-card border border-border bg-surface p-4 transition-colors hover:border-accent/60"
    >
      <div className="flex items-center justify-between text-xs text-faint">
        <span className="flex items-center gap-2">
          {isToday && !played && (
            <span className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent">
              HOY
            </span>
          )}
          <span className="tnum">{match.fecha}</span>
          {match.fase_grupo && <span>· Grupo {match.fase_grupo}</span>}
        </span>
        <CaretRight size={14} className="text-faint transition-transform group-hover:translate-x-0.5" />
      </div>

      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="font-medium flex items-center gap-1.5">
          {match.equipo_home} <Rank team={match.equipo_home} />
        </span>
        {played ? (
          <span className="tnum text-lg font-semibold">
            {match.result!.goles_home}-{match.result!.goles_away}
          </span>
        ) : (
          <span className="text-xs text-faint">vs</span>
        )}
        <span className="font-medium text-right flex items-center gap-1.5 justify-end">
          <Rank team={match.equipo_away} /> {match.equipo_away}
        </span>
      </div>

      <div className="mt-3">
        <ProbBar
          home={match.prob_home}
          draw={match.prob_draw}
          away={match.prob_away}
          homeLabel={match.equipo_home}
          awayLabel={match.equipo_away}
        />
      </div>

      <div className="mt-3 flex items-center justify-between gap-2">
        {played ? (
          <GradeBadge logloss={match.result!.log_loss_partido} />
        ) : (
          <ScorelinePill top={top} />
        )}
        <OverUnderBadge over={match.over25_prob} />
      </div>
    </Link>
  );
}
