import Link from "next/link";
import { Check, X } from "@phosphor-icons/react/dist/ssr";
import type { MatchWithResult } from "@/lib/types";
import { gradeOf } from "@/lib/format";
import { GradeBadge } from "./GradeBadge";

export function TrackingTable({ matches }: { matches: MatchWithResult[] }) {
  if (matches.length === 0) {
    return (
      <div className="rounded-card border border-border bg-surface px-4 py-10 text-center text-sm text-muted">
        Aún no hay partidos con resultado registrado.
      </div>
    );
  }
  return (
    <div className="overflow-x-auto rounded-card border border-border">
      <table className="w-full min-w-[640px] text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-faint">
            <Th>Fecha</Th>
            <Th>Partido</Th>
            <Th>Pred 1X2</Th>
            <Th>Resultado</Th>
            <Th>Grado</Th>
            <Th center>1X2</Th>
            <Th right>Log-loss</Th>
          </tr>
        </thead>
        <tbody>
          {matches.map((m) => {
            const r = m.result!;
            const g = gradeOf(r.log_loss_partido);
            return (
              <tr key={m.match_id} className="border-b border-border-soft last:border-0 hover:bg-surface">
                <Td className="tnum text-faint whitespace-nowrap">{m.fecha}</Td>
                <Td>
                  <Link
                    href={`/partido/${encodeURIComponent(m.match_id)}`}
                    className="hover:text-accent transition-colors"
                  >
                    {m.equipo_home} vs {m.equipo_away}
                  </Link>
                </Td>
                <Td className="tnum text-muted whitespace-nowrap">
                  {Math.round(m.prob_home * 100)}/{Math.round(m.prob_draw * 100)}/
                  {Math.round(m.prob_away * 100)}
                </Td>
                <Td className="tnum whitespace-nowrap">
                  {r.goles_home}-{r.goles_away}
                </Td>
                <Td>
                  <GradeBadge logloss={r.log_loss_partido} />
                </Td>
                <Td center>
                  {r.acierto_1x2 ? (
                    <Check size={16} weight="bold" className="inline text-good" />
                  ) : (
                    <X size={16} weight="bold" className="inline text-bad" />
                  )}
                </Td>
                <Td right>
                  <span className="tnum" style={{ color: g.color }}>
                    {r.log_loss_partido.toFixed(3)}
                  </span>
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, center, right }: { children: React.ReactNode; center?: boolean; right?: boolean }) {
  return (
    <th className={`px-3 py-2.5 font-medium ${center ? "text-center" : right ? "text-right" : ""}`}>
      {children}
    </th>
  );
}
function Td({
  children,
  className = "",
  center,
  right,
}: {
  children: React.ReactNode;
  className?: string;
  center?: boolean;
  right?: boolean;
}) {
  return (
    <td className={`px-3 py-2.5 ${center ? "text-center" : right ? "text-right" : ""} ${className}`}>
      {children}
    </td>
  );
}
