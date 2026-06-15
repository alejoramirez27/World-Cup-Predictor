import type { TeamMeta } from "./teamMeta";

// Descripción honesta del partido esperado, derivada de los números del
// modelo (apertura, favoritismo, nivel de plantilla). NO describe tácticas
// concretas (laterales/extremos): no tenemos datos posicionales.

type P = {
  equipo_home: string;
  equipo_away: string;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  over25_prob: number;
  lambda_home: number;
  lambda_away: number;
};

export function matchContext(p: P, hm: TeamMeta, am: TeamMeta): string {
  const top = Math.max(p.prob_home, p.prob_draw, p.prob_away);
  const fav = p.prob_home >= p.prob_away ? p.equipo_home : p.equipo_away;
  const favProb = Math.max(p.prob_home, p.prob_away);
  const totalGoals = p.lambda_home + p.lambda_away;

  const shape =
    p.over25_prob >= 0.55
      ? "Se perfila un partido abierto y con goles"
      : p.over25_prob <= 0.45
        ? "Se perfila un partido trabado y de pocos goles"
        : "Se perfila un partido parejo en goles";

  const even = p.prob_draw >= top - 0.02 || Math.abs(p.prob_home - p.prob_away) < 0.08;
  const favPhrase = even
    ? "sin un favorito claro"
    : favProb >= 0.6
      ? `con ${fav} como favorito claro`
      : `con ${fav} como ligero favorito`;

  let squadPhrase = "";
  if (hm.squad != null && am.squad != null) {
    const gap = Math.abs(hm.squad - am.squad);
    const stronger = hm.squad >= am.squad ? p.equipo_home : p.equipo_away;
    squadPhrase =
      gap >= 5
        ? ` ${stronger} tiene bastante más nivel de plantilla.`
        : gap >= 2
          ? ` ${stronger} parte con algo más de nivel de plantilla.`
          : " Plantillas parejas en nivel.";
  }

  const extra =
    p.prob_draw >= 0.3
      ? " El empate asoma como escenario probable."
      : totalGoals >= 3.0
        ? " Pinta para varios goles."
        : "";

  return `${shape}, ${favPhrase}.${squadPhrase}${extra}`;
}
