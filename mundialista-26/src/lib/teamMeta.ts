import meta from "./teamMeta.json";

export type TeamMeta = {
  squad: number | null;
  fifaRank: number | null;
  eloRank: number | null;
};

const TEAMS = (meta as { teams: Record<string, TeamMeta> }).teams;
export const FIFA_RANK_DATE = (meta as { fifaRankDate: string }).fifaRankDate;

const EMPTY: TeamMeta = { squad: null, fifaRank: null, eloRank: null };

export function teamMeta(name: string): TeamMeta {
  return TEAMS[name] ?? EMPTY;
}
