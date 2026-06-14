import { getSupabase } from "./supabase";
import type { MatchWithResult, MetricRow, PredictionRow, ResultRow } from "./types";

// Embebido predictions + results (relación 1-1 por la FK results.match_id).
const SELECT = "*, result:results(*)";

type RawJoin = PredictionRow & { result: ResultRow | ResultRow[] | null };

function normalize(row: RawJoin): MatchWithResult {
  const r = Array.isArray(row.result) ? (row.result[0] ?? null) : row.result;
  return { ...row, result: r };
}

export function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Próximos partidos (fecha >= hoy), ascendente. */
export async function getUpcoming(limit = 12): Promise<MatchWithResult[]> {
  const db = getSupabase();
  if (!db) return [];
  const { data, error } = await db
    .from("predictions")
    .select(SELECT)
    .gte("fecha", todayISO())
    .order("fecha", { ascending: true })
    .limit(limit);
  if (error || !data) return [];
  return (data as RawJoin[]).map(normalize);
}

/** Últimos partidos jugados (con resultado), descendente. */
export async function getRecentResults(limit = 12): Promise<MatchWithResult[]> {
  const db = getSupabase();
  if (!db) return [];
  const { data, error } = await db
    .from("results")
    .select(
      "match_id, goles_home, goles_away, log_loss_partido, acierto_1x2, registrado_at, prediction:predictions(*)"
    )
    .order("registrado_at", { ascending: false })
    .limit(limit);
  if (error || !data) return [];
  type Row = ResultRow & { prediction: PredictionRow | PredictionRow[] | null };
  const out: MatchWithResult[] = [];
  for (const row of data as unknown as Row[]) {
    const pred = Array.isArray(row.prediction) ? row.prediction[0] : row.prediction;
    if (!pred) continue;
    const result: ResultRow = {
      match_id: row.match_id,
      goles_home: row.goles_home,
      goles_away: row.goles_away,
      log_loss_partido: row.log_loss_partido,
      acierto_1x2: row.acierto_1x2,
      registrado_at: row.registrado_at,
    };
    out.push({ ...pred, result });
  }
  return out;
}

/** Un partido por match_id. */
export async function getMatch(id: string): Promise<MatchWithResult | null> {
  const db = getSupabase();
  if (!db) return null;
  const { data, error } = await db
    .from("predictions")
    .select(SELECT)
    .eq("match_id", id)
    .maybeSingle();
  if (error || !data) return null;
  return normalize(data as RawJoin);
}

/** Todos los partidos jugados, para la tabla de tracking (ascendente por fecha). */
export async function getTracking(): Promise<MatchWithResult[]> {
  const db = getSupabase();
  if (!db) return [];
  const { data, error } = await db
    .from("predictions")
    .select(SELECT)
    .order("fecha", { ascending: true });
  if (error || !data) return [];
  return (data as RawJoin[]).map(normalize).filter((m) => m.result !== null);
}

/** Serie temporal de métricas acumuladas. */
export async function getMetrics(): Promise<MetricRow[]> {
  const db = getSupabase();
  if (!db) return [];
  const { data, error } = await db
    .from("model_metrics")
    .select("*")
    .order("fecha", { ascending: true });
  if (error || !data) return [];
  return data as MetricRow[];
}

/** Última fila de métricas (para stat-tiles). */
export async function getLatestMetric(): Promise<MetricRow | null> {
  const rows = await getMetrics();
  return rows.length ? rows[rows.length - 1] : null;
}
