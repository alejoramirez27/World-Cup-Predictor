// Tipos espejo del esquema Supabase (sql/supabase_schema.sql del proyecto Python).

export type Scoreline = { score: string; prob: number };

export type PredictionRow = {
  match_id: string;
  fecha: string; // YYYY-MM-DD
  fase_grupo: string | null;
  equipo_home: string;
  equipo_away: string;
  prob_home: number;
  prob_draw: number;
  prob_away: number;
  top_scorelines: Scoreline[] | null;
  poisson_matrix: number[][] | null;
  over25_prob: number;
  prob_goleada: number;
  lambda_home: number;
  lambda_away: number;
  model_version: string;
};

export type ResultRow = {
  match_id: string;
  goles_home: number;
  goles_away: number;
  log_loss_partido: number;
  acierto_1x2: boolean;
  registrado_at: string;
};

export type MetricRow = {
  id: number;
  fecha: string;
  log_loss_acumulado: number;
  brier_acumulado: number;
  accuracy_1x2: number;
  partidos_evaluados: number;
  log_loss_azar: number;
};

export type MatchWithResult = PredictionRow & { result: ResultRow | null };
