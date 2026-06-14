-- Esquema Supabase para worldcup-predictor (Mundial 2026).
-- Ejecutar una vez en el SQL Editor de Supabase.
--
-- Modelo de seguridad:
--   * RLS activado en las tres tablas.
--   * anon + authenticated: SOLO lectura (la web es pública read-only).
--   * Escritura: únicamente con la SERVICE key, que bypassa RLS por diseño
--     (no se define ninguna policy de insert/update/delete, así que ni anon
--     ni authenticated pueden escribir aunque tengan el grant).

-- ============================================================ predictions
create table if not exists public.predictions (
    match_id        text primary key,          -- "2026-06-11__mexico__vs__south-africa"
    fecha           date not null,
    fase_grupo      text,                       -- grupo A-L (o fase si knockout)
    equipo_home     text not null,
    equipo_away     text not null,
    prob_home       double precision,
    prob_draw       double precision,
    prob_away       double precision,
    top_scorelines  jsonb,                      -- [{"score":"1-1","prob":0.13}, ...] top-5
    poisson_matrix  jsonb,                      -- 7x7 (índice 6 = "6+ goles")
    over25_prob     double precision,
    prob_goleada    double precision,           -- favorito gana por 2+
    lambda_home     double precision,
    lambda_away     double precision,
    model_version   text,
    updated_at      timestamptz not null default now()
);

-- ================================================================ results
create table if not exists public.results (
    match_id          text primary key
                        references public.predictions(match_id) on delete cascade,
    goles_home        int,
    goles_away        int,
    log_loss_partido  double precision,
    acierto_1x2       boolean,
    registrado_at     timestamptz not null default now()
);

-- ========================================================== model_metrics
create table if not exists public.model_metrics (
    id                   bigint generated always as identity primary key,
    fecha                date not null unique,   -- un snapshot por día (upsert)
    log_loss_acumulado   double precision,
    brier_acumulado      double precision,
    accuracy_1x2         double precision,
    partidos_evaluados   int,
    log_loss_azar        double precision not null default 1.0986,
    updated_at           timestamptz not null default now()
);

-- ===================================================================== RLS
alter table public.predictions   enable row level security;
alter table public.results       enable row level security;
alter table public.model_metrics enable row level security;

-- Lectura pública (anon y authenticated). Sin policies de escritura:
-- los writes solo funcionan con la service key (bypassa RLS).
drop policy if exists "public read predictions"   on public.predictions;
drop policy if exists "public read results"        on public.results;
drop policy if exists "public read model_metrics"  on public.model_metrics;

create policy "public read predictions"  on public.predictions
    for select to anon, authenticated using (true);
create policy "public read results"       on public.results
    for select to anon, authenticated using (true);
create policy "public read model_metrics" on public.model_metrics
    for select to anon, authenticated using (true);

-- Privilegios: solo SELECT para los roles públicos (sin insert/update/delete).
grant select on public.predictions, public.results, public.model_metrics
    to anon, authenticated;

-- Índices útiles para la web.
create index if not exists predictions_fecha_idx on public.predictions (fecha);
create index if not exists model_metrics_fecha_idx on public.model_metrics (fecha);
