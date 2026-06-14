# World Cup Predictor

Modelo de predicción del Mundial 2026 y una web pública que lo muestra y lo evalúa en vivo. Las predicciones se **congelan antes de cada partido** y se comparan contra el resultado real con log-loss, la métrica que premia la confianza calibrada y castiga la sobreconfianza.

🌐 **Web en vivo:** (pendiente de deploy en Vercel)

## Estructura

```
World-Cup-Predictor/
├── worldcup-predictor/   # modelo (Python): datos, features, entrenamiento, sync
└── mundialista-26/       # web (Next.js 16 + Tailwind, lee de Supabase, read-only)
```

## El modelo (`worldcup-predictor/`)

No es un clasificador de ganador: estima los goles esperados de cada selección, construye la distribución completa de marcadores y de ahí derivan el 1X2, el over/under y cada resultado exacto.

- **Elo ponderado por torneo** (K: Mundial 60, eliminatorias/copas 50, Nations League 40, amistosos 20), con warm-up desde 1990 para que los ratings estén convergidos.
- **Features ajustadas por rival:** goles relativos a la expectativa según el Elo del oponente, y forma reciente ponderada por dificultad.
- **Dos regresores XGBoost** con objetivo `count:poisson` (lambda local y visitante), hiperparámetros tuneados con Optuna (validación temporal expandida).
- **Dixon-Coles** para corregir la dependencia en marcadores bajos y empates.
- **Ensemble** de la matriz XGBoost con un Dixon-Coles clásico (ataque/defensa por equipo), peso optimizado en validación.
- **Evaluación honesta:** predicciones congeladas pre-partido, log-loss / Brier / accuracy / RPS y log-loss de marcador exacto, con calibración.

### Correr el modelo

```bash
cd worldcup-predictor
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python src/update_data.py        # descarga el dataset (Kaggle) y prepara features
python src/cli.py                # CLI interactiva: predicciones, tracking, valor
python src/sync_supabase.py      # sube predicciones/resultados/métricas a Supabase
```

Datos: [International football results 1872-2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle).

## La web (`mundialista-26/`)

Next.js 16 (App Router, RSC) + Tailwind v4, tema oscuro. Lee de Supabase con la **anon key** (solo lectura vía RLS); ninguna escritura desde el navegador.

- **/** partidos de hoy y próximos, con barras 1X2 y marcador probable.
- **/partido/[id]** detalle con heatmap de la matriz de Poisson, top-5 marcadores e indicadores de personalidad del partido.
- **/tracking** predicción vs resultado real con semáforo por log-loss y la curva del log-loss acumulado vs el azar.
- **/modelo** metodología.

### Correr la web

```bash
cd mundialista-26
npm install
# .env.local: NEXT_PUBLIC_SUPABASE_URL y NEXT_PUBLIC_SUPABASE_ANON_KEY
npm run dev
```

## Flujo de datos

```
partido termina → se registra el marcador (CLI / live_tracking, congela la predicción)
                → sync_supabase.py sube a Supabase
                → la web (force-dynamic) lo refleja al recargar
```
