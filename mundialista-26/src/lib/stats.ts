// Estimación de remates y corners a partir de los goles esperados (λ) del
// modelo. Coeficientes CALIBRADOS con 150 partidos reales de torneos
// internacionales (Mundiales + Euros, StatsBomb Open Data):
//   remates_equipo  = 11.24 + 1.456 · λ   (R² = 0.08)
//   corners_equipo  =  4.22 + 0.304 · λ   (R² = 0.02)
// El R² es bajo a propósito: los goles predicen remates/corners de forma
// débil, así que esto es una ESTIMACIÓN aproximada (cercana al promedio del
// torneo, ajustada por favoritismo), no una predicción precisa.

export const CALIBRATION = {
  nMatches: 150,
  source: "Mundiales y Euros (StatsBomb Open Data)",
  shots: { a: 11.24, b: 1.456, r2: 0.08 },
  corners: { a: 4.224, b: 0.304, r2: 0.02 },
};

const clamp0 = (x: number) => (x > 0 ? x : 0);

export function estShots(lambda: number): number {
  return clamp0(CALIBRATION.shots.a + CALIBRATION.shots.b * lambda);
}

export function estCorners(lambda: number): number {
  return clamp0(CALIBRATION.corners.a + CALIBRATION.corners.b * lambda);
}

export function matchStatsEstimate(lambdaHome: number, lambdaAway: number) {
  return {
    shots: { home: estShots(lambdaHome), away: estShots(lambdaAway) },
    corners: { home: estCorners(lambdaHome), away: estCorners(lambdaAway) },
  };
}

// --- probabilidades de líneas (over/under) ---
// El conteo se modela como Poisson sobre la media estimada. Es una
// aproximación (los conteos están ligeramente sobredispersos); combinada con
// el R² bajo de la media, estas probabilidades son orientativas.

function poissonPmf(k: number, mean: number): number {
  let f = 1;
  for (let i = 2; i <= k; i++) f *= i;
  return (Math.exp(-mean) * mean ** k) / f;
}

/** P(X > line) con line tipo x.5; suma de Poissons es Poisson(media total). */
export function overProb(mean: number, line: number): number {
  const kmax = Math.floor(line);
  let cdf = 0;
  for (let k = 0; k <= kmax; k++) cdf += poissonPmf(k, mean);
  return Math.min(Math.max(1 - cdf, 0), 1);
}

export type Line = { line: number; over: number };

export function cornerShotMarkets(lambdaHome: number, lambdaAway: number) {
  const c = { home: estCorners(lambdaHome), away: estCorners(lambdaAway) };
  const s = { home: estShots(lambdaHome), away: estShots(lambdaAway) };
  const lines = (mean: number, ls: number[]): Line[] =>
    ls.map((line) => ({ line, over: overProb(mean, line) }));
  return {
    corners: {
      home: lines(c.home, [3.5, 4.5]),
      away: lines(c.away, [3.5, 4.5]),
      total: lines(c.home + c.away, [8.5, 9.5, 10.5]),
      expTotal: c.home + c.away,
    },
    shots: {
      home: lines(s.home, [10.5, 12.5]),
      away: lines(s.away, [10.5, 12.5]),
      total: lines(s.home + s.away, [22.5, 24.5, 26.5]),
      expTotal: s.home + s.away,
    },
  };
}
