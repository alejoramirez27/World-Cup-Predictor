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
