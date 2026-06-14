// Mercados derivados de la matriz de marcadores (Poisson + Dixon-Coles).
// Todo se calcula desde el poisson_matrix que ya viene en Supabase: no
// requiere tocar el modelo. Solo mercados de GOLES (es lo que el modelo
// estima); no hay corners ni remates porque el dataset no los tiene.

type Matrix = number[][];

const sum = (m: Matrix, pred: (i: number, j: number) => boolean): number => {
  let s = 0;
  for (let i = 0; i < m.length; i++)
    for (let j = 0; j < m[i].length; j++) if (pred(i, j)) s += m[i][j];
  return s;
};

export type Market = { label: string; prob: number };
export type MarketGroup = { title: string; items: Market[] };

export function overUnder(m: Matrix, line: number): { over: number; under: number } {
  const under = sum(m, (i, j) => i + j < line);
  return { under, over: 1 - under };
}

export function totalGoalsDist(m: Matrix): Market[] {
  const n = m.length;
  const maxTotal = 2 * (n - 1);
  const out: Market[] = [];
  for (let k = 0; k <= Math.min(maxTotal, 6); k++) {
    out.push({ label: k === 6 ? "6+" : `${k}`, prob: sum(m, (i, j) => (k === 6 ? i + j >= 6 : i + j === k)) });
  }
  return out;
}

export function buildMarketGroups(
  m: Matrix,
  home: string,
  away: string
): MarketGroup[] {
  const pHome = sum(m, (i, j) => i > j);
  const pDraw = sum(m, (i, j) => i === j);
  const pAway = sum(m, (i, j) => i < j);

  const ou = (line: number) => overUnder(m, line).over;

  return [
    {
      title: "Doble oportunidad",
      items: [
        { label: `${home} o empate`, prob: pHome + pDraw },
        { label: `${home} o ${away}`, prob: pHome + pAway },
        { label: `Empate o ${away}`, prob: pDraw + pAway },
      ],
    },
    {
      title: "Más / Menos goles",
      items: [
        { label: "Más de 0.5", prob: ou(0.5) },
        { label: "Más de 1.5", prob: ou(1.5) },
        { label: "Más de 2.5", prob: ou(2.5) },
        { label: "Más de 3.5", prob: ou(3.5) },
      ],
    },
    {
      title: "Ambos marcan",
      items: [
        { label: "Sí", prob: sum(m, (i, j) => i >= 1 && j >= 1) },
        { label: "No", prob: sum(m, (i, j) => i === 0 || j === 0) },
      ],
    },
    {
      title: "Portería a cero",
      items: [
        { label: `${home} sin recibir`, prob: sum(m, (_i, j) => j === 0) },
        { label: `${away} sin recibir`, prob: sum(m, (i) => i === 0) },
      ],
    },
    {
      title: "Margen de victoria",
      items: [
        { label: `${home} por 2+`, prob: sum(m, (i, j) => i - j >= 2) },
        { label: `${home} por 1`, prob: sum(m, (i, j) => i - j === 1) },
        { label: "Empate", prob: pDraw },
        { label: `${away} por 1`, prob: sum(m, (i, j) => j - i === 1) },
        { label: `${away} por 2+`, prob: sum(m, (i, j) => j - i >= 2) },
      ],
    },
  ];
}
