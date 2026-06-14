// Formato y escalas compartidas (espejo de la lógica del proyecto Python).

export const DRAW_LL = Math.log(3); // 1.0986 — log-loss del azar uniforme 1X2

export type GradeLevel = "good" | "weak" | "bad" | "none";

export type Grade = {
  label: string;
  level: GradeLevel;
  color: string; // var() del semáforo
};

/** Grado por log-loss del partido: ACIERTO <0.70, FLOJO 0.70-1.0986, FALLO >1.0986. */
export function gradeOf(logloss: number | null | undefined): Grade {
  if (logloss == null || Number.isNaN(logloss))
    return { label: "", level: "none", color: "var(--color-faint)" };
  if (logloss < 0.7) return { label: "ACIERTO", level: "good", color: "var(--color-good)" };
  if (logloss <= DRAW_LL) return { label: "FLOJO", level: "weak", color: "var(--color-weak)" };
  return { label: "FALLO", level: "bad", color: "var(--color-bad)" };
}

export const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

/** Color de celda del heatmap: accent secuencial sobre fondo oscuro. */
export function heatColor(p: number, pmax: number): string {
  const t = pmax > 0 ? Math.min(p / pmax, 1) : 0;
  const alpha = 0.05 + 0.92 * Math.sqrt(t); // raíz para dar peso a celdas medias
  return `color-mix(in oklab, var(--color-accent) ${(alpha * 100).toFixed(1)}%, transparent)`;
}

/** Etiqueta de marcador: índice máximo de la matriz = "N+". */
export function scoreLabel(i: number, j: number, n: number): string {
  const a = i === n - 1 ? `${i}+` : `${i}`;
  const b = j === n - 1 ? `${j}+` : `${j}`;
  return `${a}-${b}`;
}

/** Equipo favorito según probabilidades 1X2. */
export function favorite(p: PredictionLike): "home" | "away" | "draw" {
  const { prob_home, prob_draw, prob_away } = p;
  const max = Math.max(prob_home, prob_draw, prob_away);
  if (max === prob_home) return "home";
  if (max === prob_away) return "away";
  return "draw";
}

type PredictionLike = { prob_home: number; prob_draw: number; prob_away: number };

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
