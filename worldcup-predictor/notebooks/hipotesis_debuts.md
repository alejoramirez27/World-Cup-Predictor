# Hipótesis: cautela de debut en fase de grupos

## H-001 — Registrada 2026-06-13

### Observación
El modelo parece **sobreestimar a los favoritos en los partidos de debut**
de fase de grupos:
- Empates inesperados en **Qatar–Suiza** y **Brasil–Marruecos** (favorito no rompe).
- **Canadá 1-1 Bosnia**: el favorito (anfitrión) tampoco rompe — el modelo daba
  71% a Canadá y terminó empate (log-loss 1.65, el peor del torneo hasta ahora).

### Hipótesis
Los debuts de Mundial son **más cerrados** de lo que el modelo espera: nervios,
planteamientos conservadores y poco ritmo en el primer partido comprimen los
marcadores. El modelo, entrenado sobre todos los partidos, asigna a los favoritos
una probabilidad de ganar (y de golear) mayor que la que se materializa en una
jornada inaugural.

### Test
Al terminar la **jornada 1 completa** (~el primer partido de los 12 grupos):
1. Calcular el log-loss del modelo **solo en partidos de debut** (primer partido
   de ambos equipos en el torneo).
2. Compararlo con el **log-loss histórico en validación** (baseline 2024-2025:
   **0.8602** en 1X2; referencia azar 1.0986).

### Criterio de decisión
- **Si el log-loss de debuts es consistentemente peor (diferencia > 0.10)** →
  implementar un ajuste de **"cautela de debut"**: reducir los lambdas (goles
  esperados) en el primer partido de cada equipo, comprimiendo la distribución
  hacia marcadores más cerrados / más empates.
- **Si converge al histórico** → era **ruido** de muestra pequeña; no se cambia
  nada y se cierra la hipótesis.

### Notas
- Muestra esperada en jornada 1: ~12 partidos → un solo dato ruidoso pesa mucho;
  la diferencia >0.10 es el umbral mínimo para tomar la señal en serio.
- Cuidado con el sesgo de confirmación: los empates llaman la atención, pero
  **México 2-0 Sudáfrica** (debut, log-loss 0.24) fue un favorito que sí cumplió.
  El test agregado sobre los 12 evita anclarse en los casos llamativos.
- Si se implementa, el ajuste debe ser **pre-partido** (no usar el resultado) y
  evaluable: marcar el debut con el conteo de partidos por equipo en el torneo
  (ya existe la lógica de `_tourn_counts` en `features.py`), y validar el efecto
  contra esta misma métrica antes de congelarlo.

### Estado: ABIERTA — esperando fin de jornada 1
```
Partidos de debut registrados hasta hoy (tracking):
  2026-06-11  Mexico 2-0 South Africa          ll 0.24  ACIERTO  (favorito cumplió)
  2026-06-11  South Korea 2-1 Czech Republic   ll 0.70  FLOJO
  2026-06-12  Canada 1-1 Bosnia                ll 1.65  FALLO    (favorito no rompió)
  2026-06-12  United States 4-1 Paraguay       ll 1.01  FLOJO    (favorito ganó goleando)
```
