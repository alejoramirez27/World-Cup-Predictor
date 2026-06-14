import {
  Database,
  ChartLineUp,
  MathOperations as FnIcon,
  Sliders,
  Strategy,
  SealCheck,
} from "@phosphor-icons/react/dist/ssr";

export const metadata = { title: "Modelo · mundialista·26" };

const PIPELINE = [
  { icon: Database, label: "Resultados 1872-2026", sub: "Kaggle, actualizado" },
  { icon: ChartLineUp, label: "Elo ponderado", sub: "warm-up desde 1990" },
  { icon: Sliders, label: "Features por rival", sub: "goles y forma ajustados" },
  { icon: FnIcon, label: "XGBoost Poisson", sub: "λ local y visitante" },
  { icon: Strategy, label: "Ensemble + Dixon-Coles", sub: "matriz de marcadores" },
];

const KFACTORS = [
  { torneo: "Mundial", k: 60 },
  { torneo: "Eliminatorias y copas continentales", k: 50 },
  { torneo: "Nations League", k: 40 },
  { torneo: "Amistosos", k: 20 },
];

export default function ModeloPage() {
  return (
    <div className="space-y-20">
      {/* intro */}
      <section className="anim-rise max-w-2xl">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
          Cómo predice el modelo
        </h1>
        <p className="mt-4 text-lg text-muted leading-relaxed">
          No es un clasificador que escupe un ganador. Estima los goles esperados
          de cada selección, construye la distribución completa de marcadores y
          de ahí derivan el 1X2, el over/under y la probabilidad de cada
          resultado exacto. Cada pieza se eligió midiendo su efecto en datos de
          validación, no por intuición.
        </p>
      </section>

      {/* pipeline: full-width */}
      <section>
        <h2 className="text-sm font-semibold text-faint mb-5">El pipeline</h2>
        <ol className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {PIPELINE.map((step, i) => (
            <li key={step.label} className="rounded-card border border-border bg-surface p-4">
              <div className="flex items-center gap-2 text-accent">
                <step.icon size={20} />
                <span className="tnum text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              </div>
              <div className="mt-3 font-medium text-sm">{step.label}</div>
              <div className="text-xs text-faint mt-0.5">{step.sub}</div>
            </li>
          ))}
        </ol>
      </section>

      {/* Elo: 2-col */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Elo ponderado por torneo</h2>
          <p className="mt-3 text-muted leading-relaxed">
            Cada selección tiene un rating Elo que sube o baja tras cada partido.
            Cuánto se mueve depende de la importancia del torneo: ganar un Mundial
            pesa el triple que un amistoso. El ajuste usa el margen de goles del
            Elo de fútbol estándar y un bonus de localía. Los ratings arrancan en
            1990, así que ya están convergidos cuando empieza el periodo de
            entrenamiento.
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface divide-y divide-border-soft">
          {KFACTORS.map((f) => (
            <div key={f.torneo} className="flex items-center justify-between px-4 py-3">
              <span className="text-sm text-muted">{f.torneo}</span>
              <span className="tnum text-lg font-semibold text-accent">K {f.k}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Dixon-Coles: 2-col invertido con diagrama */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        <div className="order-2 lg:order-1">
          <div className="rounded-card border border-border bg-surface p-6">
            <p className="text-xs text-faint mb-3">Celdas corregidas (τ)</p>
            <div className="grid grid-cols-2 gap-2 max-w-[220px]">
              {["0-0", "0-1", "1-0", "1-1"].map((s) => (
                <div
                  key={s}
                  className="tnum aspect-square rounded-card border border-accent/40 bg-accent-soft flex items-center justify-center text-accent font-medium"
                >
                  {s}
                </div>
              ))}
            </div>
            <p className="tnum text-sm text-muted mt-4">ρ = -0.048</p>
          </div>
        </div>
        <div className="order-1 lg:order-2">
          <h2 className="text-xl font-semibold tracking-tight">Corrección Dixon-Coles</h2>
          <p className="mt-3 text-muted leading-relaxed">
            Una Poisson pura asume que los goles de cada equipo son
            independientes, y subestima sistemáticamente los marcadores bajos y
            los empates. El ajuste Dixon-Coles (1997) corrige exactamente las
            cuatro celdas problemáticas (0-0, 1-0, 0-1, 1-1) con un parámetro ρ
            estimado por máxima verosimilitud. Mejora el log-loss de marcador sin
            tocar el resto de la matriz.
          </p>
        </div>
      </section>

      {/* Ensemble: centrado con barra de blend */}
      <section className="max-w-2xl mx-auto text-center">
        <h2 className="text-xl font-semibold tracking-tight">El ensemble</h2>
        <p className="mt-3 text-muted leading-relaxed">
          La matriz final mezcla dos modelos que se equivocan distinto: el
          XGBoost (features ricas) y un Dixon-Coles clásico de ataque y defensa
          por equipo. El peso se optimizó en validación.
        </p>
        <div className="mt-6 flex h-10 w-full overflow-hidden rounded-card border border-border">
          <div className="flex items-center justify-center bg-accent-soft text-accent text-sm font-medium" style={{ width: "55%" }}>
            XGBoost 55%
          </div>
          <div className="flex items-center justify-center bg-surface-2 text-muted text-sm font-medium" style={{ width: "45%" }}>
            Dixon-Coles 45%
          </div>
        </div>
      </section>

      {/* Features: dos cards */}
      <section>
        <h2 className="text-xl font-semibold tracking-tight mb-5">Features ajustadas por rival</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-card border border-border bg-surface p-5">
            <h3 className="font-medium">Goles relativos a la expectativa</h3>
            <p className="mt-2 text-sm text-muted leading-relaxed">
              Marcar dos goles a una potencia no es lo mismo que marcárselos a un
              rival débil. Cada gol se divide por lo que se esperaría según el Elo
              del rival, así la racha mide rendimiento real, no calendario.
            </p>
          </div>
          <div className="rounded-card border border-border bg-surface p-5">
            <h3 className="font-medium">Forma ponderada por dificultad</h3>
            <p className="mt-2 text-sm text-muted leading-relaxed">
              Empatarle a Francia vale más que ganarle a un rival menor. La forma
              reciente usa rendimiento contra la expectativa Elo, y los amistosos
              pesan menos que los partidos de competición.
            </p>
          </div>
        </div>
      </section>

      {/* Honestidad: callout full-width */}
      <section className="rounded-lg border border-border bg-surface p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <SealCheck size={28} className="text-good shrink-0 mt-0.5" />
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Evaluación honesta</h2>
            <p className="mt-3 text-muted leading-relaxed max-w-2xl">
              Cada predicción se congela antes del partido y se compara con el
              resultado real usando log-loss, la métrica que premia la confianza
              calibrada y castiga la sobreconfianza. La referencia es el azar
              uniforme (1.0986): por debajo, el modelo aporta información. Todo el
              historial está abierto en la página de seguimiento, aciertos y
              fallos incluidos.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
