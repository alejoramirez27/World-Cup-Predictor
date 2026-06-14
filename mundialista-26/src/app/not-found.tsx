import Link from "next/link";

export default function NotFound() {
  return (
    <div className="py-20 text-center">
      <p className="tnum text-5xl font-semibold text-accent">404</p>
      <h1 className="mt-3 text-xl font-semibold tracking-tight">Página no encontrada</h1>
      <p className="mt-2 text-muted">Ese partido o ruta no existe.</p>
      <Link
        href="/"
        className="mt-6 inline-block rounded-card border border-border px-4 py-2 text-sm hover:border-accent/60 transition-colors"
      >
        Volver al inicio
      </Link>
    </div>
  );
}
