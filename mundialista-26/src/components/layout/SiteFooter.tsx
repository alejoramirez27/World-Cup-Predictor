import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-border mt-16">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 text-sm text-faint">
        <p>
          Modelo propio: Elo ponderado + Dixon-Coles + ensemble. Predicciones
          congeladas pre-partido.
        </p>
        <Link href="/modelo" className="text-muted hover:text-fg transition-colors">
          Cómo funciona el modelo
        </Link>
      </div>
    </footer>
  );
}
