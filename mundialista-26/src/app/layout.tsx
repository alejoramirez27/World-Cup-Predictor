import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { SiteNav } from "@/components/layout/SiteNav";
import { SiteFooter } from "@/components/layout/SiteFooter";

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "https://world-cup-predictor-dusky.vercel.app"
  ),
  title: "mundialista-26 · predicciones Mundial 2026",
  description:
    "Modelo de predicción del Mundial 2026: Elo ponderado, Dixon-Coles y ensemble. Probabilidades 1X2, marcadores y seguimiento honesto del modelo.",
  openGraph: {
    title: "mundialista·26 · predicciones Mundial 2026",
    description:
      "Probabilidades 1X2, marcadores, mercados y seguimiento honesto del modelo.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="es"
      className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-bg text-fg">
        <SiteNav />
        <main className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
