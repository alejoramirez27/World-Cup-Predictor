"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { SoccerBall } from "@phosphor-icons/react/dist/ssr";
import { cn } from "@/lib/format";

const LINKS = [
  { href: "/", label: "Inicio" },
  { href: "/tracking", label: "Tracking" },
  { href: "/modelo", label: "Modelo" },
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/85 backdrop-blur">
      <nav className="h-16 max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <SoccerBall size={20} weight="fill" className="text-accent" />
          <span>
            mundialista<span className="text-accent">·26</span>
          </span>
        </Link>
        <ul className="flex items-center gap-1 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <li key={l.href}>
                <Link
                  href={l.href}
                  className={cn(
                    "px-3 py-1.5 rounded-card transition-colors",
                    active ? "text-fg bg-surface-2" : "text-muted hover:text-fg"
                  )}
                >
                  {l.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
