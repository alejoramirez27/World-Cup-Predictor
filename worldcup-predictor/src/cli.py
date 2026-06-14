"""CLI interactiva del predictor del Mundial 2026 (rich).

    python src/cli.py

Menú: (1) predecir partido, (2) partidos de hoy, (3) tracking en vivo,
(4) registrar resultado, (0) salir.
"""

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from live_tracking import LiveTracker, grade_match, hit_1x2
from model import poisson_matrix, predict_scoreline
from worldcup_2026 import (WorldCup2026Predictor, _norm, over_under,
                           top_scorelines_from_matrix)

console = Console()


# ------------------------------------------------------------- helpers UI ---

def pick_team(predictor: WorldCup2026Predictor, label: str) -> str:
    """Pide un equipo con autocompletado: alias, prefijos y subcadenas;
    si hay varios candidatos muestra una lista numerada."""
    while True:
        text = Prompt.ask(f"[bold cyan]{label}[/]").strip()
        if not text:
            continue
        try:
            return predictor.resolve_team(text)
        except ValueError:
            cands = [t for t in predictor.teams if _norm(text) in _norm(t)]
            if not cands:
                console.print(f"  [red]'{text}' no coincide con ningún "
                              f"clasificado.[/] Prueba otra vez.")
                continue
            for i, t in enumerate(cands, 1):
                console.print(f"    [yellow]{i}[/]. {t}")
            n = IntPrompt.ask("  ¿cuál?", default=1)
            if 1 <= n <= len(cands):
                return cands[n - 1]


def prob_bar(label: str, p: float, color: str, width: int = 32) -> str:
    filled = round(p * width)
    return (f"  {label:<22}[{color}]{'█' * filled}[/]"
            f"{'░' * (width - filled)} {p:6.1%}")


def _heat_style(p: float, pmax: float) -> str:
    """Fondo del heatmap: azul oscuro (raro) -> rojo cálido (probable)."""
    x = (p / pmax) ** 0.5 if pmax > 0 else 0.0
    r, g, b = int(25 + 215 * x), int(25 + 95 * x), int(70 - 45 * x)
    return f"white on rgb({r},{g},{b})"


def show_prediction(predictor: WorldCup2026Predictor,
                    team_a: str, team_b: str) -> None:
    p = predictor.predict(team_a, team_b)
    home, away, rho = p["home"], p["away"], predictor.rho
    lh, la = p["lambda_home"], p["lambda_away"]
    probs = p["outcome_probs"]
    ou = p["over_under_2.5"]

    console.print(Panel(
        f"[bold]{home}[/] vs [bold]{away}[/]\n{p['venue']}\n"
        f"goles esperados: [bold]{lh:.2f} - {la:.2f}[/]",
        title="predicción", border_style="cyan"))

    console.print(prob_bar(f"gana {home}", probs["home_win"], "green"))
    console.print(prob_bar("empate", probs["draw"], "yellow"))
    console.print(prob_bar(f"gana {away}", probs["away_win"], "red"))
    console.print(prob_bar("over 2.5", ou["over"], "magenta"))
    console.print(prob_bar("under 2.5", ou["under"], "blue"))

    top5 = predict_scoreline(lh, la, top=5, rho=rho)
    t = Table(title="marcadores más probables", box=box.SIMPLE)
    t.add_column("marcador", justify="center")
    t.add_column("prob", justify="right")
    for score, prob in top5:
        t.add_row(score, f"{prob:.1%}")
    console.print(t)

    m = poisson_matrix(lh, la, rho=rho)
    pmax = float(m.max())
    h = Table(title=f"matriz de Poisson (filas: {home} ↓, columnas: {away} →)",
              box=box.SQUARE, padding=(0, 1))
    h.add_column("", justify="right", style="bold")
    for j in range(m.shape[1]):
        h.add_column(str(j), justify="center")
    for i in range(m.shape[0]):
        cells = [f"[{_heat_style(m[i, j], pmax)}] {m[i, j]:.1%} [/]"
                 for j in range(m.shape[1])]
        h.add_row(str(i), *cells)
    console.print(h)


def show_today(predictor: WorldCup2026Predictor) -> None:
    today = pd.Timestamp.today().normalize()
    todays = predictor.fixtures[predictor.fixtures["date"] == today]
    if todays.empty:
        console.print(f"[yellow]No hay partidos del fixture hoy "
                      f"({today.date()}).[/]")
        return
    t = Table(title=f"partidos de hoy — {today.date()}", box=box.ROUNDED)
    t.add_column("partido")
    t.add_column("1X2 (L/E/V)")
    t.add_column("top-4 marcadores")
    t.add_column("personalidad")
    for row in todays.itertuples():
        p = predictor.predict(row.home_team, row.away_team)
        o = p["outcome_probs"]
        top4 = top_scorelines_from_matrix(p["score_matrix"], 4)
        top4_str = "\n".join(f"{s}  {pr:>4.0%}" for s, pr in top4)
        fav = p["home"] if p["fav_is_home"] else p["away"]
        over = p["over_under_2.5"]["over"]
        personality = (
            f"[magenta]over2.5 {over:>4.0%}[/]\n"
            f"[cyan]goleada {p['p_goleada_fav']:>4.0%}[/] ({fav})\n"
            f"[yellow]empate  {o['draw']:>4.0%}[/]"
        )
        t.add_row(
            f"{p['home']} vs {p['away']}\n[dim]{row.city}[/]",
            f"[green]{o['home_win']:.0%}[/] / [yellow]{o['draw']:.0%}[/]"
            f" / [red]{o['away_win']:.0%}[/]",
            top4_str,
            personality,
        )
    console.print(t)
    console.print("[dim]personalidad: over2.5 alto + goleada alta = partido "
                  "abierto · empate alto + over bajo = partido cerrado.[/]")


def show_tracking(tracker: LiveTracker) -> None:
    df = tracker.df
    if df.empty:
        console.print("[yellow]Sin partidos registrados todavía.[/]")
        return
    t = Table(title="tracking Mundial 2026", box=box.ROUNDED)
    for col in ("fecha", "partido", "predicción 1X2", "resultado", "grado",
                "1X2", "log-loss"):
        t.add_column(col)
    for _, r in df.iterrows():
        if r["outcome"]:
            res = f"{int(r['home_score'])}-{int(r['away_score'])}"
            g = grade_match(r["logloss_model"])          # color manda el log-loss
            grade = f"[{g['color']}]{g['label']}[/]"
            x12 = "[green]✓[/]" if hit_1x2(r) else "[red]✗[/]"
            ll = f"[{g['color']}]{r['logloss_model']:.3f}[/]"
        else:
            res, grade, x12, ll = "[dim]pendiente[/]", "", "", ""
        t.add_row(str(r["date"]), f"{r['home_team']} vs {r['away_team']}",
                  f"{r['p_home']:.0%}/{r['p_draw']:.0%}/{r['p_away']:.0%}",
                  res, grade, x12, ll)
    console.print(t)
    console.print("[dim]grado por log-loss: [green]ACIERTO[/] <0.70 · "
                  "[yellow]FLOJO[/] 0.70–1.0986 · [red]FALLO[/] >1.0986 (azar). "
                  "Columna 1X2 = acierto direccional puro.[/]")

    m = tracker.metrics()
    if m["n_played"]:
        line = (f"[bold]log-loss modelo: {m['ll_model']:.4f}[/] "
                f"({m['n_played']} partidos) — azar: 1.0986")
        if m["n_market"]:
            line += (f"\nen los {m['n_market']} con odds — modelo "
                     f"{m['ll_model_subset']:.4f} | mercado "
                     f"{m['ll_market']:.4f} | [bold]blend "
                     f"{m['ll_blend']:.4f}[/] (w={m['blend_w']:.2f})")
        else:
            line += f"\nblend: w={m['blend_w']:.2f} ({m['blend_w_source']})"
        console.print(Panel(line, border_style="green"))


def show_value(tracker: LiveTracker) -> None:
    t = tracker.value_table()
    if t.empty:
        console.print("[yellow]Sin partidos próximos con odds. Corre antes: "
                      "python src/odds_fetcher.py --register[/]")
        return
    table = Table(title="detector de valor — modelo vs mercado",
                  box=box.ROUNDED)
    for col in ("partido", "lado", "modelo", "mercado", "diff", "odds", "EV", ""):
        table.add_column(col)
    for _, r in t.iterrows():
        ev_style = "green" if r["ev"] > 0 else "red"
        table.add_row(
            r["match"], r["side"],
            f"{r['p_model']:.1%}", f"{r['p_market']:.1%}",
            f"{r['diff_pp']:+.1f}pp", f"{r['odds']:.2f}",
            f"[{ev_style}]{r['ev']:+.3f}[/]",
            "[bold yellow]<!> alineaciones[/]" if r["warn"] else "",
        )
    console.print(table)
    if t["warn"].any():
        console.print("[yellow]<!> Discrepancias >10pp: verifica noticias de "
                      "alineación/lesiones antes de confiar en el modelo — "
                      "el modelo no las ve, el mercado sí.[/]")


def register_result(predictor: WorldCup2026Predictor,
                    tracker: LiveTracker) -> None:
    console.print("[dim]Registra el resultado ANTES de correr "
                  "update_data.py, para que la predicción congelada sea "
                  "genuinamente pre-partido.[/]")
    a = pick_team(predictor, "equipo 1")
    b = pick_team(predictor, "equipo 2")
    ga = IntPrompt.ask(f"goles de {a}")
    gb = IntPrompt.ask(f"goles de {b}")
    tracker.add_result(a, b, ga, gb)
    show_tracking(tracker)


# ------------------------------------------------------------------- main ---

def main() -> None:
    console.print(Panel("[bold]worldcup-predictor — Mundial 2026[/]",
                        border_style="cyan"))
    with console.status("cargando modelos y features..."):
        predictor = WorldCup2026Predictor()
    tracker = LiveTracker()
    tracker._predictor = predictor  # compartir pipeline ya cargado

    MENU = ("\n[bold cyan]1[/] predecir partido  [bold cyan]2[/] partidos de "
            "hoy  [bold cyan]3[/] tracking en vivo  [bold cyan]4[/] registrar "
            "resultado  [bold cyan]5[/] detector de valor  [bold cyan]6[/] "
            "exportar web  [bold cyan]0[/] salir")
    while True:
        console.print(MENU)
        try:
            op = Prompt.ask("opción",
                            choices=["1", "2", "3", "4", "5", "6", "0"],
                            default="1")
            if op == "0":
                break
            if op == "1":
                show_prediction(predictor,
                                pick_team(predictor, "equipo 1"),
                                pick_team(predictor, "equipo 2"))
            elif op == "2":
                show_today(predictor)
            elif op == "3":
                show_tracking(tracker)
            elif op == "4":
                register_result(predictor, tracker)
            elif op == "5":
                show_value(tracker)
            elif op == "6":
                out = tracker.export_web()
                console.print(f"[green]export ->[/] {out / 'index.html'}")
        except (KeyboardInterrupt, EOFError):
            break
    console.print("[dim]hasta luego[/]")


if __name__ == "__main__":
    main()
