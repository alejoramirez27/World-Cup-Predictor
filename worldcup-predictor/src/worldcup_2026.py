"""Mundial 2026: equipos clasificados, fixture de fase de grupos y predicciones.

Los 48 equipos y los 72 partidos de grupos se extraen del propio results.csv
(la versión actual del dataset trae el fixture con marcadores vacíos). Los
grupos se reconstruyen por componentes conexas del grafo de enfrentamientos
(cada grupo de 4 juega 6 partidos entre sí) y se etiquetan A-L por orden de
primer partido, que coincide con el calendario oficial (México abre el A).

Uso:
    python src/worldcup_2026.py                      # grupos + demo
    python src/worldcup_2026.py Mexico "South Africa"
    python src/worldcup_2026.py españa argentina     # acepta alias en español
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data_loader import RAW_DIR
from features import FeatureBuilder
from model import (MODELS_DIR, TRAIN_END, TRAIN_START, estimate_rho,
                   outcome_probs, poisson_matrix, predict_match, prepare_data,
                   predict_scoreline, save_models, train_models)

# alias frecuentes (en minúsculas y sin acentos) -> nombre en el dataset
ALIASES = {
    "usa": "United States", "eeuu": "United States",
    "estados unidos": "United States",
    "corea del sur": "South Korea", "corea": "South Korea",
    "espana": "Spain", "alemania": "Germany", "francia": "France",
    "inglaterra": "England", "paises bajos": "Netherlands",
    "holanda": "Netherlands", "belgica": "Belgium", "suiza": "Switzerland",
    "marruecos": "Morocco", "japon": "Japan", "brasil": "Brazil",
    "turquia": "Turkey", "turkiye": "Turkey", "croacia": "Croatia",
    "noruega": "Norway", "escocia": "Scotland", "portugal": "Portugal",
    "sudafrica": "South Africa", "egipto": "Egypt", "argelia": "Algeria",
    "tunez": "Tunisia", "iran": "Iran", "ir iran": "Iran",
    "arabia saudita": "Saudi Arabia", "catar": "Qatar",
    "nueva zelanda": "New Zealand", "haiti": "Haiti", "panama": "Panama",
    "curazao": "Curaçao", "curacao": "Curaçao",
    "costa de marfil": "Ivory Coast", "cabo verde": "Cape Verde",
    "republica checa": "Czech Republic", "chequia": "Czech Republic",
    "bosnia": "Bosnia and Herzegovina", "italia": "Italy",
    "dinamarca": "Denmark", "suecia": "Sweden", "polonia": "Poland",
    "ucrania": "Ukraine", "austria": "Austria", "irlanda": "Republic of Ireland",
    "jordania": "Jordan", "uzbekistan": "Uzbekistan",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def load_group_fixtures() -> pd.DataFrame:
    """Los 72 partidos de fase de grupos del Mundial 2026 tal como vienen
    en results.csv (incluye sede y flag neutral reales)."""
    df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    return wc.sort_values("date").reset_index(drop=True)


def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Reconstruye los 12 grupos por componentes conexas del fixture."""
    parent: dict[str, str] = {}

    def find(t: str) -> str:
        parent.setdefault(t, t)
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for row in fixtures.itertuples():
        parent[find(row.home_team)] = find(row.away_team)

    comps: dict[str, list[str]] = {}
    for team in parent:
        comps.setdefault(find(team), []).append(team)

    # etiquetar A-L por fecha del primer partido de cada componente
    first_date = {
        root: fixtures[
            fixtures["home_team"].isin(teams) | fixtures["away_team"].isin(teams)
        ]["date"].min()
        for root, teams in comps.items()
    }
    ordered = sorted(comps, key=lambda r: first_date[r])
    return {
        chr(ord("A") + i): sorted(comps[root]) for i, root in enumerate(ordered)
    }


def over_under(lam_home: float, lam_away: float, line: float = 2.5,
               rho: float = 0.0) -> dict[str, float]:
    """P(total de goles por encima/debajo de la línea) desde la matriz DC."""
    m = poisson_matrix(lam_home, lam_away, rho=rho)
    n = m.shape[0]
    under = float(sum(m[i, j] for i in range(n) for j in range(n)
                      if i + j < line))
    return {"under": under, "over": 1.0 - under}


def top_scorelines_from_matrix(m: np.ndarray, top: int = 4) -> list[tuple[str, float]]:
    """Los `top` marcadores más probables de una matriz (índice máximo = '6+')."""
    n = m.shape[0]
    flat = [(f"{i}-{j}" if max(i, j) < n - 1 else f"{i}-{j}".replace(
                str(n - 1), f"{n - 1}+"), float(m[i, j]))
            for i in range(n) for j in range(n)]
    return sorted(flat, key=lambda t: t[1], reverse=True)[:top]


def match_personality(m: np.ndarray, probs: dict) -> dict:
    """Indicadores de 'personalidad' del partido desde la matriz final:
    P(goleada del favorito = ganar por 2+) por lado. over 2.5 y empate ya
    vienen en over_under / outcome_probs."""
    n = m.shape[0]
    home_by2 = float(sum(m[i, j] for i in range(n) for j in range(n)
                         if i - j >= 2))
    away_by2 = float(sum(m[i, j] for i in range(n) for j in range(n)
                         if j - i >= 2))
    fav_home = probs["home_win"] >= probs["away_win"]
    return {
        "p_home_by2": home_by2,
        "p_away_by2": away_by2,
        "fav_is_home": fav_home,
        "p_goleada_fav": home_by2 if fav_home else away_by2,
    }


class WorldCup2026Predictor:
    """Predicciones para el Mundial 2026 por nombre de equipos."""

    def __init__(self, retrain: bool = False) -> None:
        self.fixtures = load_group_fixtures()
        self.groups = infer_groups(self.fixtures)
        self.teams = sorted(
            set(self.fixtures["home_team"]) | set(self.fixtures["away_team"])
        )

        print("Preparando features (warm-up Elo 1990)...")
        self.df, self.builder = prepare_data()
        self.dc = None          # Dixon-Coles clásico del ensemble congelado
        self.ensemble_w = 1.0   # peso del XGBoost en la matriz final
        self.models = self._load_or_train(retrain)
        if "rho" in self.models:  # modelo congelado: rho viene del meta
            self.rho = self.models["rho"]
        else:
            cats = self.df["match_type"].astype("category").cat.categories
            train_df = self.df[self.df["year"].between(TRAIN_START, TRAIN_END)]
            self.rho = estimate_rho(self.models, train_df, cats)

    def _load_or_train(self, retrain: bool) -> dict:
        # política de torneo: si existe el modelo final congelado, SIEMPRE
        # se usa ese (las features sí se recalculan con datos nuevos, el
        # modelo no se toca)
        final_dir = MODELS_DIR / "model_final_wc2026"
        if final_dir.exists():
            import json

            from dixon_coles import DixonColesModel
            meta = json.loads((final_dir / "meta.json").read_text())
            models = {"feature_cols": meta["feature_cols"], "rho": meta["rho"]}
            for k in ("home", "away"):
                reg = XGBRegressor(enable_categorical=True)
                reg.load_model(final_dir / f"xgb_{k}.json")
                models[f"model_{k}"] = reg
            dc_path = final_dir / "dc.json"
            if dc_path.exists():
                self.dc = DixonColesModel.from_dict(
                    json.loads(dc_path.read_text()))
                self.ensemble_w = meta.get("ensemble_w", 1.0)
            print(f"Modelo FINAL congelado: {meta['version']} "
                  f"(datos hasta {meta['trained_through']}, "
                  f"rho {meta['rho']:+.4f}"
                  + (f", ensemble w={self.ensemble_w:.2f}" if self.dc else "")
                  + ")")
            return models

        paths = {k: MODELS_DIR / f"xgb_{k}.json" for k in ("home", "away")}
        if not retrain and all(p.exists() for p in paths.values()):
            models = {}
            for k, p in paths.items():
                reg = XGBRegressor(enable_categorical=True)
                reg.load_model(p)
                models[f"model_{k}"] = reg
            print(f"Modelos cargados de {MODELS_DIR}")
            return models
        print("Entrenando modelos...")
        models = train_models(self.df)
        save_models(models)
        return models

    def resolve_team(self, name: str) -> str:
        """Nombre del dataset a partir de un alias o variante de escritura."""
        norm = _norm(name)
        if norm in ALIASES:
            return ALIASES[norm]
        for team in self.teams:
            if _norm(team) == norm:
                return team
        matches = [t for t in self.teams if norm in _norm(t)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Equipo '{name}' no reconocido entre los 48 clasificados. "
            f"¿Quizás uno de estos? {matches or self.teams}"
        )

    def find_fixture(self, team_a: str, team_b: str) -> pd.Series | None:
        f = self.fixtures
        mask = (
            ((f["home_team"] == team_a) & (f["away_team"] == team_b))
            | ((f["home_team"] == team_b) & (f["away_team"] == team_a))
        )
        hits = f[mask]
        return hits.iloc[0] if len(hits) else None

    def predict(self, team_a: str, team_b: str) -> dict:
        """Predicción completa. Si el cruce está en el fixture de grupos usa
        fecha, orden local/visitante y sede reales; si no (hipotético de
        eliminatorias), asume sede neutral a mitad del torneo."""
        team_a, team_b = self.resolve_team(team_a), self.resolve_team(team_b)
        fixture = self.find_fixture(team_a, team_b)
        if fixture is not None:
            home, away = fixture["home_team"], fixture["away_team"]
            date, neutral = fixture["date"], bool(fixture["neutral"])
            city, country, knockout = fixture["city"], fixture["country"], False
            venue = f"{city}, {country} ({date.date()})"
        else:
            home, away = team_a, team_b
            date, neutral = pd.Timestamp("2026-07-01"), True
            city, country, knockout = None, "United States", True
            venue = "cruce hipotético (eliminatorias), sede neutral"

        pred = predict_match(self.models, self.builder, home, away,
                             date, "world_cup", neutral=neutral, rho=self.rho,
                             city=city, country=country, knockout=knockout)
        pred["home"], pred["away"], pred["venue"] = home, away, venue

        if self.dc is not None:
            # ensemble congelado: matriz_final = w*ML + (1-w)*DC
            from features import WC2026_HOSTS
            from model import score_matrix_agg
            home_adv = home in WC2026_HOSTS
            m = (self.ensemble_w
                 * score_matrix_agg(pred["lambda_home"], pred["lambda_away"],
                                    rho=self.rho)
                 + (1 - self.ensemble_w)
                 * self.dc.score_matrix(home, away, home_adv))
            n = m.shape[0]
            pred["outcome_probs"] = {
                "home_win": float(np.tril(m, -1).sum()),
                "draw": float(np.trace(m)),
                "away_win": float(np.triu(m, 1).sum()),
            }
            flat = [(f"{i}-{j}", float(m[i, j]))
                    for i in range(n) for j in range(n)]
            pred["top_scorelines"] = sorted(flat, key=lambda t: t[1],
                                            reverse=True)[:3]
            under = float(sum(m[i, j] for i in range(n) for j in range(n)
                              if i + j < 2.5))
            pred["over_under_2.5"] = {"under": under, "over": 1.0 - under}
        else:
            from model import score_matrix_agg
            m = score_matrix_agg(pred["lambda_home"], pred["lambda_away"],
                                 rho=self.rho)
            pred["over_under_2.5"] = over_under(
                pred["lambda_home"], pred["lambda_away"], 2.5, self.rho
            )
        pred["score_matrix"] = m
        pred.update(match_personality(m, pred["outcome_probs"]))
        return pred

    def show(self, team_a: str, team_b: str) -> None:
        p = self.predict(team_a, team_b)
        probs, ou = p["outcome_probs"], p["over_under_2.5"]
        print(f"\n{p['home']} vs {p['away']}  [{p['venue']}]")
        print(f"  goles esperados: {p['lambda_home']:.2f} - {p['lambda_away']:.2f}")
        print(f"  1X2: {p['home']} {probs['home_win']:.1%} | "
              f"empate {probs['draw']:.1%} | {p['away']} {probs['away_win']:.1%}")
        print(f"  over 2.5: {ou['over']:.1%} | under 2.5: {ou['under']:.1%}")
        print("  marcadores más probables:")
        for score, prob in p["top_scorelines"]:
            print(f"    {score}  ({prob:.1%})")

    def print_groups(self) -> None:
        print(f"\nMundial 2026 — {len(self.teams)} clasificados, "
              f"{len(self.fixtures)} partidos de grupos "
              f"({self.fixtures['date'].min().date()} a "
              f"{self.fixtures['date'].max().date()})")
        for letter, teams in self.groups.items():
            print(f"  Grupo {letter}: {', '.join(teams)}")


if __name__ == "__main__":
    predictor = WorldCup2026Predictor()
    if len(sys.argv) >= 3:
        predictor.show(sys.argv[1], sys.argv[2])
    else:
        predictor.print_groups()
        for a, b in [("Mexico", "South Africa"), ("United States", "Paraguay"),
                     ("Brazil", "Morocco"), ("Spain", "Argentina")]:
            predictor.show(a, b)
