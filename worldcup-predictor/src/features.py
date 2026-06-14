"""Features para predicción, calculadas incrementalmente para evitar data leakage.

Todas las features de una fila se calculan SOLO con partidos anteriores:
el dataframe se procesa en orden cronológico y el estado de cada equipo
(Elo, historial reciente) se actualiza DESPUÉS de registrar las features
del partido actual.

Elo: variante estándar de fútbol (eloratings.net):
  - expectativa con bonus de +100 puntos para el local real (no sede neutral)
  - ajuste por margen de goles: x1 (diff<=1), x1.5 (diff=2), (11+N)/8 (diff>=3)
  - K según importancia del torneo (ver K_FACTORS)
"""

from collections import deque

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
HOME_ELO_BONUS = 100.0

K_FACTORS = {
    "world_cup": 60.0,
    "wc_qualifier": 50.0,
    "continental": 50.0,
    "continental_qualifier": 50.0,
    "nations_league": 40.0,
    "friendly": 20.0,
    "other": 30.0,
}

FRIENDLY_FORM_WEIGHT = 0.4
COMPETITIVE_FORM_WEIGHT = 1.0

# En el Mundial 2026 los tres anfitriones juegan como locales reales en su
# propio país; el resto de selecciones juega en sede neutral.
WC2026_HOSTS = {"United States", "Mexico", "Canada"}

# Altitud (m) de las 16 sedes del Mundial 2026 y de ciudades de altura
# relevantes en el histórico. Ciudad no listada => 0 m (valor neutro:
# la gran mayoría de sedes están cerca del nivel del mar).
CITY_ALTITUDE = {
    # sedes Mundial 2026
    "Mexico City": 2240, "Zapopan": 1548, "Guadalupe": 540, "Monterrey": 540,
    "Toronto": 76, "Vancouver": 0, "Inglewood": 40, "Santa Clara": 25,
    "East Rutherford": 3, "Foxborough": 89, "Arlington": 184, "Houston": 15,
    "Atlanta": 320, "Miami Gardens": 3, "Philadelphia": 12, "Seattle": 56,
    "Kansas City": 270,
    # altura histórica relevante
    "La Paz": 3640, "El Alto": 4090, "Quito": 2850, "Bogotá": 2640,
    "Bogota": 2640, "Cusco": 3400, "Sucre": 2810, "Cochabamba": 2560,
    "Arequipa": 2335, "Toluca": 2660, "Puebla": 2160, "Pachuca": 2430,
    "San Luis Potosí": 1860, "Aguascalientes": 1880, "Querétaro": 1820,
    "Guadalajara": 1566, "Torreón": 1120, "León": 1815,
    "Addis Ababa": 2355, "Asmara": 2325, "Nairobi": 1795, "Kampala": 1190,
    "Kigali": 1567, "Johannesburg": 1753, "Pretoria": 1339,
    "Bloemfontein": 1395, "Harare": 1490, "Lusaka": 1280, "Windhoek": 1655,
    "Tegucigalpa": 990, "Guatemala City": 1500, "San José": 1170,
    "Quetzaltenango": 2330, "Medellín": 1495, "Cali": 1018,
    "Tehran": 1190, "Sana'a": 2250, "Sanaa": 2250, "Kabul": 1790,
    "Kathmandu": 1400, "Thimphu": 2320, "Erzurum": 1900, "Ankara": 938,
    "Madrid": 667, "Yerevan": 990,
}

# Coordenadas aproximadas (capital) por país para distancia de viaje.
# País no listado => sin dato => feature NaN (neutro para XGBoost).
COUNTRY_COORDS = {
    "England": (51.5, -0.1), "Scotland": (55.9, -3.2), "Wales": (51.5, -3.2),
    "Northern Ireland": (54.6, -5.9), "Republic of Ireland": (53.3, -6.3),
    "France": (48.9, 2.3), "Spain": (40.4, -3.7), "Portugal": (38.7, -9.1),
    "Germany": (52.5, 13.4), "Italy": (41.9, 12.5), "Netherlands": (52.4, 4.9),
    "Belgium": (50.8, 4.4), "Switzerland": (46.9, 7.4), "Austria": (48.2, 16.4),
    "Croatia": (45.8, 16.0), "Serbia": (44.8, 20.5),
    "Bosnia and Herzegovina": (43.9, 18.4), "Slovenia": (46.1, 14.5),
    "Slovakia": (48.1, 17.1), "Czech Republic": (50.1, 14.4),
    "Poland": (52.2, 21.0), "Hungary": (47.5, 19.0), "Romania": (44.4, 26.1),
    "Bulgaria": (42.7, 23.3), "Greece": (38.0, 23.7), "Turkey": (39.9, 32.9),
    "Russia": (55.8, 37.6), "Ukraine": (50.5, 30.5), "Belarus": (53.9, 27.6),
    "Lithuania": (54.7, 25.3), "Latvia": (56.9, 24.1), "Estonia": (59.4, 24.8),
    "Finland": (60.2, 24.9), "Sweden": (59.3, 18.1), "Norway": (59.9, 10.8),
    "Denmark": (55.7, 12.6), "Iceland": (64.1, -21.9), "Albania": (41.3, 19.8),
    "North Macedonia": (42.0, 21.4), "Montenegro": (42.4, 19.3),
    "Kosovo": (42.7, 21.2), "Moldova": (47.0, 28.9), "Georgia": (41.7, 44.8),
    "Armenia": (40.2, 44.5), "Azerbaijan": (40.4, 49.9), "Cyprus": (35.2, 33.4),
    "Malta": (35.9, 14.5), "Luxembourg": (49.6, 6.1), "Israel": (32.1, 34.8),
    "United States": (38.9, -77.0), "Mexico": (19.4, -99.1),
    "Canada": (45.4, -75.7), "Brazil": (-15.8, -47.9),
    "Argentina": (-34.6, -58.4), "Uruguay": (-34.9, -56.2),
    "Paraguay": (-25.3, -57.6), "Chile": (-33.4, -70.7), "Peru": (-12.0, -77.0),
    "Bolivia": (-16.5, -68.1), "Ecuador": (-0.2, -78.5),
    "Colombia": (4.7, -74.1), "Venezuela": (10.5, -66.9),
    "Costa Rica": (9.9, -84.1), "Panama": (9.0, -79.5),
    "Honduras": (14.1, -87.2), "El Salvador": (13.7, -89.2),
    "Guatemala": (14.6, -90.5), "Nicaragua": (12.1, -86.3),
    "Jamaica": (18.0, -76.8), "Haiti": (18.5, -72.3),
    "Trinidad and Tobago": (10.7, -61.5), "Curaçao": (12.1, -68.9),
    "Cuba": (23.1, -82.4), "Dominican Republic": (18.5, -69.9),
    "Suriname": (5.9, -55.2),
    "Morocco": (34.0, -6.8), "Algeria": (36.8, 3.1), "Tunisia": (36.8, 10.2),
    "Egypt": (30.0, 31.2), "Libya": (32.9, 13.2), "Senegal": (14.7, -17.5),
    "Ivory Coast": (5.3, -4.0), "Ghana": (5.6, -0.2), "Nigeria": (9.1, 7.4),
    "Cameroon": (3.9, 11.5), "DR Congo": (-4.3, 15.3),
    "South Africa": (-25.7, 28.2), "Zambia": (-15.4, 28.3),
    "Zimbabwe": (-17.8, 31.0), "Kenya": (-1.3, 36.8), "Ethiopia": (9.0, 38.7),
    "Mali": (12.6, -8.0), "Burkina Faso": (12.4, -1.5), "Guinea": (9.6, -13.6),
    "Cape Verde": (14.9, -23.5), "Angola": (-8.8, 13.2),
    "Mozambique": (-25.9, 32.6), "Tanzania": (-6.8, 39.3),
    "Uganda": (0.3, 32.6), "Gabon": (0.4, 9.5), "Benin": (6.4, 2.4),
    "Togo": (6.1, 1.2), "Niger": (13.5, 2.1), "Sudan": (15.6, 32.5),
    "Rwanda": (-1.9, 30.1), "Namibia": (-22.6, 17.1),
    "Botswana": (-24.7, 25.9), "Equatorial Guinea": (3.8, 8.8),
    "Gambia": (13.4, -16.6), "Sierra Leone": (8.5, -13.2),
    "Liberia": (6.3, -10.8), "Mauritania": (18.1, -16.0),
    "Madagascar": (-18.9, 47.5), "Malawi": (-14.0, 33.8),
    "Japan": (35.7, 139.7), "South Korea": (37.6, 127.0),
    "North Korea": (39.0, 125.8), "China PR": (39.9, 116.4),
    "Iran": (35.7, 51.4), "Iraq": (33.3, 44.4), "Saudi Arabia": (24.7, 46.7),
    "Qatar": (25.3, 51.5), "United Arab Emirates": (24.5, 54.4),
    "Kuwait": (29.4, 48.0), "Bahrain": (26.2, 50.6), "Oman": (23.6, 58.4),
    "Yemen": (15.4, 44.2), "Jordan": (31.9, 35.9), "Lebanon": (33.9, 35.5),
    "Syria": (33.5, 36.3), "Uzbekistan": (41.3, 69.2),
    "Kazakhstan": (51.2, 71.4), "Kyrgyzstan": (42.9, 74.6),
    "Tajikistan": (38.6, 68.8), "Turkmenistan": (37.9, 58.4),
    "Afghanistan": (34.5, 69.2), "India": (28.6, 77.2),
    "Pakistan": (33.7, 73.1), "Bangladesh": (23.8, 90.4),
    "Sri Lanka": (6.9, 79.9), "Nepal": (27.7, 85.3),
    "Thailand": (13.8, 100.5), "Vietnam": (21.0, 105.8),
    "Malaysia": (3.1, 101.7), "Singapore": (1.4, 103.8),
    "Indonesia": (-6.2, 106.8), "Philippines": (14.6, 121.0),
    "Hong Kong": (22.3, 114.2), "Chinese Taipei": (25.0, 121.5),
    "Australia": (-35.3, 149.1), "New Zealand": (-41.3, 174.8),
    "Fiji": (-18.1, 178.4),
}

REST_DAYS_CAP = 30  # más de un mes de descanso ya no aporta información
PREP_WINDOW_DAYS = 60  # ventana pre-torneo para marcar amistosos de preparación


def _haversine_km(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (*c1, *c2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def _goal_multiplier(margin: int) -> float:
    """Ajuste por margen de goles del Elo de fútbol estándar."""
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def _expected_home(elo_home: float, elo_away: float, home_advantage: bool) -> float:
    dr = elo_home - elo_away + (HOME_ELO_BONUS if home_advantage else 0.0)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


class GoalExpectationModel:
    """Expectativa de goles según la diferencia de Elo: E[g] = exp(a + b*dr).

    Se ajusta por máxima verosimilitud Poisson sobre los partidos del
    warm-up (1990-2009), de modo que la curva nunca ve el período de
    entrenamiento. Un rival de Elo 2000 'permite' esperar menos goles a
    favor (dr negativo) que uno de 1500."""

    def __init__(self) -> None:
        self.a, self.b = float(np.log(1.4)), 0.0011  # defaults razonables
        self._samples: list[tuple[float, int]] = []

    def add_sample(self, dr: float, goals: int) -> None:
        self._samples.append((dr, goals))

    def fit(self) -> None:
        if len(self._samples) < 500:
            return
        from scipy.optimize import minimize

        dr = np.array([s[0] for s in self._samples])
        g = np.array([s[1] for s in self._samples])

        def nll(p: np.ndarray) -> float:
            eta = p[0] + p[1] * dr
            return float(np.sum(np.exp(eta) - g * eta))

        res = minimize(nll, x0=[self.a, self.b], method="Nelder-Mead")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        self._samples.clear()

    def expected(self, dr: float) -> float:
        return float(np.clip(np.exp(self.a + self.b * dr), 0.2, 4.5))


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _avg(values: list[float], n: int) -> float:
    """Promedio de los últimos n valores; NaN si no hay historial."""
    window = values[-n:]
    return float(np.mean(window)) if window else np.nan


class _TeamState:
    __slots__ = ("elo", "all_matches", "comp_matches", "qualifier_dates",
                 "last_date", "last_country", "alt_hist")

    def __init__(self) -> None:
        self.elo = BASE_ELO
        # tuplas (gf, ga, points, is_competitive); solo se necesitan los últimos 10
        self.all_matches: deque = deque(maxlen=10)
        self.comp_matches: deque = deque(maxlen=10)
        self.qualifier_dates: list[pd.Timestamp] = []
        # contexto: descanso, viaje y altitud habitual
        self.last_date: pd.Timestamp | None = None
        self.last_country: str | None = None
        self.alt_hist: deque = deque(maxlen=20)

    def snapshot(self, cycle_start: pd.Timestamp,
                 prep_weight: float | None = None) -> dict:
        """Features del equipo ANTES del partido actual.

        `prep_weight`: si != None, los amistosos marcados como preparación
        (m[7]) pesan ese valor en la forma en vez de FRIENDLY_FORM_WEIGHT.
        Default None => comportamiento idéntico al original."""
        gf_all = [m[0] for m in self.all_matches]
        ga_all = [m[1] for m in self.all_matches]
        gf_comp = [m[0] for m in self.comp_matches]
        ga_comp = [m[1] for m in self.comp_matches]
        # ratios goles/expectativa-según-Elo-del-rival y rendimiento vs Elo
        gf_adj = [m[4] for m in self.all_matches]
        ga_adj = [m[5] for m in self.all_matches]

        last5 = list(self.all_matches)[-5:]

        def _form_w(m):
            if m[3]:
                return COMPETITIVE_FORM_WEIGHT
            if len(m) > 7 and m[7] and prep_weight is not None:
                return prep_weight
            return FRIENDLY_FORM_WEIGHT

        weights = [_form_w(m) for m in last5]
        form5 = float(sum(m[2] * w for m, w in zip(last5, weights)))
        # forma ajustada por dificultad: (resultado - esperado por Elo),
        # con el mismo peso reducido para amistosos
        form5_perf = float(sum(m[6] * w for m, w in zip(last5, weights)))

        played_qualifiers = any(d >= cycle_start for d in self.qualifier_dates)

        # proporción de partidos competitivos en los últimos 10: versión
        # continua y con soporte en todo el dataset de la señal "calendario
        # blando" que el flag de eliminatorias captura solo en los bordes
        comp_flags = [m[3] for m in self.all_matches]
        comp_share = float(np.mean(comp_flags)) if comp_flags else np.nan

        return {
            "elo": self.elo,
            "gf_avg5": _avg(gf_all, 5),
            "ga_avg5": _avg(ga_all, 5),
            "gf_avg10": _avg(gf_all, 10),
            "ga_avg10": _avg(ga_all, 10),
            "gf_avg5_comp": _avg(gf_comp, 5),
            "ga_avg5_comp": _avg(ga_comp, 5),
            "gf_avg10_comp": _avg(gf_comp, 10),
            "ga_avg10_comp": _avg(ga_comp, 10),
            "form5_weighted": form5,
            "form5_perf": form5_perf,
            "gf_adj5": _avg(gf_adj, 5),
            "ga_adj5": _avg(ga_adj, 5),
            "gf_adj10": _avg(gf_adj, 10),
            "ga_adj10": _avg(ga_adj, 10),
            "played_qualifiers_cycle": played_qualifiers,
            "comp_share10": comp_share,
        }

    def record(self, date: pd.Timestamp, gf: int, ga: int,
               is_competitive: bool, tournament_type: str,
               gf_ratio: float = np.nan, ga_ratio: float = np.nan,
               perf: float = 0.0, is_prep: bool = False) -> None:
        match = (gf, ga, _points(gf, ga), is_competitive, gf_ratio, ga_ratio,
                 perf, is_prep)
        self.all_matches.append(match)
        if is_competitive:
            self.comp_matches.append(match)
        if tournament_type == "wc_qualifier":
            self.qualifier_dates.append(date)
            # solo importa el ciclo actual; recortar para no crecer sin límite
            if len(self.qualifier_dates) > 60:
                self.qualifier_dates = self.qualifier_dates[-60:]


def _home_advantage(tournament_type: str, year: int, home_team: str,
                    neutral) -> bool:
    """Local real vs. sede neutral, con valores explícitos (evita construir
    un pd.Series por fila en los bucles calientes).

    Regla general: el flag `neutral` del dataset. Excepción Mundial 2026:
    México, USA y Canadá son locales reales; el resto juega en neutral
    aunque el fixture los liste como "home".
    """
    if tournament_type == "world_cup" and year == 2026:
        return home_team in WC2026_HOSTS
    return not bool(neutral)


def _resolve_home_advantage(row: pd.Series) -> bool:
    """Wrapper sobre _home_advantage para llamadas con un pd.Series (p. ej.
    match_features), que son puntuales y no están en el camino caliente."""
    return _home_advantage(row["tournament_type"], row["date"].year,
                           row["home_team"], row["neutral"])


class FeatureBuilder:
    """Construye features incrementales y conserva el estado final,
    de modo que también puede generar features para partidos futuros
    (p. ej. fixtures del Mundial 2026) sin recalcular todo."""

    def __init__(self) -> None:
        self.states: dict[str, _TeamState] = {}
        self.wc_end_dates: list[pd.Timestamp] = []
        self.last_date: pd.Timestamp | None = None
        self.goal_model = GoalExpectationModel()
        # partidos jugados por (equipo, torneo, año) para inferir knockout
        self._tourn_counts: dict[tuple[str, str, int], int] = {}
        # ventana de preparación: peso de amistosos pre-torneo en la forma
        # (None = sin trato especial). _prep_tourn_dates[team] = array
        # ordenado np.datetime64 de fechas de torneos mayores del equipo.
        self.prep_friendly_weight: float | None = None
        self._prep_tourn_dates: dict[str, np.ndarray] = {}

    def _is_prep(self, team: str, date: pd.Timestamp) -> bool:
        """¿Es `date` un amistoso dentro de PREP_WINDOW_DAYS antes de un
        torneo mayor del equipo? La agenda es conocida de antemano, así que
        no hay leakage."""
        dates = self._prep_tourn_dates.get(team)
        if dates is None or len(dates) == 0:
            return False
        lo = np.datetime64(date)
        hi = lo + np.timedelta64(PREP_WINDOW_DAYS, "D")
        i = int(np.searchsorted(dates, lo, side="right"))
        return i < len(dates) and dates[i] <= hi

    def _context_features(self, hs: _TeamState, as_: _TeamState,
                          date: pd.Timestamp, city: str | None,
                          country: str | None, is_knockout: bool) -> dict:
        """Features de contexto pre-partido. Valores neutros sin dato:
        altitud de ciudad desconocida = 0 m (nivel del mar), descanso y
        viaje sin historial = NaN (missing nativo de XGBoost)."""
        venue_alt = float(CITY_ALTITUDE.get(city, 0.0)) if city else 0.0
        cur_coords = COUNTRY_COORDS.get(country) if country else None
        ctx: dict = {"is_knockout": bool(is_knockout)}
        for side, st in (("home", hs), ("away", as_)):
            if st.last_date is not None:
                ctx[f"{side}_rest_days"] = float(
                    min((date - st.last_date).days, REST_DAYS_CAP))
            else:
                ctx[f"{side}_rest_days"] = np.nan
            usual = float(np.mean(st.alt_hist)) if st.alt_hist else 0.0
            ctx[f"{side}_alt_diff"] = venue_alt - usual
            prev_coords = (COUNTRY_COORDS.get(st.last_country)
                           if st.last_country else None)
            ctx[f"{side}_travel_km"] = (
                _haversine_km(prev_coords, cur_coords)
                if prev_coords and cur_coords else np.nan)
        return ctx

    def _state(self, team: str) -> _TeamState:
        if team not in self.states:
            self.states[team] = _TeamState()
        return self.states[team]

    def _cycle_start(self, date: pd.Timestamp) -> pd.Timestamp:
        """Inicio del ciclo mundialista vigente: fin del último Mundial
        anterior a `date` (o el inicio de los tiempos si no hay ninguno)."""
        start = pd.Timestamp.min
        for end in self.wc_end_dates:
            if end < date:
                start = end
            else:
                break
        return start

    def warm_up(self, df: pd.DataFrame) -> None:
        """Corre el Elo incremental sobre partidos históricos SIN generar
        features, para que los ratings ya estén convergidos al inicio del
        período de entrenamiento. Solo actualiza Elo: las features de goles
        recientes y forma usan únicamente los últimos 10 partidos, así que
        no necesitan warm-up."""
        df = df.sort_values("date", kind="stable")
        for row in df.itertuples(index=False):
            home_adv = _home_advantage(
                row.tournament_type, row.date.year, row.home_team, row.neutral)
            hs = self._state(row.home_team)
            as_ = self._state(row.away_team)
            # muestras (elo_diff, goles) para la curva de expectativa,
            # con los Elo PRE-partido
            dr = hs.elo - as_.elo + (HOME_ELO_BONUS if home_adv else 0.0)
            gh, ga = int(row.home_score), int(row.away_score)
            self.goal_model.add_sample(dr, gh)
            self.goal_model.add_sample(-dr, ga)
            self._update_elo(hs, as_, gh, ga, home_adv, row.tournament_type)
            self.last_date = row.date
        self.goal_model.fit()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa el histórico en orden cronológico y devuelve el dataframe
        con las features pre-partido agregadas. Si se llamó `warm_up()` antes,
        los equipos parten con su Elo ya convergido en vez de 1500."""
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

        # fechas de cierre de cada edición del Mundial presente en los datos
        wc = df[df["tournament_type"] == "world_cup"]
        self.wc_end_dates = sorted(wc.groupby(wc["date"].dt.year)["date"].max())

        rows = []
        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            date = row.date
            ttype = row.tournament_type
            home_adv = _home_advantage(ttype, date.year, home, row.neutral)
            cycle_start = self._cycle_start(date)

            hs, as_ = self._state(home), self._state(away)
            feats = self._pair_features(hs, as_, home_adv, ttype, cycle_start)

            # knockout inferido: 4o+ partido de ambos equipos en la misma
            # edición de un torneo con fase de grupos (desde 2010 la fase de
            # grupos es de 3 partidos en Mundial y copas continentales)
            tourn, yr = row.tournament, date.year
            is_ko = False
            if ttype in ("world_cup", "continental"):
                is_ko = (self._tourn_counts.get((home, tourn, yr), 0) >= 3
                         and self._tourn_counts.get((away, tourn, yr), 0) >= 3)
                self._tourn_counts[(home, tourn, yr)] = \
                    self._tourn_counts.get((home, tourn, yr), 0) + 1
                self._tourn_counts[(away, tourn, yr)] = \
                    self._tourn_counts.get((away, tourn, yr), 0) + 1
            feats.update(self._context_features(
                hs, as_, date, row.city, row.country, is_ko))
            rows.append(feats)

            # actualizar estado DESPUÉS de extraer las features (sin leakage)
            gh, ga = int(row.home_score), int(row.away_score)
            # expectativas y rendimiento con los Elo PRE-partido
            dr = hs.elo - as_.elo + (HOME_ELO_BONUS if home_adv else 0.0)
            exp_h = self.goal_model.expected(dr)
            exp_a = self.goal_model.expected(-dr)
            we = _expected_home(hs.elo, as_.elo, home_adv)
            w = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)

            self._update_elo(hs, as_, gh, ga, home_adv, ttype)
            is_comp = bool(row.is_competitive)
            prep_h = (not is_comp) and self._is_prep(home, date)
            prep_a = (not is_comp) and self._is_prep(away, date)
            cap = 5.0  # evita ratios absurdos contra expectativas muy bajas
            hs.record(date, gh, ga, is_comp, ttype,
                      gf_ratio=min(gh / exp_h, cap),
                      ga_ratio=min(ga / exp_a, cap), perf=w - we, is_prep=prep_h)
            as_.record(date, ga, gh, is_comp, ttype,
                       gf_ratio=min(ga / exp_a, cap),
                       ga_ratio=min(gh / exp_h, cap), perf=we - w, is_prep=prep_a)
            venue_alt = float(CITY_ALTITUDE.get(row.city, 0.0))
            for st in (hs, as_):
                st.last_date = date
                st.last_country = row.country
                st.alt_hist.append(venue_alt)
            self.last_date = date

        return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)

    def _pair_features(self, hs: _TeamState, as_: _TeamState, home_adv: bool,
                       ttype: str, cycle_start: pd.Timestamp) -> dict:
        h = hs.snapshot(cycle_start, self.prep_friendly_weight)
        a = as_.snapshot(cycle_start, self.prep_friendly_weight)
        feats = {f"home_{k}": v for k, v in h.items()}
        feats.update({f"away_{k}": v for k, v in a.items()})
        feats["elo_diff"] = hs.elo - as_.elo
        feats["elo_expected_home"] = _expected_home(hs.elo, as_.elo, home_adv)
        feats["true_home_advantage"] = home_adv
        feats["match_type"] = ttype
        return feats

    def _update_elo(self, hs: _TeamState, as_: _TeamState, gh: int, ga: int,
                    home_adv: bool, ttype: str) -> None:
        margin = gh - ga
        result = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        expected = _expected_home(hs.elo, as_.elo, home_adv)
        k = K_FACTORS.get(ttype, K_FACTORS["other"])
        delta = k * _goal_multiplier(margin) * (result - expected)
        hs.elo += delta
        as_.elo -= delta

    def match_features(self, home_team: str, away_team: str, date,
                       match_type: str, neutral: bool = True,
                       city: str | None = None, country: str | None = None,
                       knockout: bool = False) -> dict:
        """Features para un partido FUTURO (a predecir) con el estado actual.

        Para el Mundial 2026 pasa match_type='world_cup': la localía se
        resuelve sola con la regla de anfitriones. `city`/`country` activan
        las features de altitud y viaje; `knockout` marca fase eliminatoria.
        Lanza error si la fecha es anterior al último partido procesado
        (eso sería leakage).
        """
        date = pd.Timestamp(date)
        if self.last_date is not None and date < self.last_date:
            raise ValueError(
                f"Fecha {date.date()} anterior al último partido procesado "
                f"({self.last_date.date()}): produciría data leakage."
            )
        row = pd.Series({
            "tournament_type": match_type, "date": date,
            "home_team": home_team, "neutral": neutral,
        })
        home_adv = _resolve_home_advantage(row)
        hs, as_ = self._state(home_team), self._state(away_team)
        feats = self._pair_features(
            hs, as_, home_adv, match_type, self._cycle_start(date),
        )
        feats.update(self._context_features(hs, as_, date, city, country, knockout))
        feats.update({"home_team": home_team, "away_team": away_team, "date": date})
        return feats


def build_features(
    df: pd.DataFrame, warmup_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Conveniencia: devuelve (dataframe con features, builder con estado final).

    `warmup_df`: partidos anteriores al período de entrenamiento usados solo
    para converger el Elo (p. ej. 1990-2009 cuando se entrena desde 2010).
    """
    builder = FeatureBuilder()
    if warmup_df is not None:
        builder.warm_up(warmup_df)
    return builder.transform(df), builder


if __name__ == "__main__":
    from data_loader import add_basic_features, filter_since, load_results

    ELO_START, TRAIN_START = 1990, 2010
    full = add_basic_features(filter_since(load_results(), ELO_START))
    cutoff = pd.Timestamp(f"{TRAIN_START}-01-01")
    warmup = full[full["date"] < cutoff]
    train = full[full["date"] >= cutoff].reset_index(drop=True)

    builder = FeatureBuilder()
    builder.warm_up(warmup)

    print(f"Elo al {cutoff.date()} (warm-up {ELO_START}-{TRAIN_START - 1}, "
          f"{len(warmup)} partidos):")
    for team in ["Brazil", "Germany", "Spain", "San Marino"]:
        print(f"  {builder.states[team].elo:7.1f}  {team}")

    df = builder.transform(train)

    print(f"{len(df)} partidos, {df.shape[1]} columnas")
    top = sorted(builder.states.items(), key=lambda kv: kv[1].elo, reverse=True)[:10]
    print("\nTop 10 Elo actual:")
    for team, st in top:
        print(f"  {st.elo:7.1f}  {team}")

    ejemplo = builder.match_features(
        "Mexico", "Argentina", "2026-06-20", "world_cup", neutral=True
    )
    print("\nEjemplo México vs Argentina (Mundial 2026):")
    print(f"  elo_diff={ejemplo['elo_diff']:.1f}, "
          f"P(local)={ejemplo['elo_expected_home']:.3f}, "
          f"localía real={ejemplo['true_home_advantage']}, "
          f"MEX jugó eliminatorias={ejemplo['home_played_qualifiers_cycle']}, "
          f"ARG jugó eliminatorias={ejemplo['away_played_qualifiers_cycle']}")
