"""Dixon-Coles clásico (1997): ataque/defensa por equipo + ventaja local +
rho, por máxima verosimilitud con decaimiento temporal exponencial.

    lam = exp(c + ataque_local - defensa_visitante + gamma*localía_real)
    mu  = exp(c + ataque_visitante - defensa_local)

Se optimiza con L-BFGS-B y gradiente analítico (sin él, ~400 parámetros
serían inviables). Un ridge suave sobre ataques/defensas fija la
identificabilidad y regulariza equipos con pocos partidos.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model import score_matrix_agg


class DixonColesModel:
    def __init__(self, since: int = 2018, half_life_years: float = 3.0,
                 ridge: float = 1.0) -> None:
        self.since = since
        self.half_life_years = half_life_years
        self.ridge = ridge
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.intercept = np.log(1.3)
        self.gamma = 0.3   # ventaja local (en log-goles)
        self.rho = -0.05

    # ------------------------------------------------------------- ajuste ---
    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> "DixonColesModel":
        """`df` debe traer date, home_team, away_team, home_score, away_score
        y true_home_advantage (sale de features.py). `cutoff` excluye
        partidos posteriores (para validación temporal) y ancla el decaimiento."""
        d = df[df["date"].dt.year >= self.since]
        if cutoff is not None:
            d = d[d["date"] < cutoff]
        anchor = cutoff if cutoff is not None else d["date"].max()
        age = (anchor - d["date"]).dt.days / 365.25
        w = (0.5 ** (age / self.half_life_years)).values

        self.teams = sorted(set(d["home_team"]) | set(d["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h = d["home_team"].map(idx).values
        a = d["away_team"].map(idx).values
        x = d["home_score"].values.astype(float)
        y = d["away_score"].values.astype(float)
        adv = d["true_home_advantage"].values.astype(float)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def nll_grad(p: np.ndarray) -> tuple[float, np.ndarray]:
            att, dfn = p[:n], p[n:2 * n]
            c, g, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            lam = np.exp(c + att[h] - dfn[a] + g * adv)
            mu = np.exp(c + att[a] - dfn[h])

            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.maximum(tau, 1e-6)

            ll = w * (np.log(tau) + x * np.log(lam) - lam + y * np.log(mu) - mu)
            nll = -ll.sum() + self.ridge * (att @ att + dfn @ dfn)

            # d(log tau)/d(log lam), /d(log mu), d(log tau)/d(rho)
            tlam = np.zeros_like(lam)
            tmu = np.zeros_like(lam)
            trho = np.zeros_like(lam)
            tlam[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
            tmu[m00] = tlam[m00]
            trho[m00] = -lam[m00] * mu[m00] / tau[m00]
            tlam[m01] = lam[m01] * rho / tau[m01]
            trho[m01] = lam[m01] / tau[m01]
            tmu[m10] = mu[m10] * rho / tau[m10]
            trho[m10] = mu[m10] / tau[m10]
            trho[m11] = -1.0 / tau[m11]

            dllam = w * (x - lam + tlam)   # dLL/d(log lam) por partido
            dlmu = w * (y - mu + tmu)

            g_att = np.zeros(n)
            g_def = np.zeros(n)
            np.add.at(g_att, h, dllam)
            np.add.at(g_att, a, dlmu)
            np.add.at(g_def, a, -dllam)
            np.add.at(g_def, h, -dlmu)
            grad = np.concatenate([
                -g_att + 2 * self.ridge * att,
                -g_def + 2 * self.ridge * dfn,
                [-(dllam.sum() + dlmu.sum()),
                 -(dllam * adv).sum(),
                 -(w * trho).sum()],
            ])
            return float(nll), grad

        p0 = np.concatenate([np.zeros(2 * n), [self.intercept, self.gamma, self.rho]])
        bounds = ([(-3.0, 3.0)] * (2 * n)
                  + [(-2.0, 2.0), (-1.0, 2.0), (-0.25, 0.25)])
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500})
        p = res.x
        self.attack = dict(zip(self.teams, p[:n]))
        self.defense = dict(zip(self.teams, p[n:2 * n]))
        self.intercept = float(p[2 * n])
        self.gamma = float(p[2 * n + 1])
        self.rho = float(p[2 * n + 2])
        self.converged = bool(res.success)
        self.n_matches = len(d)
        return self

    # ---------------------------------------------------------- predicción ---
    def lambdas(self, home: str, away: str, home_advantage: bool) -> tuple[float, float]:
        """Equipo no visto en el ajuste => ataque/defensa 0 (equipo promedio)."""
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        lam = float(np.exp(self.intercept + ah - da + self.gamma * home_advantage))
        mu = float(np.exp(self.intercept + aa - dh))
        return lam, mu

    def score_matrix(self, home: str, away: str,
                     home_advantage: bool) -> np.ndarray:
        lam, mu = self.lambdas(home, away, home_advantage)
        return score_matrix_agg(lam, mu, rho=self.rho)

    # ------------------------------------------------------- serialización ---
    def to_dict(self) -> dict:
        return {
            "since": self.since, "half_life_years": self.half_life_years,
            "ridge": self.ridge, "intercept": self.intercept,
            "gamma": self.gamma, "rho": self.rho,
            "attack": self.attack, "defense": self.defense,
            "n_matches": getattr(self, "n_matches", None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColesModel":
        dc = cls(since=d["since"], half_life_years=d["half_life_years"],
                 ridge=d["ridge"])
        dc.intercept, dc.gamma, dc.rho = d["intercept"], d["gamma"], d["rho"]
        dc.attack, dc.defense = d["attack"], d["defense"]
        dc.teams = sorted(d["attack"])
        return dc
