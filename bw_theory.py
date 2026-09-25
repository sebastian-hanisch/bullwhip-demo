"""Exakte lineare Theorie der Bestellkette (bw_theory.py): FIR-Filter je Stufe, Kettenfilter, Verstärkung als Varianzquotient, Chen-Formel für Stufe 1.
Gilt für z = 0, gamma = 1, ohne Abschneidung bei 0. Mechanisch aus messreihe_bullwhip/bull.py übernommen (Logik unverändert).
"""

from __future__ import annotations

import numpy as np


def stage_filter(m, p):
    """FIR-Koeffizienten einer Stufe:  q_t = (1 + m/p) x_t - (m/p) x_{t-p}."""
    h = np.zeros(p + 1)
    h[0] = 1.0 + m / p
    h[p] = -m / p
    return h


def chain_filters(Ls, ps, share=False):
    """Filter von D auf die Bestellung jeder Stufe (Liste von FIR-Arrays)."""
    out = []
    if not share:
        cur = np.array([1.0])
        for m, p in zip(Ls, ps):
            cur = np.convolve(cur, stage_filter(m, p))
            out.append(cur.copy())
    else:
        # q_k(t) = x_k(t) + m_k (mu_hat_t - mu_hat_{t-1}), x_k = q_{k-1}: alle Stufen mit demselben Prognosefenster p
        p = ps[0]
        assert all(pp == p for pp in ps)
        M = 0
        for m in Ls:
            M += m
            out.append(stage_filter(M, p))
    return out


def ratio_from_filter(h, rho):
    """Var(q)/Var(D) fuer AR(1)-Bedarf (rho) aus dem FIR-Filter h, exakt per Autokovarianz."""
    K = len(h)
    lags = np.arange(K)
    acov = rho ** np.abs(lags[:, None] - lags[None, :])       # Korrelationsmatrix der Bedarfe D_{t-i}
    return float(h @ acov @ h)


def chen_single_bound(m, p):
    """1 + 2 m/p + 2 m^2/p^2 (Chen et al. 2000, iid, Untergrenze mit Sicherheitsbestandsterm; exakt ohne)."""
    return 1.0 + 2.0 * m / p + 2.0 * m * m / (p * p)

