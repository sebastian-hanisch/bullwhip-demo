"""Unabhängige Orakel für den Bullwhip-Kern (Regressionstest der Orakelprüfung).

1. Skalare Referenzsimulation, aus der Modellbeschreibung neu gebaut: Lieferungen als explizite Liste (Ankunftsperiode, Menge), Fenster mit numpy (Mittel, Standardabweichung ddof=1) und die Bestellung
   aus der PHYSISCHEN Bestandsposition (Lager + Unterwegs + beim Lieferanten offen - eigener Rückstand) statt aus der Rekursion IP - x + q des Modells. Bestellungen, Servicegrad, Lager, Rückstand und
   Kosten des Modells (`order_streams` + `physical`) müssen übereinstimmen: das belegt zugleich die Bestandspositions-Identität.
   Der Zähler alpha (Perioden mit Rückstand <= 1e-9) bleibt außen vor: die laufenden Summen des Fensters haben bei konstantem Fenster (Start) ein Rauschen von etwa 1e-5 in den Bestellungen.
2. Verstärkung: Stufe-1-Formel 1 + 2m/p + 2m²/p², Stufe 4 aus einem Polynomprodukt der Filter, und die AR(1)-Verstärkung der Theorie über das SPEKTRUM (Integral über |H|² mal Spektraldichte)
   statt über die Autokovarianzmatrix; `var_ratio` gegen eine Schleife."""

import math

import numpy as np
import pytest

import bw_model as M
import bw_theory as TH


def _ref_sim(drow, Ls, ps, zs, gammas, share, mu0, burn):
    N, T = len(Ls), len(drow)
    onhand, backlog = [0.0] * N, [0.0] * N
    arrivals = [dict() for _ in range(N)]
    for k in range(N):
        for j in range(Ls[k]):
            arrivals[k][j] = arrivals[k].get(j, 0.0) + mu0                       # stationärer Start: je Ankunftsperiode mu0 unterwegs
    hist = [[mu0] * ps[k] for k in range(N)]
    q = np.zeros((N, T))
    cost, ok, short, dem, oh, bl = (np.zeros(N) for _ in range(6))
    nobs = 0
    for t in range(T):
        for k in range(N):
            onhand[k] += arrivals[k].pop(t, 0.0)
        for k in range(N):
            x = drow[t] if k == 0 else q[k - 1, t]
            owed = backlog[k]
            ship = min(onhand[k], owed + x)
            fresh = min(x, max(onhand[k] - owed, 0.0))
            onhand[k] -= ship
            backlog[k] = owed + x - ship
            if k > 0:
                arrivals[k - 1][t + Ls[k - 1]] = arrivals[k - 1].get(t + Ls[k - 1], 0.0) + ship
            if t >= burn:
                short[k] += x - fresh
                dem[k] += x
            hist[k] = hist[k][1:] + [drow[t] if share else x]
            w = np.array(hist[k])
            S = Ls[k] * w.mean() + zs[k] * w.std(ddof=1) * math.sqrt(Ls[k])
            ip = onhand[k] + sum(arrivals[k].values()) + (backlog[k + 1] if k + 1 < N else 0.0) - backlog[k]
            gap = S - ip
            qq = max(gap if gammas[k] == 1.0 else w.mean() + gammas[k] * (gap - w.mean()), 0.0)
            q[k, t] = qq
            if k == N - 1:
                arrivals[k][t + Ls[k]] = arrivals[k].get(t + Ls[k], 0.0) + qq     # unbegrenzter Lieferant
        if t >= burn:
            for k in range(N):
                cost[k] += onhand[k] + 9.0 * backlog[k]
                oh[k] += onhand[k]
                bl[k] += backlog[k]
            nobs += 1
    return q, {"cost": cost / nobs, "beta": 1 - short / dem, "onhand": oh / nobs, "backlog": bl / nobs}


def test_orders_and_physical_chain_match_an_independent_scalar_simulation():
    rng = np.random.default_rng(2)
    for it in range(60):
        N, R, T = int(rng.integers(1, 5)), int(rng.integers(1, 3)), int(rng.integers(8, 50))
        Ls = [int(rng.integers(1, 5)) for _ in range(N)]
        p = int(rng.choice([2, 3, 6, 10]))
        share = it % 4 == 0
        ps = [p] * N if (it % 3 or share) else [int(rng.choice([2, 3, 5, 8])) for _ in range(N)]
        zs = [float(rng.choice([0.0, 0.5, 1.645, 3.0])) for _ in range(N)]
        gam = [float(rng.choice([1.0, 0.7, 0.5, 0.3])) for _ in range(N)]
        D = M.gen_ar1(np.random.default_rng(int(rng.integers(0, 10 ** 6))), R, T, 100.0, float(rng.choice([5, 20, 40])), float(rng.choice([0.0, 0.5, 0.8])))
        burn = int(rng.choice([0, 3]))
        q, _, _ = M.order_streams(D, Ls, p, z=zs, share=share, ps=ps, gammas=gam)
        ph = M.physical(D, q, Ls, 1.0, 9.0, burn=burn)
        for r in range(R):
            qr, pr = _ref_sim(D[r], Ls, ps, zs, gam, share, float(D[:, 0].mean()), burn)
            assert np.abs(qr - q[:, r, :]).max() < 1e-3
            for key in ("cost", "beta", "onhand", "backlog"):
                assert pr[key] == pytest.approx(ph[key][:, r], rel=1e-6, abs=1e-6), key


def test_one_stage_amplification_formula_and_the_four_stage_polynomial_product():
    for m in (1, 2, 3, 4):
        for p in (2, 3, 5, 10, 20):
            assert TH.ratio_from_filter(TH.stage_filter(m, p), 0.0) == pytest.approx(1 + 2 * m / p + 2 * m * m / (p * p), abs=1e-12)
            poly = np.array([1.0])
            for k in range(4):
                f = np.zeros(p + 1)
                f[0], f[p] = 1 + m / p, -m / p
                poly = np.convolve(poly, f)                                            # Faltung der Stufenfilter: iid-Bedarf, Verstärkung = Summe der Quadrate
            assert TH.ratio_from_filter(TH.chain_filters([m] * 4, [p] * 4)[3], 0.0) == pytest.approx(float((poly ** 2).sum()), rel=1e-12)
            assert TH.chen_single_bound(m, p) ** 4 <= (poly ** 2).sum() + 1e-9               # die Produktformel ist nur eine Untergrenze


def test_ar1_amplification_matches_the_spectral_integral():
    w = np.linspace(-np.pi, np.pi, 2001)
    for Ls, p, share, rho in [([2, 2, 2], 10, False, 0.5), ([1, 3], 4, False, 0.8), ([2, 2, 2, 2], 5, True, 0.0), ([3, 1, 2], 6, True, -0.3)]:
        for k, h in enumerate(TH.chain_filters(Ls, [p] * len(Ls), share=share)):
            H = np.array([np.sum(h * np.exp(-1j * x * np.arange(len(h)))) for x in w])
            spec = (1 - rho ** 2) / (1 - 2 * rho * np.cos(w) + rho ** 2)               # Spektraldichte des AR(1) mit Varianz 1
            f = np.abs(H) ** 2 * spec
            val = float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(w))) / (2 * np.pi)       # Trapezregel
            assert val == pytest.approx(TH.ratio_from_filter(h, rho), rel=2e-3)


def test_var_ratio_matches_a_loop():
    rng = np.random.default_rng(4)
    for _ in range(40):
        R, T, N, burn = int(rng.integers(2, 6)), int(rng.integers(8, 40)), int(rng.integers(1, 4)), int(rng.integers(0, 5))
        D, q = rng.normal(100, 10, (R, T)), rng.normal(100, 30, (N, R, T))
        for k, (mean, se) in enumerate(M.var_ratio(q, D, burn)):
            r = np.array([np.var(q[k, i, burn:], ddof=1) / np.var(D[i, burn:], ddof=1) for i in range(R)])
            assert mean == pytest.approx(r.mean(), abs=1e-9) and se == pytest.approx(r.std(ddof=1) / math.sqrt(R), abs=1e-9)
