"""Die 47 Korrektheits-Checks der Messreihe (messreihe_bullwhip/check.py) als Funktionen für die Tests und das Bau-Gate (tools/check_full.py), mit den bw_-Modulen statt bull.py.

Jede Funktion prüft EINEN Check und liefert (bestanden, Text). Die Checks der Messreihe laufen in voller Größe (etwa 9 s): Theorie == Simulation, Handinstanz, Erhaltungssätze,
Einstufig gegen Newsvendor-Analytik, Zweig-Tests, Kalibrierung, Wiederholbarkeit. Die statistischen Checks (Simulation == Theorie auf 4 Standardfehler + 1 %, Kalibrierung, Analytik)
hängen an KEINEM einzelnen Seed: die Umgebungsvariable BW_CHECK_SEED_SHIFT verschiebt alle Seeds (Robustheitsprüfung in tools/check_full.py und im Test).
Die exakten Identitäten (Faltung, Handinstanz, Bestandspositions-Identität, Warenerhaltung) sind unabhängig vom Zufallsstrom."""
import functools
import math
import os

import numpy as np

import bw_model as bull
import bw_theory as theory

SHIFT = int(os.environ.get("BW_CHECK_SEED_SHIFT", "0"))


def seed(s):
    return s + SHIFT


# ----- Newsvendor-Analytik (aus inv_bridge.py der Messreihe: nur für die Einstufen-Querprüfung) -------------------------------------
def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _G(z):
    return _pdf(z) - z * (1.0 - _cdf(z))


def normal_alpha_beta(mean, cv, m, z):
    """alpha und beta einer Order-up-to-Politik S = m*mean + z*sigma*sqrt(m) bei iid Normalbedarf, Schutzintervall m Perioden (kontinuierliche Normal-Analytik, unabhängig vom Modell)."""
    sd = cv * mean
    S = m * mean + z * sd * math.sqrt(m)
    alpha = _cdf(z)
    short = sd * math.sqrt(m) * _G(z)
    if m >= 2:
        s2 = sd * math.sqrt(m - 1)
        short -= s2 * _G((S - (m - 1) * mean) / s2)
    return alpha, 1.0 - short / mean


# ----- 1. Lineare Theorie == Simulation ----------------------------------------------------------------------------------------------
LINEAR_CASES = [(rho, Ls, p, share) for rho in (0.0, 0.5, 0.8) for Ls, p in (([2, 2, 2, 2], 5), ([1, 3, 2, 4], 6), ([4, 4, 4, 4], 10)) for share in (False, True)]


@functools.lru_cache(maxsize=8)
def _linear_demand(rho):
    rng = np.random.default_rng(seed(10 + int(rho * 10)))
    return bull.gen_ar1(rng, 400, 2000, 1000.0, 3.0, rho)


def check1(rho, Ls, p, share):
    D = _linear_demand(rho)
    q, _, _ = bull.order_streams(D, Ls, p, z=0.0, share=share, nonneg=False)
    vr = bull.var_ratio(q, D, 100)
    th = [theory.ratio_from_filter(f, rho) for f in theory.chain_filters(Ls, [p] * 4, share)]
    good = all(abs(a - t_) < 4 * s + 0.01 * t_ for (a, s), t_ in zip(vr, th))
    return good, " ".join(f"{a:.3f}/{t_:.3f}" for (a, s), t_ in zip(vr, th))


# ----- 2. Welche Formel passt -----------------------------------------------------------------------------------------------------
def check2a():
    for m in (1, 2, 3, 4, 6):
        for p in (3, 5, 10, 20):
            th = theory.ratio_from_filter(theory.chain_filters([m], [p])[0], 0.0)
            if abs(th - theory.chen_single_bound(m, p)) >= 1e-12:
                return False, f"m {m}, p {p}"
    return True, "20 Kombinationen"


def check2b():
    rng = np.random.default_rng(seed(22))
    D = bull.gen_ar1(rng, 600, 2000, 1000.0, 3.0, 0.0)
    L, p = 3, 6
    q, _, _ = bull.order_streams(D, [L], p, z=0.0, nonneg=False)
    (a, s), = bull.var_ratio(q, D, 100)
    f_L, f_L1 = theory.chen_single_bound(L, p), theory.chen_single_bound(L + 1, p)
    return abs(a - f_L) < 4 * s and abs(a - f_L1) > 20 * s, f"sim {a:.4f}+-{s:.4f}, Formel(L) {f_L:.4f}, Formel(L+1) {f_L1:.4f}"


def check2c():
    worst, all_ge = 0.0, True
    for m in (1, 2, 4):
        for p in (3, 5, 10):
            hs = theory.chain_filters([m] * 4, [p] * 4)
            th = [theory.ratio_from_filter(h, 0.0) for h in hs]
            prod = [theory.chen_single_bound(m, p) ** (k + 1) for k in range(4)]
            all_ge &= all(t_ >= pr - 1e-12 for t_, pr in zip(th, prod)) and abs(th[0] - prod[0]) < 1e-12 and th[1] > prod[1]
            a_, b_ = 1 + m / p, m / p
            worst = max(worst, abs(th[1] - (a_ ** 4 + 4 * a_ ** 2 * b_ ** 2 + b_ ** 4)))
    return all_ge and worst < 1e-10, f"max Abw. Handformel {worst:.2e}"


def check2d():
    good = True
    for Ls in ([2, 2, 2, 2], [1, 3, 2, 4]):
        for k, h in enumerate(theory.chain_filters(Ls, [5] * 4, share=True)):
            good &= abs(theory.ratio_from_filter(h, 0.0) - theory.chen_single_bound(sum(Ls[:k + 1]), 5)) < 1e-12
    return bool(good), ""


# ----- 3. Unabhängige Filteranwendung ------------------------------------------------------------------------------------------------
def check3(share):
    R, T = 50, 500
    rng = np.random.default_rng(seed(33))
    D = bull.gen_ar1(rng, R, T, 500.0, 40.0, 0.4)
    Ls, p = [2, 3, 1], 4
    q, _, _ = bull.order_streams(D, Ls, p, z=0.0, share=share, nonneg=False)
    worst = 0.0
    for k, h in enumerate(theory.chain_filters(Ls, [p] * 3, share=share)):
        y = np.array([np.convolve(D[r], h)[:T] for r in range(R)])
        worst = max(worst, float(np.abs(y[:, 40:] - q[k][:, 40:]).max()))
    return worst < 1e-8, f"max |Delta| = {worst:.2e}"


# ----- 4. Handinstanz ----------------------------------------------------------------------------------------------------------------
def check4a():
    T = 30
    mu = 100.0
    D = np.full((1, T), mu)
    D[0, 15] += 10.0
    q, _, _ = bull.order_streams(D, [1], 2, z=0.0, nonneg=True, mu0=mu)
    exp = {15: mu + 15.0, 16: mu, 17: mu - 5.0}
    good = all(abs(q[0, 0, t] - v) < 1e-9 for t, v in exp.items()) and np.allclose(np.delete(q[0, 0], [15, 16, 17]), mu)
    return bool(good), str([round(float(q[0, 0, t]), 3) for t in (14, 15, 16, 17, 18)])


def check4b():
    T = 30
    D2 = np.full((1, T), 3.0)
    D2[0, 15] += 30.0
    qn, _, _ = bull.order_streams(D2, [1], 2, z=0.0, nonneg=True, mu0=3.0)
    qf, _, _ = bull.order_streams(D2, [1], 2, z=0.0, nonneg=False, mu0=3.0)
    return bool(abs(qf[0, 0, 17] + 12.0) < 1e-9 and qn[0, 0, 17] == 0.0), f"ohne {qf[0, 0, 17]:.2f}, mit {qn[0, 0, 17]:.2f}"


def check4c():
    Dc = np.full((3, 100), 50.0)
    q0, _, _ = bull.order_streams(Dc, [2, 2, 2], 5, z=1.645)
    ph = bull.physical(Dc, q0, [2, 2, 2], burn=20)
    return bool(np.allclose(q0, 50.0) and np.allclose(ph["beta"], 1.0) and float(ph["backlog"].max()) < 1e-9), ""


# ----- 5. Bestandspositions-Identität und Warenerhaltung -----------------------------------------------------------------------------
IDENTITY_CASES = [({}, 1.645), ({"gamma": 0.5}, 1.645), ({"share": True}, 1.645), ({"share": True, "gamma": 0.4}, 1.645), ({}, 0.0)]


def _identity_run(kw, z):
    rng = np.random.default_rng(seed(55))
    D = bull.gen_ar1(rng, 60, 500, 100.0, 45.0, 0.3)            # cv 0.45: Abschneidung bindet
    Ls = [2, 3, 1, 2]
    q, S, IPb = bull.order_streams(D, Ls, 6, z=z, **kw)
    ipa = bull.ip_after_order(D, q, Ls)
    ph = bull.physical(D, q, Ls, burn=50, check_ip=ipa)
    return D, Ls, q, ph


def check5(kw, z):
    """Bestandspositions-Identität (physisch gezählt == rekursiv geführt) und Warenerhaltung, mit bindender Abschneidung."""
    D, Ls, q, ph = _identity_run(kw, z)
    share_zero = float((q == 0).mean())
    goods = ph["goods_init"] + q[len(Ls) - 1].sum(axis=1) - ph["goods_end"] - ph["cust_shipped"]
    good = ph["ip_max_dev"] < 1e-7 and float(np.abs(goods).max()) < 1e-6 and share_zero > 0.0
    return bool(good), f"Position {ph['ip_max_dev']:.1e}, Ware {float(np.abs(goods).max()):.1e}, Abschneidung in {100 * share_zero:.1f} % der Bestellungen"


def check5b(kw, z):
    """Kundenseite: Summe Bedarf == geliefert + Rückstand (Stufe 1)."""
    D, Ls, q, ph = _identity_run(kw, z)
    cons = D.sum(axis=1) - ph["cust_shipped"] - ph["backlog1_end"]
    return float(np.abs(cons).max()) < 1e-6, f"Abw. {float(np.abs(cons).max()):.1e}"


# ----- 6. Einstufig gegen Newsvendor-Analytik ---------------------------------------------------------------------------------------
def check6(cv, L):
    R, T, burn = 800, 1500, 200
    rng = np.random.default_rng(seed(66 + (0 if cv == 0.2 else 1)))
    D = np.maximum(np.rint(bull.gen_ar1(rng, R, T, 100.0, 100.0 * cv, 0.0)), 0.0)
    q, S, _ = bull.order_streams(D, [L], 300, z=1.645)
    ph = bull.physical(D, q, [L], burn=burn)
    a, b = float(ph["alpha"][0].mean()), float(ph["beta"][0].mean())
    ea, eb = normal_alpha_beta(100.0, cv, L, 1.645)
    se_a = float(ph["alpha"][0].std(ddof=1) / math.sqrt(R))
    return bool(abs(a - ea) < 4 * se_a + 0.006 and abs(b - eb) < 0.004), f"alpha {a:.4f} vs {ea:.4f}, beta {b:.4f} vs {eb:.4f}"


# ----- 7. Zweig-Tests ----------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _switch_demand(rho):
    rng = np.random.default_rng(seed(77) + (0 if rho == 0.0 else 1))
    return bull.gen_ar1(rng, 150, 900, 100.0, 25.0, rho)


def _amp(D, Ls=(2, 2, 2, 2), **kw):
    z = kw.pop("z", 1.645)
    L = list(kw.pop("Ls", Ls))
    q, _, _ = bull.order_streams(D, L, kw.pop("p", 8), z=z, **kw)
    return [a for a, _ in bull.var_ratio(q, D, 100)][-1]


def check7a():
    D = _switch_demand(0.0)
    base = _amp(D)
    variants = {"gamma=0.5": _amp(D, gamma=0.5), "geteilt": _amp(D, share=True), "z=0": _amp(D, z=0.0), "nonneg aus": _amp(D, nonneg=False), "p=16": _amp(D, p=16),
                "L=1": _amp(D, Ls=[1, 1, 1, 1]), "z=3": _amp(D, z=3.0)}
    return all(abs(v / base - 1) > 0.01 for v in variants.values()), f"Basis {base:.2f}; " + ", ".join(f"{k}: {v:.2f}" for k, v in variants.items())


def check7b():
    base = _amp(_switch_demand(0.0))
    return abs(_amp(_switch_demand(0.6)) / base - 1) > 0.05, ""


def check7c():
    D = _switch_demand(0.0)
    Ls = [2, 2, 2, 2]
    q1, _, _ = bull.order_streams(D, Ls, 8, z=1.645, gammas=[1, 1, 1, 0.5])
    q0, _, _ = bull.order_streams(D, Ls, 8, z=1.645)
    return bool(np.array_equal(q1[:3], q0[:3]) and not np.array_equal(q1[3], q0[3])), ""


def check7d():
    D = _switch_demand(0.0)
    Ls = [2, 2, 2, 2]
    q0, _, _ = bull.order_streams(D, Ls, 8, z=1.645)
    qs, _, _ = bull.order_streams(D, Ls, 8, z=1.645, share=True)
    return bool(np.array_equal(qs[0], q0[0]) and not np.array_equal(qs[1], q0[1])), ""


# ----- 8. Kalibrierung ---------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _calibration():
    R, T, burn = 120, 900, 100
    D = bull.gen_ar1(np.random.default_rng(seed(88)), R, T, 100.0, 20.0, 0.0)
    D2 = bull.gen_ar1(np.random.default_rng(seed(89)), 300, 1100, 100.0, 20.0, 0.0)
    Ls = [2, 2, 2]
    zs, q, ph = bull.calibrate_z(D, Ls, 10, target=0.95, burn=burn, passes=2, iters=9)
    return D2, Ls, burn, zs, ph


def check8a():
    ph = _calibration()[4]
    return bool(np.all(np.abs(ph["beta"].mean(axis=1) - 0.95) < 0.005)), str(np.round(ph["beta"].mean(axis=1), 4))


def check8b():
    D2, Ls, burn, zs, _ = _calibration()
    q2, _, _ = bull.order_streams(D2, Ls, 10, z=zs)
    ph2 = bull.physical(D2, q2, Ls, burn=burn)
    return bool(np.all(np.abs(ph2["beta"].mean(axis=1) - 0.95) < 0.015)), str(np.round(ph2["beta"].mean(axis=1), 4))


def check8d():
    D2, Ls, burn, _, _ = _calibration()
    b_lo = bull.physical(D2, bull.order_streams(D2, Ls, 10, z=[0.5, 1.6, 1.6])[0], Ls, burn=burn)["beta"][0].mean()
    b_hi = bull.physical(D2, bull.order_streams(D2, Ls, 10, z=[2.5, 1.6, 1.6])[0], Ls, burn=burn)["beta"][0].mean()
    return bool(b_hi > b_lo + 0.01), f"{b_hi:.4f} > {b_lo:.4f}"


# ----- 9. Wiederholbarkeit -----------------------------------------------------------------------------------------------------------
def check9():
    def one():
        rng = np.random.default_rng(seed(999))
        D = bull.gen_ar1(rng, 40, 400, 100.0, 20.0, 0.3)
        q, _, _ = bull.order_streams(D, [2, 2, 2], 8, z=1.645, gamma=0.7)
        ph = bull.physical(D, q, [2, 2, 2], burn=50)
        zs, q2, ph2 = bull.calibrate_z(D, [2, 2], 8, target=0.95, burn=50, passes=1, iters=5)
        return float(q.sum()), float(ph["cost"].sum()), tuple(zs)
    return one() == one(), ""


# Reihenfolge und Zählung wie in check.py: 18 + 4 + 2 + 3 + 5 + 5 + 2 + 4 + 3 + 1 = 47
ALL_CHECKS = ([(f"1 rho={r} L={Ls} p={p} {'geteilt' if sh else 'lokal'}", check1, (r, Ls, p, sh)) for r, Ls, p, sh in LINEAR_CASES]
              + [("2a", check2a, ()), ("2b", check2b, ()), ("2c", check2c, ()), ("2d", check2d, ())]
              + [("3 lokal", check3, (False,)), ("3b geteilt", check3, (True,))]
              + [("4a", check4a, ()), ("4b", check4b, ()), ("4c", check4c, ())]
              + [(f"5 {kw} z={z}", check5, (kw, z)) for kw, z in IDENTITY_CASES]
              + [(f"5b {kw} z={z}", check5b, (kw, z)) for kw, z in IDENTITY_CASES]
              + [("6 cv=0.2 L=2", check6, (0.2, 2)), ("6 cv=0.3 L=1", check6, (0.3, 1))]
              + [("7a", check7a, ()), ("7b", check7b, ()), ("7c", check7c, ()), ("7d", check7d, ())]
              + [("8a", check8a, ()), ("8b", check8b, ()), ("8d", check8d, ())]
              + [("9", check9, ())])
