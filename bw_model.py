"""Bullwhip-Modell (bw_model.py, Kern der Demo): mehrstufige Kette (Einzelhandel = Stufe 1, ..., Fabrik = Stufe N), Order-up-to mit gleitendem Durchschnitt.

Nur numpy + Standardbibliothek.

Zeitablauf je Periode t (alles vektorisiert ueber R Replikationen):
  1. Lieferungen, die zu Beginn von t eintreffen, gehen ins Lager der jeweiligen Stufe.
  2. Stufe 1 sieht den Endkundenbedarf D_t, Stufe k+1 sieht die Bestellung q_k(t) der Stufe darunter (gleiche Periode, keine
     Informationsverzoegerung; die Bestellmengen haengen dann nicht von Lieferungen ab).
  3. Jede Stufe k liefert aus dem Lager (zuerst offene Rueckstaende, dann neuer Bedarf; Rest wird Rueckstand = Backorder) an die Stufe
     darunter (Ankunft dort nach L_{k-1} Perioden) bzw. an den Kunden.
  4. Jede Stufe bestellt am Ende der Periode nach der Order-up-to-Politik (Rueckstand zaehlt zur Bestandsposition):
        S_k(t)  = m_k * mu_hat_k(t) + z * sigma_hat_k(t) * sqrt(m_k)         (m_k = L_k = Schutzintervall)
        q_k(t)  = max(0, S_k(t) - IP_k(t))                                   (gamma = 1: reine Order-up-to)
        q_k(t)  = max(0, mu_hat + gamma*(S_k - IP_k - mu_hat))               (gamma < 1: Glaettung = nur Anteil gamma der Luecke)
     mu_hat, sigma_hat = Mittel und Standardabweichung (ddof=1) der letzten p Werte der Prognose-Eingangsreihe INKLUSIVE des
     aktuellen Wertes.  Lokal: Eingangsreihe der Stufe k = ihr eigener Bedarf x_k (Bestellungen der Stufe darunter).
     Bedarfsinformation teilen (share=True): alle Stufen prognostizieren aus dem Endkundenbedarf D_t.
  5. Die Bestellung der Stufe N kommt von einem unbegrenzten Lieferanten nach L_N Perioden.

Bestellungen haengen nur von Bedarfen, Fenstern und der rekursiv gefuehrten Bestandsposition ab, nicht von den physischen Lieferungen
(Bestandsposition = Lager + Unterwegs + beim Lieferanten offene Bestellung - eigener Rueckstand; das prueft check.py).

Lead-time-Konvention: L_k = Wiederbeschaffungszeit in Perioden (>= 1) zwischen Bestellung (Periodenende) und Eintreffen (Beginn von t+L).
Das Schutzintervall der Order-up-to-Stufe ist genau m_k = L_k Bedarfsperioden.  Chen/Drezner/Ryan/Simchi-Levi (2000) nennen diese Groesse "L"
(Ergebnis in check.py: Formel passt exakt mit L = Schutzintervall).
"""

# Mechanisch aus messreihe_bullwhip/bull.py übernommen (Logik unverändert; Aufteilung: Modell und Kalibrierung hier, exakte lineare Theorie in bw_theory.py).
# Bitgleichheit belegt in tests/test_frozen_reference.py (eingefrorene Nachfragereihen) und tools/check_full.py (Kalibrierungsjobs gegen sweep_data.json).

from __future__ import annotations

import math

import numpy as np


def gen_ar1(rng, R, T, mu, sigma, rho):
    """AR(1) mit Randmittel mu, Randstandardabweichung sigma, Autokorrelation rho (stationaer gestartet)."""
    eps = rng.standard_normal((R, T)) * (sigma * math.sqrt(1.0 - rho * rho))
    D = np.empty((R, T))
    D[:, 0] = mu + sigma * rng.standard_normal(R)
    for t in range(1, T):
        D[:, t] = mu + rho * (D[:, t - 1] - mu) + eps[:, t]
    return D


class Window:
    """Gleitendes Fenster (letzte p Werte) mit laufender Summe/Quadratsumme, vektorisiert ueber R."""

    def __init__(self, R, p, init):
        self.p = p
        self.buf = np.full((R, p), float(init))
        self.i = 0
        self.s1 = self.buf.sum(axis=1)
        self.s2 = (self.buf ** 2).sum(axis=1)

    def push(self, x):
        old = self.buf[:, self.i].copy()
        self.buf[:, self.i] = x
        self.i = (self.i + 1) % self.p
        self.s1 += x - old
        self.s2 += x * x - old * old

    def mean(self):
        return self.s1 / self.p

    def sd(self):
        # numerisch stabil genug fuer float64 bei Werten um 100
        v = (self.s2 - self.s1 * self.s1 / self.p) / (self.p - 1)
        return np.sqrt(np.maximum(v, 0.0))


def order_streams(D, Ls, p, z=0.0, gamma=1.0, share=False, nonneg=True, mu0=None, ps=None, gammas=None):
    """Bestellungen aller Stufen.  D: (R, T).  Rueckgabe q: (N, R, T) sowie S: (N, R, T) und IP-Verlauf (N, R, T) (vor Bestellung)."""
    R, T = D.shape
    N = len(Ls)
    ps = ps if ps is not None else [p] * N
    zs = list(z) if hasattr(z, '__len__') else [z] * N
    gammas = gammas if gammas is not None else [gamma] * N
    mu0 = float(D[:, 0].mean()) if mu0 is None else mu0
    wins = [Window(R, ps[k], mu0) for k in range(N)]
    IP = [np.full(R, Ls[k] * mu0) for k in range(N)]
    q = np.zeros((N, R, T))
    S_out = np.zeros((N, R, T))
    IPb_out = np.zeros((N, R, T))
    for t in range(T):
        x = D[:, t]
        for k in range(N):
            if k > 0:
                x = q[k - 1, :, t]
            f = D[:, t] if share else x
            w = wins[k]
            w.push(f)
            mu = w.mean()
            m = Ls[k]
            S = m * mu + zs[k] * w.sd() * math.sqrt(m)
            IPb = IP[k] - x
            gap = S - IPb
            g = gammas[k]
            qq = gap if g == 1.0 else mu + g * (gap - mu)
            if nonneg:
                qq = np.maximum(qq, 0.0)
            q[k, :, t] = qq
            S_out[k, :, t] = S
            IPb_out[k, :, t] = IPb
            IP[k] = IPb + qq
    return q, S_out, IPb_out


def physical(D, q, Ls, h=1.0, pc=9.0, burn=0, check_ip=None):
    """Physische Lager-/Rueckstandsfuehrung aus den Bestellstroemen.  Rueckgabe je Stufe: Kosten, beta, alpha, mittleres Lager/Rueckstand."""
    N, R, T = q.shape
    Lmax = max(Ls)
    slots = Lmax + 1
    mu0 = float(D[:, 0].mean())
    onhand = np.zeros((N, R))
    backlog = np.zeros((N, R))
    pipe = np.zeros((N, R, slots))
    for k in range(N):                                      # stationaerer Start: je Ankunftsperiode mu0 unterwegs
        for j in range(Ls[k]):
            pipe[k, :, j % slots] = mu0
    cost = np.zeros((N, R)); short = np.zeros((N, R)); dem = np.zeros((N, R)); ok = np.zeros((N, R))
    oh_sum = np.zeros((N, R)); bl_sum = np.zeros((N, R)); n_obs = 0
    ip_max_dev = 0.0
    cust = np.zeros(R)
    init_goods = onhand.sum(axis=0) + pipe.sum(axis=(0, 2))
    for t in range(T):
        for k in range(N):
            onhand[k] += pipe[k, :, t % slots]
            pipe[k, :, t % slots] = 0.0
        for k in range(N):
            x = D[:, t] if k == 0 else q[k - 1, :, t]
            avail = onhand[k]
            owed = backlog[k]
            required = owed + x
            ship = np.minimum(avail, required)
            ontime = np.minimum(x, np.maximum(avail - owed, 0.0))
            onhand[k] = avail - ship
            backlog[k] = required - ship
            if k > 0:
                pipe[k - 1, :, (t + Ls[k - 1]) % slots] += ship
            else:
                cust += ship
            if t >= burn:
                short[k] += x - ontime
                dem[k] += x
        # Bestellung der letzten Stufe kommt vom unbegrenzten Lieferanten
        pipe[N - 1, :, (t + Ls[N - 1]) % slots] += q[N - 1, :, t]
        if t >= burn:
            for k in range(N):
                cost[k] += h * onhand[k] + pc * backlog[k]
                ok[k] += (backlog[k] <= 1e-9)
                oh_sum[k] += onhand[k]; bl_sum[k] += backlog[k]
            n_obs += 1
        if check_ip is not None:                              # Bestandspositions-Identitaet
            for k in range(N):
                ip_phys = onhand[k] + pipe[k].sum(axis=1) + (backlog[k + 1] if k + 1 < N else 0.0) - backlog[k]
                ip_phys_dev = np.abs(ip_phys - check_ip[k][:, t]).max()
                ip_max_dev = max(ip_max_dev, float(ip_phys_dev))
    out = {"cost": cost / n_obs, "alpha": ok / n_obs, "beta": 1.0 - short / dem, "onhand": oh_sum / n_obs, "backlog": bl_sum / n_obs}
    out["cust_shipped"] = cust
    out["goods_end"] = onhand.sum(axis=0) + pipe.sum(axis=(0, 2))
    out["goods_init"] = init_goods
    out["backlog1_end"] = backlog[0]
    if check_ip is not None:
        out["ip_max_dev"] = ip_max_dev
    return out


def ip_after_order(D, q, Ls, mu0=None):
    """Rekonstruiert IP nach Bestellung aus q und x (fuer den Identitaetstest): IP_k(t) = IP_k(t-1) - x_k(t) + q_k(t)."""
    N, R, T = q.shape
    mu0 = float(D[:, 0].mean()) if mu0 is None else mu0
    out = np.zeros((N, R, T))
    for k in range(N):
        ip = np.full(R, Ls[k] * mu0)
        for t in range(T):
            x = D[:, t] if k == 0 else q[k - 1, :, t]
            ip = ip - x + q[k, :, t]
            out[k, :, t] = ip
    return out


def var_ratio(q, D, burn):
    """Var(q_k)/Var(D) je Stufe: Mittel ueber Replikationen der Zeitvarianzen (ddof=1), plus Standardfehler ueber Replikationen."""
    N = q.shape[0]
    vd = D[:, burn:].var(axis=1, ddof=1)
    res = []
    for k in range(N):
        vq = q[k, :, burn:].var(axis=1, ddof=1)
        r = vq / vd
        res.append((float(r.mean()), float(r.std(ddof=1) / math.sqrt(len(r)))))
    return res



def calibrate_z(D, Ls, p, gamma=1.0, share=False, target=0.95, passes=2, burn=100, iters=9, zmax=8.0, gammas=None, ps=None, h=1.0, pc=9.0):
    """Waehlt je Stufe das z, sodass der Mengen-Servicegrad beta_k = target erreicht (oberste Stufe zuerst, dann abwaerts, `passes` Durchlaeufe).

    Bisektion auf einem Gitter; gleiche Zufallszahlen (D fest) fuer alle Bewertungen.  Rueckgabe (zs, q, phys, achieved_beta).
    """
    N = len(Ls)
    zs = [1.645] * N
    for _ in range(passes):
        for k in range(N - 1, -1, -1):
            lo, hi = 0.0, zmax
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                trial = list(zs); trial[k] = mid
                q, _, _ = order_streams(D, Ls, p, z=trial, gamma=gamma, share=share, gammas=gammas, ps=ps)
                ph = physical(D, q, Ls, h, pc, burn=burn)
                b = float(ph["beta"][k].mean())
                if b < target:
                    lo = mid
                else:
                    hi = mid
            zs[k] = hi
    q, _, _ = order_streams(D, Ls, p, z=zs, gamma=gamma, share=share, gammas=gammas, ps=ps)
    ph = physical(D, q, Ls, h, pc, burn=burn)
    return zs, q, ph


def gen_shift(rng, R, T, mu, sigma, rate=1.0 / 50.0, size=0.2):
    """Bedarf mit zufaelligen Niveauspruengen: mit Wahrscheinlichkeit `rate` je Periode springt das Niveau um +-size*mu (Vorzeichen zufaellig,
    Niveau auf [0.5, 1.5] mu begrenzt); Streuung sigma absolut (iid Normal)."""
    lvl = np.full(R, float(mu))
    D = np.empty((R, T))
    jump = rng.random((R, T)) < rate
    sign = np.where(rng.random((R, T)) < 0.5, -1.0, 1.0)
    noise = rng.standard_normal((R, T)) * sigma
    for t in range(T):
        lvl = np.clip(lvl + jump[:, t] * sign[:, t] * size * mu, 0.5 * mu, 1.5 * mu)
        D[:, t] = lvl + noise[:, t]
    return D
