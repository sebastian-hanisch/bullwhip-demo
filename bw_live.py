"""Live-Verlauf: EINE Kette über 400 Perioden (die ersten 100 sind Einschwingen, angezeigt werden 300) mit IMMER allen sieben Maßnahmen auf demselben Nachfrageverlauf.

Die Rechnung läuft über bw_model / bw_theory (unverändert aus der Messreihe); hier stehen nur das Zusammenstellen der Maßnahmen, der Lagerverlauf je Periode (nachgespielt wie in
bw_model.physical), die Zuordnung des z je Stufe (vorgerechnet aus der Messreihe oder grob live nachkalibriert), die Kontrollzeile gegen die exakte Theorie und die Schnellkalibrierung.
Gemessen: sieben Ketten zu 400 Perioden zusammen etwa 0,3 s, deshalb kein Knopf; die App speichert das Ergebnis je Einstellung (st.cache_data)."""
import time

import numpy as np

import bw_constants as C
import bw_model as M
import bw_results as R
import bw_theory as T


def gen_demand(cv, rho, seed, periods=C.LIVE_T):
    """Endkundenbedarf des gezeigten Verlaufs: AR(1) mit Mittel 100, Randstreuung 100 · CV, Autokorrelation ρ; Form (1, Perioden). Ein Verlauf je Seed."""
    return M.gen_ar1(np.random.default_rng(int(seed)), 1, int(periods), C.MU, C.MU * float(cv), float(rho))


def measure_setup(measure, gamma, lead, window):
    """Parameter der Kette einer Maßnahme: Wiederbeschaffungszeiten je Stufe, Fenster und die Schalter für order_streams / calibrate_z."""
    lead_eff, window_eff = R.effective_config(measure, lead, window)
    kw = {"gamma": 1.0, "share": False, "gammas": None}
    if measure == C.M_SMOOTH:
        kw["gamma"] = float(gamma)
    elif measure == C.M_RETAIL:
        kw["gammas"] = [float(gamma)] + [1.0] * (C.N_STAGES - 1)
    elif measure == C.M_INFO:
        kw["share"] = True
    elif measure == C.M_INFO_SMOOTH:
        kw["share"] = True
        kw["gamma"] = float(gamma)
    return [lead_eff] * C.N_STAGES, window_eff, kw


def physical_series(D, q, Ls):
    """Lager und Rückstand je Periode (N, R, T), nachgespielt mit derselben Lagerführung wie bw_model.physical (Test: Mittel über die bewerteten Perioden == physical)."""
    N, Rr, Tn = q.shape
    slots = max(Ls) + 1
    mu0 = float(D[:, 0].mean())
    onhand = np.zeros((N, Rr))
    backlog = np.zeros((N, Rr))
    pipe = np.zeros((N, Rr, slots))
    for k in range(N):
        for j in range(Ls[k]):
            pipe[k, :, j % slots] = mu0
    oh_ts = np.zeros((N, Rr, Tn))
    bl_ts = np.zeros((N, Rr, Tn))
    for t in range(Tn):
        for k in range(N):
            onhand[k] += pipe[k, :, t % slots]
            pipe[k, :, t % slots] = 0.0
        for k in range(N):
            x = D[:, t] if k == 0 else q[k - 1, :, t]
            avail = onhand[k]
            owed = backlog[k]
            required = owed + x
            ship = np.minimum(avail, required)
            onhand[k] = avail - ship
            backlog[k] = required - ship
            if k > 0:
                pipe[k - 1, :, (t + Ls[k - 1]) % slots] += ship
        pipe[N - 1, :, (t + Ls[N - 1]) % slots] += q[N - 1, :, t]
        oh_ts[:, :, t] = onhand
        bl_ts[:, :, t] = backlog
    return oh_ts, bl_ts


def amplification(q, D, burn=C.BURN):
    """Var(q_k)/Var(D) je Stufe über die bewerteten Perioden eines Verlaufs (R = 1)."""
    vd = D[0, burn:].var(ddof=1)
    return [float(q[k, 0, burn:].var(ddof=1) / vd) for k in range(q.shape[0])]


def run_measure(D, measure, gamma, lead, window, z, burn=C.BURN):
    """Eine Maßnahme auf dem Verlauf D (1, T) mit den z je Stufe: Bestellungen, Lager- und Rückstandsverlauf, Kennzahlen der bewerteten Perioden."""
    Ls, p, kw = measure_setup(measure, gamma, lead, window)
    q, _, _ = M.order_streams(D, Ls, p, z=list(z), **kw)
    ph = M.physical(D, q, Ls, C.H_COST, C.P_COST, burn=burn)
    oh_ts, bl_ts = physical_series(D, q, Ls)
    n = C.N_STAGES
    return {"measure": measure, "label": C.MEASURE_LABELS[measure], "Ls": Ls, "p": p, "z": [float(v) for v in z], "q": q[:, 0, :], "onhand": oh_ts[:, 0, :], "backlog": bl_ts[:, 0, :],
            "amp": amplification(q, D, burn), "beta": [float(ph["beta"][k, 0]) for k in range(n)], "onhand_mean": [float(ph["onhand"][k, 0]) for k in range(n)],
            "backlog_mean": [float(ph["backlog"][k, 0]) for k in range(n)], "cost": [float(ph["cost"][k, 0]) for k in range(n)],
            "total_onhand": float(ph["onhand"][:, 0].sum()), "total_cost": float(ph["cost"][:, 0].sum()), "zero_share": [float((q[k, 0, burn:] <= C.ZERO_TOL).mean()) for k in range(n)]}


def control_row(D, lead, window, rho, burn=C.BURN):
    """Kontrollzeile: dieselbe Kette ohne Sicherheitsbestand (z = 0) und ohne Abschneidung, auf demselben Verlauf. Dann ist jede Bestellung eine feste Linearkombination des Bedarfs
    (exakte lineare Theorie): größte Abweichung der simulierten Bestellungen von der Faltung des Bedarfs mit dem Kettenfilter (nach dem Einschwingen der Filterlänge), die Verstärkung dieses
    einen Verlaufs, die exakte Verstärkung (Faltung, ρ) und die Produktformel je Stufe."""
    Ls = [int(lead)] * C.N_STAGES
    q, _, _ = M.order_streams(D, Ls, int(window), z=0.0, nonneg=False)
    filters = T.chain_filters(Ls, [int(window)] * C.N_STAGES)
    start = 1 + C.N_STAGES * int(window)
    worst = 0.0
    for k, h in enumerate(filters):
        y = np.convolve(D[0], h)[:D.shape[1]]
        worst = max(worst, float(np.abs(y[start:] - q[k, 0, start:]).max()))
    return {"amp_run": amplification(q, D, burn), "amp_exact": [T.ratio_from_filter(h, float(rho)) for h in filters],
            "amp_product": [T.chen_single_bound(int(lead), int(window)) ** (k + 1) for k in range(C.N_STAGES)], "max_deviation": worst, "start": start}


def solve_live(cv, rho, lead, window, gamma, zmode, seed, z_override=None, data=None, demand=None):
    """Alle aktiven Maßnahmen auf einem Verlauf. z je Stufe: bei zmode 'kalibriert' aus der Messreihe (bw_results.resolve_z) oder, wo z_override (dict Maßnahme -> z) vorliegt, aus der
    Schnellkalibrierung; bei 'fest' überall 1,645. `demand` (1, T) ersetzt den erzeugten Verlauf (Tests mit eingefrorener Nachfragereihe)."""
    data = data if data is not None else R.load_results()
    D = demand if demand is not None else gen_demand(cv, rho, seed)
    z_override = z_override or {}
    measures, notes = {}, []
    active = R.active_measures(lead, window)
    for m in active:
        if m in z_override:
            z, source, exact, note, cfg = list(z_override[m]), "live nachkalibriert (grob)", False, "", f"{C.QUICK_R} Wiederholungen × {C.QUICK_T} Perioden"
        elif zmode == C.ZMODE_FIXED:
            z, source, exact, note, cfg = [C.FIXED_Z] * C.N_STAGES, f"fest {str(C.FIXED_Z).replace('.', ',')}", True, "", "–"
        else:
            r = R.resolve_z(data, m, cv, rho, lead, window, gamma)
            z, source, exact, note = r["z"], ("vorgerechnet" if r["exact"] else "vorgerechnet (nächstliegende Konfiguration)"), r["exact"], r["note"]
            c = r["config"]
            cfg = f"CV {R.F.fmt_level(c['cv'])}, ρ {R.F.fmt_level(c['rho'])}, L {c['lead']}, p {c['window']}" + (f", γ {R.F.fmt_level(c['gamma'])}" if c["gamma"] is not None else "")
        res = run_measure(D, m, gamma, lead, window, z)
        res.update(z_source=source, z_exact=exact, z_note=note, z_config=cfg)
        measures[m] = res
        if note:
            notes.append(note)
    return {"cv": float(cv), "rho": float(rho), "lead": int(lead), "window": int(window), "gamma": float(gamma), "zmode": zmode, "seed": int(seed), "T": int(D.shape[1]), "burn": C.BURN,
            "D": D[0], "active": active, "measures": measures, "control": control_row(D, lead, window, rho), "notes": notes}


def cooldown_left(last, now, cooldown=C.QUICK_COOLDOWN):
    """Verbleibende Pause (Sekunden) bis zum nächsten Aufruf der Schnellkalibrierung; 0, wenn noch keiner lief oder die Pause vorbei ist."""
    if last is None:
        return 0.0
    return max(0.0, cooldown - (now - last))


def quick_calibrate(cv, rho, lead, window, gamma, seed, measures, time_limit=C.QUICK_TIME_LIMIT, clock=time.monotonic, reps=C.QUICK_R, periods=C.QUICK_T, on_progress=None):
    """Grobe Schnellkalibrierung des z je Stufe (Ziel β = 95 %): 20 Wiederholungen × 700 Perioden, ein Durchlauf, dieselbe Stichprobe für alle Maßnahmen. Nach `time_limit` Sekunden werden die
    restlichen Maßnahmen nicht mehr kalibriert (sie behalten das vorgerechnete z). Rückgabe dict: z (Maßnahme -> z), skipped (Liste), seconds."""
    t0 = clock()
    Dc = M.gen_ar1(np.random.default_rng(int(seed) + C.QUICK_SEED_OFFSET), int(reps), int(periods), C.MU, C.MU * float(cv), float(rho))
    out, skipped = {}, []
    for i, m in enumerate(measures):
        if clock() - t0 > time_limit:
            skipped.append(m)
            continue
        if on_progress is not None:
            on_progress(i, len(measures), m)
        Ls, p, kw = measure_setup(m, gamma, lead, window)
        zs, _, _ = M.calibrate_z(Dc, Ls, p, target=C.TARGET_SERVICE, burn=C.BURN, passes=C.QUICK_PASSES, iters=C.QUICK_ITERS, zmax=C.QUICK_ZMAX, **kw)
        out[m] = [float(v) for v in zs]
    return {"z": out, "skipped": skipped, "seconds": clock() - t0}
