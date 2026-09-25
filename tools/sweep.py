"""Messreihen Bullwhip -> tools/_out/sweep_data.json, sweep_log.txt.  Mehrprozess (Pool), alle Seeds fest (Reproduktion der Messreihe, etwa 2 Minuten auf 12 Prozessen; nicht in der CI).

Mechanisch aus messreihe_bullwhip/sweep.py übernommen (nur der Import auf bw_model und die Ausgabepfade sind angepasst).  Aufruf im Projektordner:  python tools/sweep.py

  E1  lineare Theorie gegen Simulation (z = 0, ohne Abschneidung): Stufen 1-4, Chen-Produktformel als Untergrenze
  E2  realistische Kette (z = 1,645, Abschneidung bei 0): Verstaerkung gegen lineare Theorie, Anteil abgeschnittener Bestellungen
  E3  exakte lineare Theorie: Verstaerkung je Stufe gegen L, p, rho; lokal gegen geteilte Bedarfsinformation
  E4  Gegenmassnahmen bei GLEICHEM Servicegrad (beta_k = 95 % je Stufe, z je Stufe kalibriert; Auswertung out-of-sample) und bei festem z
  E5  Regime-Landkarten (rho, L, p, cv) fuer Basis / Glaettung 0,5 / Bedarfsinformation teilen
  E6  Niveauspruenge: Fenstergroesse p gegen Kosten bei gleichem Servicegrad (iid gegen Spruenge)
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "tools" / "_out"
import numpy as np

import bw_model as bull
import bw_theory as theory

N_STAGES = 4
MU = 100.0
TARGET = 0.95
BURN = 100
R_CAL, T_CAL = 200, 1200
R_EVAL, T_EVAL = 500, 1500


def demand(kind, seed, R, T, cv, rho):
    rng = np.random.default_rng(seed)
    if kind == "ar1":
        return bull.gen_ar1(rng, R, T, MU, MU * cv, rho)
    if kind == "shift":
        return bull.gen_shift(rng, R, T, MU, MU * cv)
    raise ValueError(kind)


def se_of(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def summarize(q, D, ph):
    vd = D[:, BURN:].var(axis=1, ddof=1)
    amp = []
    for k in range(q.shape[0]):
        r = q[k, :, BURN:].var(axis=1, ddof=1) / vd
        amp.append([float(r.mean()), se_of(r), float(np.median(r))])
    on = ph["onhand"]                                        # (N, R)
    cost = ph["cost"]
    tot_on = on.sum(axis=0)
    tot_cost = cost.sum(axis=0)
    return {"amp": amp,
            "beta": [float(ph["beta"][k].mean()) for k in range(q.shape[0])],
            "alpha": [float(ph["alpha"][k].mean()) for k in range(q.shape[0])],
            "onhand": [float(on[k].mean()) for k in range(q.shape[0])],
            "backlog": [float(ph["backlog"][k].mean()) for k in range(q.shape[0])],
            "cost": [float(cost[k].mean()) for k in range(q.shape[0])],
            "total_onhand": float(tot_on.mean()), "total_onhand_se": se_of(tot_on),
            "total_cost": float(tot_cost.mean()), "total_cost_se": se_of(tot_cost),
            "zero_share": [float((q[k, :, BURN:] <= 0).mean()) for k in range(q.shape[0])]}


def run_policy(job):
    """Kalibriert z je Stufe (Ziel beta = 95 %) auf Stichprobe A, wertet out-of-sample auf Stichprobe B aus; zusaetzlich fester z = 1,645."""
    sc, pol = job["scenario"], job["policy"]
    L = pol.get("L", sc["L"])
    p = pol.get("p", sc["p"])
    Ls = [L] * N_STAGES
    kw = {"gamma": pol.get("gamma", 1.0), "share": pol.get("share", False), "gammas": pol.get("gammas")}
    DA = demand(sc["kind"], sc["seed"], R_CAL, T_CAL, sc["cv"], sc["rho"])
    DB = demand(sc["kind"], sc["seed"] + 1, R_EVAL, T_EVAL, sc["cv"], sc["rho"])
    out = {"scenario": sc, "policy": pol, "name": job["name"], "L": L, "p": p}
    if job.get("calibrate", True):
        zs, qa, pha = bull.calibrate_z(DA, Ls, p, target=TARGET, burn=BURN, passes=2, iters=9, zmax=10.0, **kw)
        out["z"] = zs
        out["z_at_cap"] = [bool(z >= 9.99) for z in zs]
        out["insample_beta"] = [float(pha["beta"][k].mean()) for k in range(N_STAGES)]
        q, _, _ = bull.order_streams(DB, Ls, p, z=zs, **kw)
        ph = bull.physical(DB, q, Ls, burn=BURN)
        out["equal_service"] = summarize(q, DB, ph)
    if job.get("fixed", False):
        q, _, _ = bull.order_streams(DB, Ls, p, z=1.645, **kw)
        ph = bull.physical(DB, q, Ls, burn=BURN)
        out["fixed_z"] = summarize(q, DB, ph)
    return out


def E1():
    rows = []
    R, T = 300, 1500
    rng = np.random.default_rng(101)
    D = bull.gen_ar1(rng, R, T, 1000.0, 5.0, 0.0)
    for m in (1, 2, 4):
        for p in (3, 5, 10, 20):
            Ls = [m] * N_STAGES
            q, _, _ = bull.order_streams(D, Ls, p, z=0.0, nonneg=False)
            vr = bull.var_ratio(q, D, BURN)
            ex = [theory.ratio_from_filter(f, 0.0) for f in theory.chain_filters(Ls, [p] * N_STAGES)]
            prod = [theory.chen_single_bound(m, p) ** (k + 1) for k in range(N_STAGES)]
            sh = [theory.ratio_from_filter(f, 0.0) for f in theory.chain_filters(Ls, [p] * N_STAGES, share=True)]
            qs, _, _ = bull.order_streams(D, Ls, p, z=0.0, share=True, nonneg=False)
            vs = bull.var_ratio(qs, D, BURN)
            rows.append({"m": m, "p": p, "sim": [a for a, _ in vr], "sim_se": [s for _, s in vr], "exact": ex, "chen_product": prod,
                         "share_sim": [a for a, _ in vs], "share_exact": sh})
    return rows


def E2():
    rows = []
    R, T = 200, 1500
    for cv in (0.1, 0.2, 0.3):
        for m in (1, 2, 4):
            for p in (5, 10, 20):
                D = demand("ar1", 5000 + int(cv * 100) * 10 + m, R, T, cv, 0.0)
                Ls = [m] * N_STAGES
                lin = [theory.ratio_from_filter(f, 0.0) for f in theory.chain_filters(Ls, [p] * N_STAGES)]
                res = {}
                for name, kw in (("z0_nonneg", dict(z=0.0)), ("z1645_nonneg", dict(z=1.645)), ("z1645_linear", dict(z=1.645, nonneg=False))):
                    q, _, _ = bull.order_streams(D, Ls, p, **kw)
                    vr = bull.var_ratio(q, D, BURN)
                    res[name] = {"amp": [a for a, _ in vr], "se": [s for _, s in vr], "zero_share": [float((q[k, :, BURN:] <= 0).mean()) for k in range(N_STAGES)]}
                rows.append({"cv": cv, "m": m, "p": p, "linear_exact": lin, **res})
    return rows


def E3():
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        for m in (1, 2, 4, 8):
            for p in (3, 5, 10, 20, 40):
                Ls = [m] * N_STAGES
                loc = [theory.ratio_from_filter(f, rho) for f in theory.chain_filters(Ls, [p] * N_STAGES)]
                sh = [theory.ratio_from_filter(f, rho) for f in theory.chain_filters(Ls, [p] * N_STAGES, share=True)]
                rows.append({"rho": rho, "m": m, "p": p, "local": loc, "share": sh})
    return rows


def make_jobs():
    jobs = []
    base = {"kind": "ar1", "cv": 0.2, "rho": 0.0, "L": 2, "p": 10}

    def sc(**kw):
        s = dict(base); s.update(kw)
        # Seed haengt nur vom Szenario ab (gemeinsame Zufallszahlen fuer alle Politiken)
        key = (s["kind"], s["cv"], s["rho"], s["L"], s["p"])
        s["seed"] = 10000 + int(1000 * s["cv"]) * 7 + int(10 * s["rho"]) * 131 + s["L"] * 17 + s["p"] * 3 + (500000 if s["kind"] == "shift" else 0)
        return s

    main_pols = [("basis", {}), ("glaettung_0.7", {"gamma": 0.7}), ("glaettung_0.5", {"gamma": 0.5}), ("glaettung_0.3", {"gamma": 0.3}),
                 ("nur_einzelhandel_glaettet_0.5", {"gammas": [0.5, 1, 1, 1]}), ("nur_einzelhandel_glaettet_0.3", {"gammas": [0.3, 1, 1, 1]}),
                 ("nur_fabrik_glaettet_0.3", {"gammas": [1, 1, 1, 0.3]}), ("nur_stufe2_glaettet_0.3", {"gammas": [1, 0.3, 1, 1]}),
                 ("info_teilen", {"share": True}), ("info_teilen_glaettung_0.5", {"share": True, "gamma": 0.5}),
                 ("L_halbiert", {"L": 1}), ("Fenster_doppelt", {"p": 20}), ("L_halbiert_info_teilen", {"L": 1, "share": True})]
    for name, pol in main_pols:
        jobs.append({"name": f"main/{name}", "group": "main", "scenario": sc(), "policy": pol, "fixed": True})
    for name, pol in main_pols:                                   # Wiederholung mit anderen Seeds (Robustheit der Rangfolge)
        s2 = sc(); s2["seed"] += 777
        jobs.append({"name": f"main2/{name}", "group": "main2", "scenario": s2, "policy": pol, "fixed": False})
    three = [("basis", {}), ("glaettung_0.5", {"gamma": 0.5}), ("info_teilen", {"share": True})]
    for rho in (0.5, 0.8):
        for name, pol in three:
            jobs.append({"name": f"rho{rho}/{name}", "group": "rho", "scenario": sc(rho=rho), "policy": pol, "fixed": True})
    for L in (1, 4):
        for name, pol in three:
            jobs.append({"name": f"L{L}/{name}", "group": "L", "scenario": sc(L=L), "policy": pol, "fixed": True})
    for p in (6, 20, 40):
        for name, pol in three:
            jobs.append({"name": f"p{p}/{name}", "group": "p", "scenario": sc(p=p), "policy": pol, "fixed": True})
    for cv in (0.1, 0.3):
        for name, pol in three:
            jobs.append({"name": f"cv{cv}/{name}", "group": "cv", "scenario": sc(cv=cv), "policy": pol, "fixed": True})
    for kind in ("ar1", "shift"):
        for p in (4, 8, 16, 32, 64):
            jobs.append({"name": f"{kind}/p{p}/basis", "group": "shift", "scenario": sc(kind=kind, p=p), "policy": {}, "fixed": True})
    return jobs


def _worker(job):
    t = time.time()
    r = run_policy(job)
    return r, time.time() - t


def run_all(processes=12, lg=print, t0=None):
    """Der ganze Sweep (E1, E2, E3 und alle Kalibrierungsjobs) als dict wie sweep_data.json; `lg` protokolliert."""
    t0 = time.time() if t0 is None else t0
    OUT = {}
    OUT["E1"] = E1(); lg(f"E1 {time.time() - t0:.0f} s")
    OUT["E2"] = E2(); lg(f"E2 {time.time() - t0:.0f} s")
    OUT["E3"] = E3(); lg(f"E3 {time.time() - t0:.0f} s")
    jobs = make_jobs()
    lg(f"{len(jobs)} Kalibrierungsjobs")
    with mp.Pool(processes) as pool:
        res = pool.map(_worker, jobs, chunksize=1)
    OUT["jobs"] = [r for r, _ in res]
    lg(f"Jobs fertig {time.time() - t0:.0f} s; CPU-Sekunden Summe {sum(sec for _, sec in res):.0f}")
    OUT["config"] = {"N": N_STAGES, "MU": MU, "TARGET": TARGET, "BURN": BURN, "R_CAL": R_CAL, "T_CAL": T_CAL, "R_EVAL": R_EVAL, "T_EVAL": T_EVAL}
    return OUT


if __name__ == "__main__":
    mp.freeze_support()
    t0 = time.time()
    log = []

    def lg(s):
        print(s, flush=True)
        log.append(s)
    OUT = run_all(12, lg, t0)
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "sweep_data.json", "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    with open(OUT_DIR / "sweep_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    lg(f"fertig in {time.time() - t0:.0f} s")
