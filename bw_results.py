"""Laden und Auswerten der vorgerechneten Messreihe (data/bw_results.json, erzeugt von tools/build_results.py aus messreihe_bullwhip/sweep_data.json): Jobs klassifizieren, Zell-Zuordnung,
z je Stufe für die Live-Kette, Urteil in drei Zuständen, Tabellen der Kernabschnitte.

Die App rechnet die Messreihe NIE live (der Sweep braucht etwa 2 Minuten auf 12 Prozessen); sie liest nur diese Datei. Reine Rechnung, kein Streamlit.

Begriffe: Ein JOB ist eine Kalibrierung (z je Stufe) samt Auswertung für eine Konfiguration und eine Maßnahme. Eine ZELLE ist ein Satz von Jobs desselben Szenarios (Schwankung CV, Autokorrelation,
Wiederbeschaffungszeit L, Fenster p): das Hauptszenario (`main`, 13 Maßnahmen) und 9 Zellen mit je drei Jobs (Basis, Glättung 0,5, Info teilen). Die Messreihe variiert je Zelle nur EINEN Parameter
ausgehend vom Hauptszenario. Der Änderungswert einer Maßnahme ist immer die Änderung der Summe Lager gegenüber der Basis DERSELBEN Zelle (gleicher Servicegrad)."""
import functools
import json
import math
import pathlib

import numpy as np

import bw_constants as C
import bw_format as F

DATA_PATH = pathlib.Path(__file__).resolve().parent / "data" / "bw_results.json"

CELLS = ("main", "rho0.5", "rho0.8", "L1", "L4", "p6", "p20", "p40", "cv0.1", "cv0.3")
SETTING_PARAMS = ("cv", "rho", "lead", "window")             # Reihenfolge der Einstellung (cv, rho, lead, window)
# Zell-Zuordnung: bei gleichem Abstand gewinnt die Zelle, deren variierter Parameter FRÜHER steht (der einflussreichste Parameter soll übereinstimmen)
PARAM_PRIORITY = ("lead", "window", "rho", "cv")
PARAM_LABELS = {"lead": "Wiederbeschaffungszeit L", "window": "Prognosefenster p", "rho": "Autokorrelation ρ", "cv": "Nachfrageschwankung CV"}
# Einfluss auf das nötige z (für die Zuordnung eines z zur Live-Kette): zuerst L, dann p, ρ, γ, CV
Z_PRIORITY = ("lead", "window", "rho", "gamma", "cv")
GROUP_LABELS = {"rho": "Autokorrelation", "L": "Wiederbeschaffungszeit", "p": "Prognosefenster", "cv": "Nachfrageschwankung", "main": "Hauptszenario"}

KIND_OF_MEASURE = {C.M_BASIS: "basis", C.M_SMOOTH: "glaettung", C.M_RETAIL: "einzelhandel", C.M_INFO: "info", C.M_INFO_SMOOTH: "info_glaettung", C.M_LEAD: "basis", C.M_WINDOW: "basis"}
GAMMA_KINDS = ("glaettung", "einzelhandel", "info_glaettung", "stufe2", "fabrik")

# Die 13 Maßnahmen des Hauptszenarios (Job-Name ohne "main/", Beschriftung), Reihenfolge wie im Plan
MAIN13 = (
    ("basis", "Basis"), ("glaettung_0.7", "Glättung γ 0,7 (alle)"), ("glaettung_0.5", "Glättung γ 0,5 (alle)"), ("glaettung_0.3", "Glättung γ 0,3 (alle)"),
    ("nur_einzelhandel_glaettet_0.5", "nur Einzelhandel glättet, γ 0,5"), ("nur_einzelhandel_glaettet_0.3", "nur Einzelhandel glättet, γ 0,3"),
    ("nur_stufe2_glaettet_0.3", "nur Stufe 2 glättet, γ 0,3"), ("nur_fabrik_glaettet_0.3", "nur Fabrik glättet, γ 0,3"),
    ("info_teilen", "Bedarfsinformation teilen"), ("info_teilen_glaettung_0.5", "Info teilen + Glättung γ 0,5"),
    ("Fenster_doppelt", "Fenster verdoppelt (p 20)"), ("L_halbiert", "Wiederbeschaffungszeit halbiert"), ("L_halbiert_info_teilen", "L halbiert + Info teilen"),
)
LEVER_JOBS = ("L_halbiert", "L_halbiert_info_teilen", "Fenster_doppelt")      # ändern L oder p (für die Rangkorrelation ohne diese Maßnahmen)

SUBJECT = {C.M_SMOOTH: "Glättung", C.M_RETAIL: "Glättung nur im Einzelhandel", C.M_INFO: "Info teilen", C.M_INFO_SMOOTH: "Info teilen + Glättung",
           C.M_LEAD: "Die halbierte Wiederbeschaffungszeit", C.M_WINDOW: "Das verdoppelte Fenster", C.M_BASIS: "Die Basis"}
STATE_VERB = {C.STATE_LOWER: "senkt hier den Kettenbestand", C.STATE_LITTLE: "bringt hier wenig", C.STATE_COSTS: "kostet hier"}


def state_text(measure, state):
    """Überschrift der Meldung: 'Glättung senkt hier den Kettenbestand' / 'bringt hier wenig' / 'kostet hier'."""
    return f"{SUBJECT[measure]} {STATE_VERB[state]}"


def state_of(delta, se):
    """Meldungszustand aus der gemessenen Änderung der Summe Lager (in %, gegenüber der Basis derselben Zelle): 'senkt' (höchstens -5 % und unter -2 Standardfehlern), 'kostet' (über +2 % und
    über +2 Standardfehlern), sonst 'wenig'."""
    if delta <= C.LOWER_PCT and delta < -C.SE_FACTOR * se:
        return C.STATE_LOWER
    if delta > C.RAISE_PCT and delta > C.SE_FACTOR * se:
        return C.STATE_COSTS
    return C.STATE_LITTLE


@functools.lru_cache(maxsize=4)
def _load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_results(path=None):
    """Die vorgerechnete Messreihe (nicht verändern: das Ergebnis ist zwischengespeichert)."""
    return _load(str(path or DATA_PATH))


# ----------------------------------------------------------------------------------------------------------------
# Jobs klassifizieren
# ----------------------------------------------------------------------------------------------------------------
def classify(name, job):
    """Zuordnung eines Jobs: Zelle (Gruppe), Maßnahme des Zellenvergleichs (`measure`), effektive Politik (`kind`) und Glättung, dazu die effektive Konfiguration (cv, rho, L, p)."""
    pol, sc = job["policy"], job["scenario"]
    gamma, gammas, share = pol.get("gamma", 1.0), pol.get("gammas"), bool(pol.get("share", False))
    if gammas:
        idx = next(i for i, g in enumerate(gammas) if g != 1)
        measure = kind = {0: C.M_RETAIL, 1: "stufe2", 3: "fabrik"}[idx]
        gamma = gammas[idx]
    elif share and gamma != 1:
        measure = kind = C.M_INFO_SMOOTH
    elif share:
        measure, kind = ("lead_info", C.M_INFO) if "L" in pol else (C.M_INFO, C.M_INFO)
    elif gamma != 1:
        measure = kind = C.M_SMOOTH
    elif "L" in pol:
        measure, kind = C.M_LEAD, C.M_BASIS
    elif "p" in pol:
        measure, kind = C.M_WINDOW, C.M_BASIS
    else:
        measure = kind = C.M_BASIS
    return {"name": name, "group": name.split("/")[0], "measure": measure, "kind": kind if kind != "info" else C.M_INFO, "gamma": float(gamma) if kind in GAMMA_KINDS or measure in GAMMA_KINDS else None,
            "cv": sc["cv"], "rho": sc["rho"], "L": job["L"], "p": job["p"], "job": job}


def entries(data):
    """Alle Jobs der Zellen (Hauptszenario und die 9 weiteren Zellen), klassifiziert, in der Reihenfolge der Messreihe; Wiederholung (main2) und Niveausprung-Reihen zählen nicht."""
    out = []
    for pos, (name, job) in enumerate(data["jobs"].items()):
        e = classify(name, job)
        if e["group"] in CELLS:
            e["pos"] = pos
            out.append(e)
    return out


def job(data, name):
    return data["jobs"][name]


def amp_last(j, mode="equal"):
    """Verstärkung an der obersten Stufe (Mittel, Standardfehler)."""
    a = j[mode]["amp"][-1]
    return a[0], a[1]


def total(j, mode="equal"):
    return j[mode]["total_onhand"], j[mode]["total_onhand_se"]


def change_pct(j_meas, j_base, mode="equal"):
    """Änderung der Summe Lager in % gegenüber der Basis und ihr Standardfehler (Fehlerfortpflanzung, unabhängig gerechnet: für gepaarte Läufe eher zu groß)."""
    t1, s1 = total(j_meas, mode)
    t0, s0 = total(j_base, mode)
    ratio = t1 / t0
    return 100.0 * (ratio - 1.0), 100.0 * ratio * math.hypot(s1 / t1, s0 / t0)


# ----------------------------------------------------------------------------------------------------------------
# Zellen
# ----------------------------------------------------------------------------------------------------------------
def cell_setting(data, cell):
    """(CV, ρ, L, p) einer Zelle (aus ihrem Basis-Job)."""
    sc = job(data, f"{cell}/basis")["scenario"]
    return sc["cv"], sc["rho"], sc["L"], sc["p"]


BASE_SETTING = (C.CV_DEFAULT, C.RHO_DEFAULT, C.LEAD_DEFAULT, C.WINDOW_DEFAULT)
_AXES = {"cv": C.CV_OPTIONS, "rho": C.RHO_OPTIONS, "lead": C.LEAD_AXIS, "window": C.WINDOW_AXIS}


def _level_index(setting):
    cv, rho, lead, window = setting
    return {"cv": C.CV_OPTIONS.index(cv), "rho": C.RHO_OPTIONS.index(rho), "lead": C.LEAD_AXIS.index(lead), "window": C.WINDOW_AXIS.index(window)}


def differing_params(a, b):
    ia, ib = _level_index(a), _level_index(b)
    return [p for p in PARAM_PRIORITY if ia[p] != ib[p]]


def setting_text(setting):
    cv, rho, lead, window = setting
    return f"CV {F.fmt_level(cv)}, ρ {F.fmt_level(rho)}, L {lead}, p {window}"


def cell_label(data, cell):
    """Beschriftung einer Zelle aus dem Unterschied zum Hauptszenario."""
    s = cell_setting(data, cell)
    parts = []
    for name, sym, a, b in zip(SETTING_PARAMS, ("CV", "ρ", "L", "p"), s, BASE_SETTING):
        if a != b:
            parts.append(f"{sym} {F.fmt_level(a) if name in ('cv', 'rho') else a}")
    return ", ".join(parts) if parts else "Hauptszenario"


def cell_group(cell):
    for prefix, key in (("rho", "rho"), ("L", "L"), ("p", "p"), ("cv", "cv")):
        if cell.startswith(prefix):
            return GROUP_LABELS[key]
    return GROUP_LABELS["main"]


def cell_setting_text(data, cell):
    s = cell_setting(data, cell)
    return setting_text(s) if cell != "main" else f"Hauptszenario ({setting_text(s)})"


def measured_cells(data, measure=None):
    """Zellen, in denen die Maßnahme (und die Basis) gemessen sind; ohne Angabe alle Zellen."""
    if measure is None or measure == C.M_BASIS:
        return [c for c in CELLS if f"{c}/basis" in data["jobs"]]
    have = {e["group"] for e in entries(data) if e["measure"] == measure}
    return [c for c in CELLS if c in have and f"{c}/basis" in data["jobs"]]


def nearest_cell(data, cv, rho, lead, window, measure=None):
    """Die nächstliegende gemessene Zelle (mit der Maßnahme, falls angegeben). Rückgabe (Zelle, exakt, abweichende Parameter): exakt heißt, die Zelle hat genau diese Einstellung. Abstand =
    Summe der Stufenabstände; bei Gleichstand gewinnt die Zelle, deren variierter Parameter in PARAM_PRIORITY früher steht, dann die frühere in der Messreihe."""
    want = (float(cv), float(rho), int(lead), int(window))
    iw = _level_index(want)
    best = None
    for pos, cell in enumerate(measured_cells(data, measure)):
        setting = cell_setting(data, cell)
        ic = _level_index(setting)
        dist = sum(abs(iw[p] - ic[p]) for p in PARAM_PRIORITY)
        varied = differing_params(setting, BASE_SETTING)
        rank = PARAM_PRIORITY.index(varied[0]) if varied else -1
        key = (dist, rank, pos)
        if best is None or key < best[0]:
            best = (key, cell, setting)
    _, cell, setting = best
    diff = differing_params(want, setting)
    return cell, not diff, diff


def measure_active(measure, lead, window):
    """'L halbiert' entfällt bei L = 1, 'Fenster verdoppelt' bei p = 40 (kürzer bzw. länger als die gemessenen Stufen geht es nicht): ohne Wirkung, wird ausgeblendet."""
    if measure == C.M_LEAD:
        return int(lead) > min(C.LEAD_OPTIONS)
    if measure == C.M_WINDOW:
        return 2 * int(window) <= max(C.WINDOW_OPTIONS)
    return True


def active_measures(lead, window):
    return [m for m in C.MEASURES if measure_active(m, lead, window)]


def effective_config(measure, lead, window):
    """Effektive Wiederbeschaffungszeit und Fenster einer Maßnahme (L halbiert -> L // 2, Fenster verdoppelt -> 2 p)."""
    if measure == C.M_LEAD:
        return int(lead) // 2, int(window)
    if measure == C.M_WINDOW:
        return int(lead), 2 * int(window)
    return int(lead), int(window)


def assignment_note(data, cv, rho, lead, window, cell, diff):
    """Hinweis zur Vergleichszelle; leer, wenn die gemessene Zelle genau die Einstellung ist."""
    if not diff:
        return ""
    return (f"Für Ihre Einstellung ({setting_text((cv, rho, lead, window))}) gibt es keine eigene Messreihe: die Messreihe variiert je Zelle nur einen Parameter ausgehend vom Hauptszenario. "
            f"Die Vergleichszahlen der Messreihe zeigen die nächstliegende gemessene Zelle ({cell_setting_text(data, cell)}); abweichend: {', '.join(PARAM_LABELS[p] for p in diff)}.")


# ----------------------------------------------------------------------------------------------------------------
# Meldung: Änderung der Summe Lager einer Maßnahme gegenüber der Basis derselben Zelle
# ----------------------------------------------------------------------------------------------------------------
def gamma_choices(data, cell, measure):
    """Gemessene Glättungsstufen der Maßnahme in der Zelle."""
    return sorted({e["gamma"] for e in entries(data) if e["group"] == cell and e["measure"] == measure and e["gamma"] is not None}, reverse=True)


def cell_entry(data, cell, measure, gamma=None):
    for e in entries(data):
        if e["group"] == cell and e["measure"] == measure and (gamma is None or e["gamma"] is None or abs(e["gamma"] - gamma) < 1e-9):
            return e
    raise KeyError((cell, measure, gamma))


def resolve_pair(data, measure, gamma, cv, rho, lead, window):
    """Zelle, Glättung und Jobs für die Meldung: nächstliegende Zelle mit der Maßnahme, darin die nächstliegende gemessene Glättung. Rückgabe dict mit Änderung (gleicher Servicegrad und fester z),
    Standardfehler, Zustand, Jobnamen, `exact` (Zelle UND Glättung genau) und `note`."""
    cell, exact_cell, diff = nearest_cell(data, cv, rho, lead, window, measure)
    gamma_used, exact_gamma = None, True
    if measure in C.SMOOTH_MEASURES:
        options = gamma_choices(data, cell, measure)
        gamma_used = _nearest_gamma(options, gamma)
        exact_gamma = abs(gamma_used - gamma) < 1e-9
    base = cell_entry(data, cell, C.M_BASIS)
    meas = cell_entry(data, cell, measure, gamma_used)
    delta, se = change_pct(meas["job"], base["job"])
    out = {"measure": measure, "cell": cell, "cell_text": cell_setting_text(data, cell), "cell_exact": exact_cell, "diff": diff, "gamma": gamma_used, "gamma_exact": exact_gamma, "base_name": base["name"], "name": meas["name"],
           "delta": delta, "se": se, "state": state_of(delta, se), "total_base": total(base["job"])[0], "total": total(meas["job"])[0], "exact": exact_cell and exact_gamma}
    if base["job"]["fixed"] and meas["job"]["fixed"]:
        out["fixed_delta"], out["fixed_se"] = change_pct(meas["job"], base["job"], "fixed")
    notes = []
    if not exact_cell:
        notes.append(assignment_note(data, cv, rho, lead, window, cell, diff))
    if not exact_gamma:
        notes.append(f"Die Messreihe hat diese Maßnahme in dieser Zelle nur mit γ {F.fmt_level(gamma_used)} gemessen (nicht γ {F.fmt_level(gamma)}); gezeigt wird γ {F.fmt_level(gamma_used)}.")
    if measure in (C.M_LEAD, C.M_WINDOW, C.M_RETAIL, C.M_INFO_SMOOTH) and cell == "main" and not exact_cell:
        notes.append("Diese Maßnahme ist nur im Hauptszenario gemessen.")
    out["note"] = " ".join(n for n in notes if n)
    return out


# ----------------------------------------------------------------------------------------------------------------
# z je Stufe für die Live-Kette (gleicher Servicegrad): nächstliegende gemessene Konfiguration
# ----------------------------------------------------------------------------------------------------------------
def resolve_z(data, measure, cv, rho, lead, window, gamma):
    """z je Stufe aus der Messreihe für die Live-Kette einer Maßnahme. z gehört zur KONFIGURATION (Schwankung, ρ, effektives L, effektives p, Regel, γ), nicht zur Maßnahme: 'L halbiert' bei L = 4
    ist die Konfiguration L = 2 und nimmt das z des Hauptszenarios. Gesucht wird die gemessene Konfiguration mit gleicher Regel und kleinstem Abstand (Summe der Stufenabstände plus
    Glättungsstufen); bei Gleichstand gewinnt die, die in den einflussreicheren Parametern (L, p, ρ, γ, CV) übereinstimmt, dann die Maßnahme selbst, dann das Hauptszenario.
    Rückgabe dict: z, Jobname, exact (Abstand 0), Abstand, Konfiguration, Hinweistext."""
    lead_eff, window_eff = effective_config(measure, lead, window)
    kind = KIND_OF_MEASURE[measure]
    use_gamma = kind in GAMMA_KINDS
    want = {"cv": C.CV_OPTIONS.index(float(cv)), "rho": C.RHO_OPTIONS.index(float(rho)), "lead": C.LEAD_AXIS.index(lead_eff), "window": C.WINDOW_AXIS.index(window_eff)}
    if use_gamma:
        want["gamma"] = C.GAMMA_OPTIONS.index(float(gamma))
    best = None
    for e in entries(data):
        if e["kind"] != kind:
            continue
        have = {"cv": C.CV_OPTIONS.index(e["cv"]), "rho": C.RHO_OPTIONS.index(e["rho"]), "lead": C.LEAD_AXIS.index(e["L"]), "window": C.WINDOW_AXIS.index(e["p"])}
        if use_gamma:
            if e["gamma"] is None or e["gamma"] not in C.GAMMA_OPTIONS:
                continue
            have["gamma"] = C.GAMMA_OPTIONS.index(e["gamma"])
        diffs = {p: abs(want[p] - have[p]) for p in want}
        differing = [Z_PRIORITY.index(p) for p, d in diffs.items() if d]
        key = (sum(diffs.values()), -(min(differing) if differing else 99), 0 if e["measure"] == measure else 1, 0 if e["group"] == "main" else 1, e["pos"])
        if best is None or key < best[0]:
            best = (key, e, diffs)
    if best is None:
        raise KeyError((measure, cv, rho, lead, window, gamma))
    key, e, diffs = best
    exact = key[0] == 0
    cfg = {"cv": e["cv"], "rho": e["rho"], "lead": e["L"], "window": e["p"], "gamma": e["gamma"]}
    note = ""
    if not exact:
        note = (f"{C.MEASURE_LABELS[measure]}: keine eigene Kalibrierung für CV {F.fmt_level(cv)}, ρ {F.fmt_level(rho)}, L {lead_eff}, p {window_eff}"
                + (f", γ {F.fmt_level(gamma)}" if use_gamma else "") + f"; z der nächstliegenden gemessenen Konfiguration (CV {F.fmt_level(e['cv'])}, ρ {F.fmt_level(e['rho'])}, L {e['L']}, p {e['p']}"
                + (f", γ {F.fmt_level(e['gamma'])}" if use_gamma else "") + ").")
    return {"z": list(e["job"]["z"]), "job": e["name"], "exact": exact, "distance": key[0], "config": cfg, "note": note}


# ----------------------------------------------------------------------------------------------------------------
# Abdeckung (Bau-Vorprüfung AP 0)
# ----------------------------------------------------------------------------------------------------------------
def coverage(data):
    """Abdeckung der Reglerstellungen durch gemessene Zellen und durch gemessene z-Konfigurationen: Zahl der Kombinationen aus CV, ρ, L, p (mit Abstand zur nächstliegenden Zelle) und der
    (Kombination, γ, Maßnahme)-Fälle mit genau gemessenem z."""
    import itertools
    combos = list(itertools.product(C.CV_OPTIONS, C.RHO_OPTIONS, C.LEAD_OPTIONS, C.WINDOW_OPTIONS))
    by_dist = {}
    for combo in combos:
        _, _, diff = nearest_cell(data, *combo)
        by_dist[len(diff)] = by_dist.get(len(diff), 0) + 1
    z_total = z_exact = 0
    per_measure = {m: [0, 0] for m in C.MEASURES}
    for combo in combos:
        cv, rho, lead, window = combo
        for gamma in C.GAMMA_OPTIONS:
            for m in active_measures(lead, window):
                if m not in C.SMOOTH_MEASURES and gamma != C.GAMMA_DEFAULT:
                    continue                                              # ohne Glättung zählt γ nicht: nur einmal zählen
                r = resolve_z(data, m, cv, rho, lead, window, gamma)
                z_total += 1
                z_exact += int(r["exact"])
                per_measure[m][0] += 1
                per_measure[m][1] += int(r["exact"])
    cells_ = [c for c in CELLS]
    return {"combos": len(combos), "cell_exact": by_dist.get(0, 0), "by_distance": dict(sorted(by_dist.items())), "cells": cells_, "z_cases": z_total, "z_exact": z_exact,
            "z_per_measure": per_measure}


# ----------------------------------------------------------------------------------------------------------------
# Tabellen der Kernabschnitte
# ----------------------------------------------------------------------------------------------------------------
def main_rows(data):
    """Die 13 Maßnahmen des Hauptszenarios bei gleichem Servicegrad: Summe Lager ± SE, Änderung ± SE gegenüber der Basis, Verstärkung je Stufe, Servicegrad je Stufe, Lager je Stufe, Kosten, z."""
    base = job(data, "main/basis")
    rows = []
    for key, label in MAIN13:
        j = job(data, f"main/{key}")
        e = j["equal"]
        d, dse = change_pct(j, base)
        rows.append({"key": key, "name": f"main/{key}", "label": label, "total": e["total_onhand"], "se": e["total_onhand_se"], "delta": d, "delta_se": dse, "amp": [a[0] for a in e["amp"]],
                     "amp_se": [a[1] for a in e["amp"]], "beta": e["beta"], "onhand": e["onhand"], "cost": e["total_cost"], "cost_se": e["total_cost_se"], "z": j["z"], "L": j["L"], "p": j["p"]})
    return rows


def fixed_rows(data):
    """Dieselben 13 Maßnahmen bei festem z = 1,645 gegen gleicher Servicegrad: Summe Lager, Servicegrad je Stufe, Kettenkosten."""
    rows = []
    for key, label in MAIN13:
        j = job(data, f"main/{key}")
        eq, fx = j["equal"], j["fixed"]
        rows.append({"key": key, "label": label, "total_equal": eq["total_onhand"], "total_fixed": fx["total_onhand"], "beta_fixed": fx["beta"], "beta_equal": eq["beta"],
                     "cost_equal": eq["total_cost"], "cost_fixed": fx["total_cost"], "backlog_fixed": sum(fx["backlog"]), "backlog_equal": sum(eq["backlog"])})
    return rows


def _nearest_gamma(options, gamma):
    return min(options, key=lambda g: (abs(g - gamma), abs(g - C.GAMMA_DEFAULT), -g))


def payer_rows(data, gamma):
    """Wer zahlt: Lager je Stufe bei Basis, 'alle glätten' und 'nur Einzelhandel glättet' im Hauptszenario (nächstliegende gemessene Glättung zu `gamma`), Änderung je Stufe in %."""
    g_all = _nearest_gamma(gamma_choices(data, "main", C.M_SMOOTH), gamma)
    g_one = _nearest_gamma(gamma_choices(data, "main", C.M_RETAIL), gamma)
    base = job(data, "main/basis")["equal"]["onhand"]
    out = {"gamma_all": g_all, "gamma_retail": g_one, "base": base}
    for key, m, g in (("all", C.M_SMOOTH, g_all), ("retail", C.M_RETAIL, g_one)):
        oh = job(data, cell_entry(data, "main", m, g)["name"])["equal"]["onhand"]
        out[key] = oh
        out[key + "_change"] = [100.0 * (a / b - 1.0) for a, b in zip(oh, base)]
    return out


def regime_rows(data, measure, gamma):
    """Regime-Landkarte: je Zelle die Änderung der Summe Lager einer Maßnahme (Glättung alle Stufen oder Info teilen) gegenüber der Basis derselben Zelle, mit Urteil in drei Zuständen."""
    rows = []
    for cell in measured_cells(data, measure):
        base = cell_entry(data, cell, C.M_BASIS)
        g = None
        if measure in C.SMOOTH_MEASURES:
            g = _nearest_gamma(gamma_choices(data, cell, measure), gamma)
        meas = cell_entry(data, cell, measure, g)
        d, se = change_pct(meas["job"], base["job"])
        rows.append({"cell": cell, "group": cell_group(cell), "label": cell_label(data, cell), "setting": cell_setting(data, cell), "gamma": g, "base_total": total(base["job"])[0],
                     "delta": d, "se": se, "state": state_of(d, se), "amp4_base": amp_last(base["job"])[0], "amp4": amp_last(meas["job"])[0]})
    return rows


def _rank(v):
    o = np.argsort(v)
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    return r


def spearman(x, y):
    """Rangkorrelation (Rang nach Reihenfolge, wie in der Messreihe)."""
    return float(np.corrcoef(_rank(list(x)), _rank(list(y)))[0, 1])


def proxy_rows(data):
    """Verstärkung an Stufe 4 gegen Summe Lager über die 13 Maßnahmen (gleicher Servicegrad) und die Rangkorrelation (alle; ohne die Maßnahmen, die L oder p ändern)."""
    rows = main_rows(data)
    pts = [{"label": r["label"], "key": r["key"], "amp4": r["amp"][3], "total": r["total"], "lever": r["key"] in LEVER_JOBS} for r in rows]
    keep = [p for p in pts if not p["lever"]]
    return {"points": pts, "rho_all": spearman([p["amp4"] for p in pts], [p["total"] for p in pts]), "rho_no_levers": spearman([p["amp4"] for p in keep], [p["total"] for p in keep]),
            "n_all": len(pts), "n_no_levers": len(keep)}


def formula_rows(data):
    """Formel gegen Simulation (E1, i.i.d., z = 0, ohne Abschneidung): je (m, p) Verstärkung an Stufe 4 aus Simulation, exakter Theorie und Produktformel, Verhältnis exakt/Produkt."""
    rows = []
    for r in data["E1"]:
        rows.append({"m": r["m"], "p": r["p"], "sim": r["sim"], "sim_se": r["sim_se"], "exact": r["exact"], "product": r["chen_product"], "ratio4": r["exact"][3] / r["chen_product"][3],
                     "share_exact": r["share_exact"], "share_sim": r["share_sim"]})
    return rows


def formula_case(data, m, p):
    return next(r for r in formula_rows(data) if r["m"] == m and r["p"] == p)


def window_jump_rows(data):
    """Fenster mit Niveausprüngen: Summe Lager bei gleichem Servicegrad über das Fenster, i.i.d.-Bedarf gegen Bedarf mit Sprüngen."""
    rows = []
    for p in (4, 8, 16, 32, 64):
        a, s = job(data, f"ar1/p{p}/basis"), job(data, f"shift/p{p}/basis")
        rows.append({"p": p, "iid": total(a)[0], "iid_se": total(a)[1], "shift": total(s)[0], "shift_se": total(s)[1], "amp4_iid": amp_last(a)[0], "amp4_shift": amp_last(s)[0],
                     "beta1_fixed_iid": a["fixed"]["beta"][0], "beta1_fixed_shift": s["fixed"]["beta"][0]})
    return rows


def robustness(data):
    """Wiederholung mit anderen Seeds: Änderung der 13 Summen Lager gegenüber dem ersten Lauf (min, max in %), Rangkorrelation."""
    names = [k for k, _ in MAIN13]
    a = [total(job(data, f"main/{k}"))[0] for k in names]
    b = [total(job(data, f"main2/{k}"))[0] for k in names]
    diffs = [100.0 * (y / x - 1.0) for x, y in zip(a, b)]
    return {"min": min(diffs), "max": max(diffs), "spearman": spearman(a, b), "n": len(names)}


def service_range(data):
    """Kleinster und größter Servicegrad je Stufe über alle Jobs (gleicher Servicegrad, ausserhalb der Kalibrierstichprobe ausgewertet)."""
    vals = [b for j in data["jobs"].values() for b in j["equal"]["beta"]]
    return min(vals), max(vals)
