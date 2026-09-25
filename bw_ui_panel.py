"""Wiederverwendbares Panel: Kennzahlen (2 x 2), Meldung in drei Zuständen, Vergleichstabelle Verlauf gegen Messreihe, Tabellen der Live-Ansicht und der Kernabschnitte ② (Plan Abschnitt 6/8).
Die Tabellen sind reine Funktionen (pandas), die Darstellung ist getrennt."""
import pandas as pd
import streamlit as st

import bw_constants as C
import bw_format as F
import bw_results as R

CHART_TABLE_HEIGHT = C.CHART_HEIGHT + 60
STAGE_LABELS = [f"{k + 1} ({n})" for k, n in enumerate(C.STAGE_NAMES)]


def measure_label(measure, gamma=None):
    """Beschriftung einer Maßnahme mit ihrer Glättung: 'alle glätten (γ 0,5)', sonst der Name."""
    label = C.MEASURE_LABELS[measure]
    return f"{label} (γ {F.fmt_level(gamma)})" if measure in C.SMOOTH_MEASURES and gamma is not None else label


def what_text(measure, gamma):
    """Was die Maßnahme tut, für die Meldung."""
    g = F.fmt_level(gamma)
    return {C.M_SMOOTH: f"Glättung an allen Stufen (γ {g})", C.M_RETAIL: f"Glättung nur im Einzelhandel (γ {g})", C.M_INFO: "Bedarfsinformation teilen (alle Stufen prognostizieren aus dem Endkundenbedarf)",
            C.M_INFO_SMOOTH: f"Info teilen mit Glättung an allen Stufen (γ {g})", C.M_LEAD: "die halbierte Wiederbeschaffungszeit", C.M_WINDOW: "das verdoppelte Prognosefenster",
            C.M_BASIS: "die Basis"}[measure]


def z_hint(result):
    """Kurzer Hinweis zur Herkunft des z einer Maßnahme (leer, wenn genau vorgerechnet oder fest)."""
    if result["z_source"].startswith("live"):
        return "z grob nachkalibriert"
    return "" if result["z_exact"] else "z aus nächstliegender Konfiguration"


def service_warning(live, measure):
    """Warnung im Modus 'gleicher Servicegrad', wenn das z der Basis oder der gezeigten Maßnahme aus der nächstliegenden gemessenen Konfiguration stammt und der Servicegrad dieses Verlaufs
    mehr als 5 Prozentpunkte vom Ziel 95 % abweicht: dann vergleicht die Summe Lager nicht mehr bei gleichem Servicegrad. Leer sonst."""
    if live["zmode"] != C.ZMODE_EQUAL:
        return ""
    bad = []
    for m in dict.fromkeys((C.M_BASIS, measure)):
        v = live["measures"][m]
        if v["z_source"].startswith("vorgerechnet (") and max(abs(b - C.TARGET_SERVICE) for b in v["beta"]) > C.SERVICE_WARN_GAP:
            bad.append(f"{C.MEASURE_LABELS[m]} {F.fmt_num(100 * min(v['beta']), 1)} bis {F.fmt_num(100 * max(v['beta']), 1)} %")
    if not bad:
        return ""
    return ("Achtung: das z je Stufe stammt hier aus der nächstliegenden gemessenen Konfiguration und passt nicht zu dieser Einstellung – der Servicegrad dieses Verlaufs liegt weit vom Ziel 95 % entfernt ("
            + "; ".join(bad) + "). Die Summe Lager ist dann nicht bei gleichem Servicegrad vergleichbar. „Servicegrad live nachkalibrieren“ bestimmt z neu (grob).")


def live_changes(live, measure):
    """Kennzahlen dieses Verlaufs: Verstärkung Stufe 4 und Summe Lager von Basis und Maßnahme, Änderungen in %."""
    b, m = live["measures"][C.M_BASIS], live["measures"][measure]
    return {"amp_base": b["amp"][-1], "amp": m["amp"][-1], "amp_change": 100.0 * (m["amp"][-1] / b["amp"][-1] - 1.0), "total_base": b["total_onhand"], "total": m["total_onhand"],
            "total_diff": m["total_onhand"] - b["total_onhand"], "total_change": 100.0 * (m["total_onhand"] / b["total_onhand"] - 1.0), "beta": m["beta"], "z": m["z"], "beta_base": b["beta"]}


# ---------------------------------------------------------------------------------------------------
# Live-Verlauf: Kennzahlen und Meldung
# ---------------------------------------------------------------------------------------------------
def render_metrics(columns, live, measure, pair):
    """Vier Kennzahlen im 2 x 2-Raster: Verstärkung Stufe 4, Summe Lager der Basis, Summe Lager der Maßnahme (mit Änderung), Servicegrad je Stufe (Zahl: kleinste Stufe, darunter alle vier). Ein
    Verlauf. Die Karten tragen kurze Beschriftungen (die Werte müssen auch auf einem schmalen Bildschirm ganz sichtbar bleiben). Das Delta der Summe Lager ist immer 'Maßnahme minus Basis'
    (negativ = weniger Bestand, deshalb delta_color='inverse')."""
    x = live_changes(live, measure)
    short = C.MEASURE_SHORT[measure]
    hint = z_hint(live["measures"][measure])
    columns[0].metric(f"Verstärkung Stufe 4 ({short})", F.fmt_num(x["amp"], 1), delta=f"{F.fmt_num(x['amp_change'], 0, True)} % gegenüber Basis ({F.fmt_num(x['amp_base'], 1)})", delta_color="off",
                      delta_arrow="off", help="Var(Bestellung der Fabrik) / Var(Endkundenbedarf) über die 300 gezeigten Perioden, ein Verlauf. Verstärkung ist nicht Kosten: was sie bei gleichem Servicegrad an Bestand kostet, zeigen die nächsten Kennzahlen.")
    columns[1].metric("Summe Lager (Basis)", F.fmt_num(x["total_base"], 0), delta=f"Messreihe: {F.fmt_num(pair['total_base'], 0)}", delta_color="off", delta_arrow="off",
                      help="Mittleres Lager aller vier Stufen zusammen (Einheiten) in diesem einen Verlauf, für die Basis (jede Stufe bestellt nach Order-up-to mit gleitendem Durchschnitt). Daneben der Wert der vorgerechneten Messreihe (500 Wiederholungen, Zelle mit dieser Einstellung).")
    columns[2].metric(f"Summe Lager ({short})", F.fmt_num(x["total"], 0), delta=f"{F.fmt_num(x['total_diff'], 0, True)} ({F.fmt_num(x['total_change'], 1, True)} %)", delta_color="inverse",
                      help="Mittleres Lager aller vier Stufen zusammen in diesem Verlauf für die gewählte Maßnahme; Delta = Maßnahme minus Basis (negativ heißt weniger Bestand), in Klammern die Änderung in %. Bei gleichem Servicegrad, wenn der Modus 'gleicher Servicegrad' gewählt ist.")
    target = "Ziel 95 % je Stufe." if live["zmode"] == C.ZMODE_EQUAL else f"Festes z = {F.fmt_num(C.FIXED_Z, 3)}: der Servicegrad wird nicht festgehalten."
    columns[3].metric("Servicegrad, kleinste Stufe", F.fmt_num(100 * min(x["beta"]), 1) + " %", delta=" / ".join(F.fmt_num(100 * b, 1) for b in x["beta"]),
                      delta_color="off", delta_arrow="off",
                      help="Mengen-Servicegrad β: Anteil des neuen Bedarfs, der sofort aus dem Lager geliefert wird. Die Zahl ist die kleinste der vier Stufen, darunter stehen alle Stufen (Einzelhandel bis Fabrik), in diesem einen Verlauf. "
                           + target + (f" z-Herkunft: {hint}." if hint else "") + " Ein einzelner Verlauf streut um das Ziel der Messreihe.")
    return x


def message(live, measure, pair):
    """Bedingte Meldung der Hauptansicht in drei Zuständen aus der VORGERECHNETEN Messreihe (nicht aus dem einen Verlauf): Änderung der Summe Lager gegenüber der Basis derselben Zelle bei gleichem
    Servicegrad. Rückgabe (Zustand, Text) mit Zustand 'senkt', 'wenig' oder 'kostet'."""
    x = live_changes(live, measure)
    state = pair["state"]
    headline = R.state_text(measure, state)
    where = f"Messreihe ({pair['cell_text']}; 500 Wiederholungen)"
    band = f"{F.fmt_num(pair['delta'], 1, True)} ± {F.fmt_num(pair['se'], 1)} %"
    sums = f"Summe Lager {F.fmt_num(pair['total_base'], 0)} → {F.fmt_num(pair['total'], 0)} Einheiten"
    what = what_text(measure, pair["gamma"] if pair["gamma"] is not None else live["gamma"])
    tail = f" Auf diesem einen Verlauf: {F.fmt_num(x['total_change'], 1, True)} %." + (f" {pair['note']}" if pair["note"] else "")
    common = f"{what[:1].upper() + what[1:]}: laut {where} bei gleichem Servicegrad (β = 95 % je Stufe) {sums} ({band})."
    if state == C.STATE_LOWER:
        return state, f"✅ {headline}: {common} Wer den Bestand der Kette senken will, findet hier einen Hebel.{tail}"
    if state == C.STATE_LITTLE:
        return state, f"ℹ️ {headline}: {common} Die Änderung liegt zwischen −5 % und +2 % (oder ist nicht klar von 0 zu unterscheiden): in diesem Regime bringt die Maßnahme bei gleichem Servicegrad kaum etwas.{tail}"
    return state, f"⚠️ {headline}: {common} Bei gleichem Servicegrad zahlt die Kette hier mehr Lager, obwohl die Bestellungen ruhiger werden.{tail}"


def render_message(live, measure, pair):
    state, text = message(live, measure, pair)
    {C.STATE_LOWER: st.success, C.STATE_LITTLE: st.info, C.STATE_COSTS: st.warning}[state](text)
    return state


def fixed_z_text(pair):
    """Zeile im Modus 'festes z': was dieselbe Maßnahme bei festem z aussieht (Messreihe) - sie täuscht, weil der Servicegrad nicht festgehalten wird."""
    if "fixed_delta" not in pair:
        return ""
    return (f"Bei **festem z = 1,645** (Servicegrad nicht festgehalten) sähe dieselbe Maßnahme in der Messreihe so aus: Summe Lager {F.fmt_num(pair['fixed_delta'], 1, True)} ± {F.fmt_num(pair['fixed_se'], 1)} % "
            "gegenüber der Basis, aber mit einem anderen Servicegrad als die Basis (siehe Abschnitt „Fester z gegen gleicher Servicegrad“): ein Vergleich ohne festgehaltenen Servicegrad täuscht.")


def comparison_table(live, measure, pair, data):
    """Dieser Verlauf gegen die Messreihe (nächstliegende Zelle): Summen Lager, Änderung und Verstärkung Stufe 4, mit Standardfehler der Messreihe."""
    x = live_changes(live, measure)
    jb, jm = R.job(data, pair["base_name"]), R.job(data, pair["name"])
    tb, sb = R.total(jb)
    tm, sm = R.total(jm)
    ab, asb = R.amp_last(jb)
    am, asm = R.amp_last(jm)
    col_live = f"Dieser Verlauf (Seed {live['seed']}, 1 Verlauf)"
    col_ms = "Messreihe (500 Wiederholungen, Mittel ± SE)"
    rows = [
        ("Summe Lager Basis", F.fmt_num(x["total_base"], 0), f"{F.fmt_num(tb, 0)} ± {F.fmt_num(sb, 1)}"),
        (f"Summe Lager {C.MEASURE_LABELS[measure]}", F.fmt_num(x["total"], 0), f"{F.fmt_num(tm, 0)} ± {F.fmt_num(sm, 1)}"),
        ("Änderung der Summe Lager (%)", F.fmt_num(x["total_change"], 1, True), F.fmt_band(pair["delta"], pair["se"])),
        ("Verstärkung Stufe 4, Basis", F.fmt_num(x["amp_base"], 1), f"{F.fmt_num(ab, 1)} ± {F.fmt_num(asb, 1)}"),
        (f"Verstärkung Stufe 4, {C.MEASURE_LABELS[measure]}", F.fmt_num(x["amp"], 1), f"{F.fmt_num(am, 1)} ± {F.fmt_num(asm, 1)}"),
    ]
    return pd.DataFrame([{"Kennzahl": a, col_live: b, col_ms: c} for a, b, c in rows])


# ---------------------------------------------------------------------------------------------------
# Live-Tabellen
# ---------------------------------------------------------------------------------------------------
def measures_frame(live):
    """Alle aktiven Maßnahmen auf diesem Verlauf: z je Stufe (Herkunft), Verstärkung Stufe 4, Summe Lager, Änderung gegenüber der Basis, Servicegrad je Stufe, Kettenkosten."""
    base = live["measures"][C.M_BASIS]
    rows = []
    for m in live["active"]:
        v = live["measures"][m]
        rows.append({"Maßnahme": measure_label(m, live["gamma"]), "z je Stufe": " / ".join(F.fmt_num(z, 2) for z in v["z"]), "Herkunft z": v["z_source"], "z gehört zur Konfiguration": v["z_config"],
                     "Verstärkung Stufe 4": F.fmt_num(v["amp"][-1], 1), "Summe Lager": F.fmt_num(v["total_onhand"], 0),
                     "Änderung (%)": "Bezug" if m == C.M_BASIS else F.fmt_num(100 * (v["total_onhand"] / base["total_onhand"] - 1), 1, True),
                     "Servicegrad je Stufe (%)": " / ".join(F.fmt_num(100 * b, 1) for b in v["beta"]), "Kettenkosten": F.fmt_num(v["total_cost"], 0)})
    return pd.DataFrame(rows)


def payer_live_frame(live, measure):
    """Wer zahlt auf diesem Verlauf: mittleres Lager je Stufe, Basis gegen Maßnahme, Änderung je Stufe in %."""
    b, m = live["measures"][C.M_BASIS], live["measures"][measure]
    rows = []
    for k in range(C.N_STAGES):
        rows.append({"Stufe": STAGE_LABELS[k], "Lager Basis": F.fmt_num(b["onhand_mean"][k], 0), f"Lager {C.MEASURE_LABELS[measure]}": F.fmt_num(m["onhand_mean"][k], 0),
                     "Änderung (%)": F.fmt_num(100 * (m["onhand_mean"][k] / b["onhand_mean"][k] - 1), 1, True), "Rückstand Basis": F.fmt_num(b["backlog_mean"][k], 1),
                     "Rückstand Maßnahme": F.fmt_num(m["backlog_mean"][k], 1)})
    return pd.DataFrame(rows)


def amp_frame(live, measure):
    """Verstärkung je Stufe: Basis und Maßnahme in diesem Verlauf (mit Sicherheitsbestand und Abschneidung), dazu die Kontrollzeile: Simulation ohne z und ohne Abschneidung, exakte lineare
    Theorie (Faltung) und Produktformel; Anteil Nullbestellungen (Abschneidung bei 0) der Basis."""
    c = live["control"]
    b, m = live["measures"][C.M_BASIS], live["measures"][measure]
    rows = []
    for k in range(C.N_STAGES):
        rows.append({"Stufe": STAGE_LABELS[k], "Basis (dieser Verlauf)": F.fmt_num(b["amp"][k], 2), f"{C.MEASURE_LABELS[measure]} (dieser Verlauf)": F.fmt_num(m["amp"][k], 2),
                     "Kontrolle: Simulation z = 0": F.fmt_num(c["amp_run"][k], 2), "exakte Theorie (Faltung)": F.fmt_num(c["amp_exact"][k], 2), "Produktformel (Untergrenze)": F.fmt_num(c["amp_product"][k], 2),
                     "Bestellung = 0 (Basis)": F.fmt_share(b["zero_share"][k])})
    return pd.DataFrame(rows)


def control_text(live):
    c = live["control"]
    return (f"Kontrolle: dieselbe Kette ohne Sicherheitsbestand (z = 0) und ohne Abschneidung ist eine lineare Filterkette; ihre Bestellungen stimmen mit der Faltung des Bedarfs mit dem Kettenfilter "
            f"auf {c['max_deviation']:.1e} überein (nach dem Einschwingen, ab Periode {c['start']}). Die Verstärkung dieses einen Verlaufs streut um die exakte Theorie (ein Verlauf, nicht die Messreihe); "
            "die Produktformel liegt ab Stufe 2 darunter (nur eine Untergrenze).")


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②: Tabellen (vorgerechnet)
# ---------------------------------------------------------------------------------------------------
def main_frame(rows):
    """Die 13 Maßnahmen bei gleichem Servicegrad (Hauptszenario)."""
    out = []
    for r in rows:
        out.append({"Maßnahme": r["label"], "z je Stufe": " / ".join(F.fmt_num(z, 2) for z in r["z"]), "Verstärkung Stufen 1 bis 4": " / ".join(F.fmt_num(a, 2) for a in r["amp"]),
                    "Summe Lager (Mittel ± SE)": f"{F.fmt_num(r['total'], 0)} ± {F.fmt_num(r['se'], 1)}", "Änderung (%)": "Bezug" if r["key"] == "basis" else F.fmt_band(r["delta"], r["delta_se"]),
                    "Servicegrad je Stufe (%)": " / ".join(F.fmt_num(100 * b, 1) for b in r["beta"]), "Kettenkosten": F.fmt_num(r["cost"], 0)})
    return pd.DataFrame(out)


def fixed_frame(rows):
    """Fester z gegen gleicher Servicegrad: Summe Lager, Servicegrad je Stufe bei festem z, Kettenkosten."""
    out = []
    for r in rows:
        out.append({"Maßnahme": r["label"], "Summe Lager fest z → gleicher Servicegrad": f"{F.fmt_num(r['total_fixed'], 0)} → {F.fmt_num(r['total_equal'], 0)}",
                    "Servicegrad je Stufe bei festem z (%)": " / ".join(F.fmt_num(100 * b, 1) for b in r["beta_fixed"]), "Rückstand (Summe) fest z → gleicher Servicegrad": f"{F.fmt_num(r['backlog_fixed'], 1)} → {F.fmt_num(r['backlog_equal'], 1)}",
                    "Kettenkosten fest z → gleicher Servicegrad": f"{F.fmt_num(r['cost_fixed'], 0)} → {F.fmt_num(r['cost_equal'], 0)}"})
    return pd.DataFrame(out)


def payer_frame(payer):
    """Wer zahlt (Hauptszenario): mittleres Lager je Stufe und Änderung gegenüber der Basis, alle glätten gegen nur Einzelhandel."""
    rows = []
    for k in range(C.N_STAGES):
        rows.append({"Stufe": STAGE_LABELS[k], "Lager Basis": F.fmt_num(payer["base"][k], 0),
                     f"alle glätten (γ {F.fmt_level(payer['gamma_all'])})": f"{F.fmt_num(payer['all'][k], 0)} ({F.fmt_num(payer['all_change'][k], 1, True)} %)",
                     f"nur Einzelhandel glättet (γ {F.fmt_level(payer['gamma_retail'])})": f"{F.fmt_num(payer['retail'][k], 0)} ({F.fmt_num(payer['retail_change'][k], 1, True)} %)"})
    return pd.DataFrame(rows)


def regime_frame(smooth_rows, info_rows, marked_cell=None):
    """Regime-Tabelle: je Zelle die Änderung der Summe Lager durch Glättung und durch Info teilen mit Urteil in drei Zuständen."""
    out = []
    for s, i in zip(smooth_rows, info_rows):
        out.append({"Gruppe": s["group"], "Zelle": ("▶ " if s["cell"] == marked_cell else "") + s["label"], "Summe Lager Basis": F.fmt_num(s["base_total"], 0),
                    f"Glättung γ {F.fmt_level(s['gamma'])} (%)": F.fmt_band(s["delta"], s["se"]), "Urteil Glättung": R.state_text(C.M_SMOOTH, s["state"]),
                    "Info teilen (%)": F.fmt_band(i["delta"], i["se"]), "Urteil Info teilen": R.state_text(C.M_INFO, i["state"])})
    return pd.DataFrame(out)


def proxy_frame(proxy):
    """Verstärkung an Stufe 4 gegen Summe Lager je Maßnahme (13 Maßnahmen), mit Rang nach Verstärkung und nach Bestand."""
    pts = proxy["points"]
    ra = sorted(range(len(pts)), key=lambda i: pts[i]["amp4"])
    rt = sorted(range(len(pts)), key=lambda i: pts[i]["total"])
    rank_a = {i: r + 1 for r, i in enumerate(ra)}
    rank_t = {i: r + 1 for r, i in enumerate(rt)}
    return pd.DataFrame([{"Maßnahme": p["label"], "Verstärkung Stufe 4": F.fmt_num(p["amp4"], 2), "Rang (kleinste Verstärkung = 1)": rank_a[i], "Summe Lager": F.fmt_num(p["total"], 0),
                          "Rang (kleinstes Lager = 1)": rank_t[i]} for i, p in enumerate(pts)])


def formula_frame(rows):
    """Formel gegen Simulation, Stufe 4 (i.i.d., z = 0): Simulation ± SE, exakte Theorie, Produktformel und das Verhältnis exakt/Produkt."""
    return pd.DataFrame([{"m (= L)": r["m"], "p": r["p"], "m/p": F.fmt_num(r["m"] / r["p"], 2), "Simulation Stufe 4 (± SE)": f"{F.fmt_num(r['sim'][3], 2)} ± {F.fmt_num(r['sim_se'][3], 2)}",
                          "exakte Theorie Stufe 4": F.fmt_num(r["exact"][3], 2), "Produktformel Stufe 4": F.fmt_num(r["product"][3], 2), "exakt / Produkt": F.fmt_num(r["ratio4"], 2)} for r in rows])


def window_jump_frame(rows):
    return pd.DataFrame([{"Fenster p": r["p"], "Summe Lager i.i.d. (± SE)": f"{F.fmt_num(r['iid'], 0)} ± {F.fmt_num(r['iid_se'], 1)}", "Summe Lager mit Niveausprüngen (± SE)": f"{F.fmt_num(r['shift'], 0)} ± {F.fmt_num(r['shift_se'], 1)}",
                          "Verstärkung Stufe 4 i.i.d. / Sprünge": f"{F.fmt_num(r['amp4_iid'], 1)} / {F.fmt_num(r['amp4_shift'], 1)}",
                          "β Stufe 1 bei festem z i.i.d. / Sprünge (%)": f"{F.fmt_num(100 * r['beta1_fixed_iid'], 1)} / {F.fmt_num(100 * r['beta1_fixed_shift'], 1)}"} for r in rows])


def cell_coverage_frame(cov):
    """Abdeckung der Regler durch gemessene Zellen (Abstand in Parametern zur nächstliegenden Zelle)."""
    return pd.DataFrame([{"Abstand zur nächstliegenden gemessenen Zelle": "genau gemessen" if d == 0 else f"{d} Parameter abweichend", "Kombinationen aus CV, ρ, L, p": n} for d, n in cov["by_distance"].items()])
