"""Panel (bw_ui_panel): Kennzahlen, Meldung in drei Zuständen, Vergleichstabelle, Tabellen der Live-Ansicht und der Kernabschnitte. Die Live-Rechnung läuft auf der EINGEFRORENEN
Nachfragereihe des Preset-Verlaufs (die CI installiert das neueste NumPy); die Zahlen der Messreihe kommen aus data/bw_results.json."""
import copy

import pytest
import streamlit as st

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_results as R
import bw_ui_panel as UI

DATA = R.load_results()
SEED = C.SEED_DEFAULT


def live(**over):
    p = dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=C.ZMODE_EQUAL, seed=SEED)
    p.update(over)
    d = FZ.preset_series(p["cv"], p["rho"]) if (p["cv"], p["rho"]) in ((0.2, 0.0), (0.2, 0.8)) else None
    return LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], p["seed"], demand=d)


LIVE = live()


def pair_for(measure=C.M_SMOOTH, gamma=0.5, cv=0.2, rho=0.0, lead=2, window=10, data=DATA):
    return R.resolve_pair(data, measure, gamma, cv, rho, lead, window)


class FakeColumn:
    def __init__(self):
        self.calls = []

    def metric(self, label, value, **kw):
        self.calls.append((label, value, kw))


def test_measure_label_and_what_text():
    assert UI.measure_label(C.M_SMOOTH, 0.5) == "alle glätten (γ 0,5)" and UI.measure_label(C.M_INFO, 0.5) == "Info teilen" and UI.measure_label(C.M_SMOOTH) == "alle glätten"
    assert UI.measure_label(C.M_RETAIL, 0.3) == "nur Einzelhandel glättet (γ 0,3)" and UI.measure_label(C.M_LEAD, 0.5) == "L halbiert"
    for m in C.MEASURES:
        assert isinstance(UI.what_text(m, 0.5), str) and len(UI.what_text(m, 0.5)) > 5
    assert "γ 0,3" in UI.what_text(C.M_SMOOTH, 0.3) and "Einzelhandel" in UI.what_text(C.M_RETAIL, 0.5) and "γ 0,7" in UI.what_text(C.M_INFO_SMOOTH, 0.7)


def test_live_changes_compare_the_measure_with_the_basis():
    x = UI.live_changes(LIVE, C.M_SMOOTH)
    b, m = LIVE["measures"][C.M_BASIS], LIVE["measures"][C.M_SMOOTH]
    assert x["total_base"] == b["total_onhand"] and x["total"] == m["total_onhand"] and x["total_diff"] == pytest.approx(m["total_onhand"] - b["total_onhand"])
    assert x["total_change"] == pytest.approx(100 * (m["total_onhand"] / b["total_onhand"] - 1)) and x["amp"] == m["amp"][3] and x["amp_base"] == b["amp"][3]
    assert x["amp_change"] == pytest.approx(100 * (m["amp"][3] / b["amp"][3] - 1)) and x["beta"] == m["beta"] and x["z"] == m["z"] and x["beta_base"] == b["beta"]
    assert x["total_change"] < 0 and x["amp_change"] < -50


def test_render_metrics_fills_four_cards_with_the_documented_labels_and_deltas():
    cols = [FakeColumn() for _ in range(4)]
    x = UI.render_metrics(cols, LIVE, C.M_SMOOTH, pair_for())
    labels = [c.calls[0][0] for c in cols]
    assert labels == ["Verstärkung Stufe 4 (alle glätten)", "Summe Lager (Basis)", "Summe Lager (alle glätten)", "Servicegrad, kleinste Stufe"]
    values = [c.calls[0][1] for c in cols]
    assert values[0] == UI.F.fmt_num(x["amp"], 1) and values[1] == UI.F.fmt_num(x["total_base"], 0) and values[2] == UI.F.fmt_num(x["total"], 0) and values[3] == UI.F.fmt_num(100 * min(x["beta"]), 1) + " %"
    kw = [c.calls[0][2] for c in cols]
    assert kw[2]["delta_color"] == "inverse" and kw[2]["delta"].startswith("-") and "%)" in kw[2]["delta"]                                # Delta = Maßnahme minus Basis, negativ = billiger
    assert kw[0]["delta_color"] == "off" and kw[0]["delta"].endswith("% gegenüber Basis (" + UI.F.fmt_num(x["amp_base"], 1) + ")") and kw[1]["delta"].startswith("Messreihe: 270")
    assert kw[3]["delta"] == " / ".join(UI.F.fmt_num(100 * b, 1) for b in x["beta"]) and "Ziel 95 % je Stufe." in kw[3]["help"] and "z-Herkunft" not in kw[3]["help"]
    assert all(len(k["help"]) > 40 for k in kw)
    fixed = live(zmode=C.ZMODE_FIXED)
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, fixed, C.M_SMOOTH, pair_for())
    assert "Festes z = 1,645: der Servicegrad wird nicht festgehalten." in cols[3].calls[0][2]["help"] and "Ziel 95 %" not in cols[3].calls[0][2]["help"]


def test_delta_sign_of_the_measure_card_follows_measure_minus_basis_for_a_costly_measure():
    """Vorzeichen-Falle: Delta immer 'Maßnahme minus Basis'. Bei ρ = 0,8 kostet Glättung Bestand: Delta positiv."""
    lv = live(rho=0.8)
    x = UI.live_changes(lv, C.M_SMOOTH)
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, lv, C.M_SMOOTH, pair_for(rho=0.8, cv=0.2))
    delta = cols[2].calls[0][2]["delta"]
    assert (delta.startswith("+")) == (x["total_diff"] > 0)


def test_message_states_follow_the_measured_change():
    for kwargs, state in ((dict(), C.STATE_LOWER), (dict(rho=0.8), C.STATE_COSTS), (dict(window=40), C.STATE_LITTLE)):
        lv = live(**kwargs)
        pair = pair_for(**{k: v for k, v in kwargs.items()})
        got, text = UI.message(lv, C.M_SMOOTH, pair)
        assert got == state and text.startswith({C.STATE_LOWER: "✅ ", C.STATE_LITTLE: "ℹ️ ", C.STATE_COSTS: "⚠️ "}[state]) and R.state_text(C.M_SMOOTH, state) in text
        assert "500 Wiederholungen" in text and "β = 95 % je Stufe" in text and pair["cell_text"] in text and "Auf diesem einen Verlauf:" in text


def test_message_texts_carry_the_measured_numbers():
    _, text = UI.message(LIVE, C.M_SMOOTH, pair_for())
    assert "Summe Lager 270 → 228 Einheiten (-15,7 ± 0,2 %)" in text and "Glättung an allen Stufen (γ 0,5)" in text
    _, t2 = UI.message(live(rho=0.8), C.M_SMOOTH, pair_for(rho=0.8))
    assert "+6,9 ± 0,4 %" in t2 and "555 → 593" in t2
    _, t3 = UI.message(LIVE, C.M_LEAD, pair_for(C.M_LEAD))
    assert "-65,1" in t3 and R.state_text(C.M_LEAD, C.STATE_LOWER) in t3


def test_message_says_it_when_the_cell_or_the_smoothing_level_is_not_exact():
    lv = live(rho=0.8, gamma=0.7)
    pair = pair_for(gamma=0.7, rho=0.8)
    _, text = UI.message(lv, C.M_SMOOTH, pair)
    assert pair["note"] and pair["note"] in text and "γ 0,5" in text
    pair2 = pair_for(cv=0.3, rho=0.5, lead=4, window=10)
    _, text2 = UI.message(lv, C.M_SMOOTH, pair2)
    assert "keine eigene Messreihe" in text2


def test_message_for_each_of_the_three_states_from_modified_data():
    for factor, state in ((0.9, C.STATE_LOWER), (0.99, C.STATE_LITTLE), (1.05, C.STATE_COSTS)):
        d = copy.deepcopy(DATA)
        d["jobs"]["main/glaettung_0.5"]["equal"]["total_onhand"] = d["jobs"]["main/basis"]["equal"]["total_onhand"] * factor
        pair = pair_for(data=d)
        got, text = UI.message(LIVE, C.M_SMOOTH, pair)
        assert got == state and R.state_text(C.M_SMOOTH, state) in text


def test_render_message_uses_success_info_and_warning(monkeypatch):
    calls = []
    for name in ("success", "info", "warning"):
        monkeypatch.setattr(st, name, lambda text, n=name: calls.append((n, text)))
    for kwargs, state, fn in ((dict(), C.STATE_LOWER, "success"), (dict(window=40), C.STATE_LITTLE, "info"), (dict(rho=0.8), C.STATE_COSTS, "warning")):
        calls.clear()
        lv = live(**kwargs)
        assert UI.render_message(lv, C.M_SMOOTH, pair_for(**kwargs)) == state
        assert [c[0] for c in calls] == [fn]


def test_fixed_z_text_names_the_misleading_numbers():
    t = UI.fixed_z_text(pair_for())
    assert "festem z = 1,645" in t and "täuscht" in t and "%" in t and UI.fixed_z_text({}) == ""


def test_comparison_table_lines_up_the_run_and_the_measurement_series():
    df = UI.comparison_table(LIVE, C.M_SMOOTH, pair_for(), DATA)
    assert list(df.columns) == ["Kennzahl", f"Dieser Verlauf (Seed {SEED}, 1 Verlauf)", "Messreihe (500 Wiederholungen, Mittel ± SE)"] and len(df) == 5
    ms = list(df["Messreihe (500 Wiederholungen, Mittel ± SE)"])
    assert ms[0].startswith("270 ± 0,3") and ms[1].startswith("228 ± 0,") and ms[2].startswith("-15,7 ±") and ms[3].startswith("11,1 ±") and ms[4].startswith("2,9 ±")
    assert list(df["Kennzahl"])[2] == "Änderung der Summe Lager (%)"


def test_measures_frame_lists_every_active_measure_with_the_origin_of_z():
    df = UI.measures_frame(LIVE)
    assert list(df["Maßnahme"]) == ["Basis", "alle glätten (γ 0,5)", "nur Einzelhandel glättet (γ 0,5)", "Info teilen", "Info teilen + Glättung (γ 0,5)", "L halbiert", "Fenster verdoppelt"]
    assert df["Änderung (%)"][0] == "Bezug" and all(v.startswith("-") for v in df["Änderung (%)"][1:]) and set(df["Herkunft z"]) == {"vorgerechnet"}
    assert all(len(z.split(" / ")) == 4 for z in df["z je Stufe"]) and all(len(b.split(" / ")) == 4 for b in df["Servicegrad je Stufe (%)"])
    assert len(UI.measures_frame(live(lead=1, window=40))) == 5
    assert list(df["z gehört zur Konfiguration"]) == ["CV 0,2, ρ 0,0, L 2, p 10", "CV 0,2, ρ 0,0, L 2, p 10, γ 0,5", "CV 0,2, ρ 0,0, L 2, p 10, γ 0,5", "CV 0,2, ρ 0,0, L 2, p 10", "CV 0,2, ρ 0,0, L 2, p 10, γ 0,5",
                                                  "CV 0,2, ρ 0,0, L 1, p 10", "CV 0,2, ρ 0,0, L 2, p 20"]
    far = UI.measures_frame(live(rho=0.8, lead=4))
    assert far["Herkunft z"][0] == "vorgerechnet (nächstliegende Konfiguration)" and far["z gehört zur Konfiguration"][0] == "CV 0,2, ρ 0,8, L 2, p 10" and far["Herkunft z"][5] == "vorgerechnet"


def test_payer_live_frame_shows_who_pays_on_this_run():
    df = UI.payer_live_frame(LIVE, C.M_RETAIL)
    assert len(df) == 4 and list(df.columns)[:3] == ["Stufe", "Lager Basis", "Lager nur Einzelhandel glättet"] and df["Änderung (%)"][0].startswith("+") and df["Änderung (%)"][3].startswith("-")
    assert df["Stufe"][0] == "1 (Einzelhandel)" and df["Stufe"][3] == "4 (Fabrik)"


def test_amp_frame_has_the_control_row_columns_and_the_theory():
    df = UI.amp_frame(LIVE, C.M_SMOOTH)
    assert list(df.columns) == ["Stufe", "Basis (dieser Verlauf)", "alle glätten (dieser Verlauf)", "Kontrolle: Simulation z = 0", "exakte Theorie (Faltung)", "Produktformel (Untergrenze)", "Bestellung = 0 (Basis)"]
    assert list(df["exakte Theorie (Faltung)"]) == ["1,48", "2,31", "3,75", "6,33"] and list(df["Produktformel (Untergrenze)"]) == ["1,48", "2,19", "3,24", "4,80"]
    assert "1,5e" not in UI.control_text(LIVE) and "e-1" in UI.control_text(LIVE) and "ab Periode 41" in UI.control_text(LIVE) and "Untergrenze" in UI.control_text(LIVE)


def test_main_frame_matches_the_measurement_series():
    df = UI.main_frame(R.main_rows(DATA))
    assert len(df) == 13 and df["Maßnahme"][0] == "Basis" and df["Änderung (%)"][0] == "Bezug" and df["z je Stufe"][0] == "1,07 / 1,33 / 1,66 / 1,84" and df["Summe Lager (Mittel ± SE)"][0].startswith("270 ± 0,")
    row = df[df["Maßnahme"] == "Wiederbeschaffungszeit halbiert"].iloc[0]
    assert row["Änderung (%)"].startswith("-65,1 ±") and row["Verstärkung Stufen 1 bis 4"].split(" / ")[3] == "3,06"


def test_fixed_frame_shows_both_modes():
    df = UI.fixed_frame(R.fixed_rows(DATA))
    row = df[df["Maßnahme"] == "Glättung γ 0,5 (alle)"].iloc[0]
    assert row["Summe Lager fest z → gleicher Servicegrad"].startswith("152 →") and row["Servicegrad je Stufe bei festem z (%)"].startswith("92,3 / 85,2")


def test_payer_and_regime_and_proxy_and_formula_and_window_frames():
    p = UI.payer_frame(R.payer_rows(DATA, 0.3))
    assert list(p.columns)[1:] == ["Lager Basis", "alle glätten (γ 0,3)", "nur Einzelhandel glättet (γ 0,3)"] and "(+29,0 %)" in p.iloc[0, 2] and "(-42,3 %)" in p.iloc[3, 3]
    rg = UI.regime_frame(R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5), "rho0.8")
    assert len(rg) == 10 and rg["Zelle"][2].startswith("▶ ρ 0,8") and rg["Urteil Glättung"][2] == "Glättung kostet hier" and rg["Urteil Info teilen"][9] == "Info teilen senkt hier den Kettenbestand"
    px = UI.proxy_frame(R.proxy_rows(DATA))
    assert len(px) == 13 and set(px["Rang (kleinste Verstärkung = 1)"]) == set(range(1, 14)) and set(px["Rang (kleinstes Lager = 1)"]) == set(range(1, 14))
    fm = UI.formula_frame(R.formula_rows(DATA))
    assert len(fm) == 12 and fm[(fm["m (= L)"] == 4) & (fm["p"] == 3)]["exakt / Produkt"].iloc[0] == "3,43" and fm[(fm["m (= L)"] == 1) & (fm["p"] == 20)]["exakt / Produkt"].iloc[0] == "1,03"
    wj = UI.window_jump_frame(R.window_jump_rows(DATA))
    assert list(wj["Fenster p"]) == [4, 8, 16, 32, 64] and wj["Summe Lager mit Niveausprüngen (± SE)"][3].startswith("168 ±")
    cov = UI.cell_coverage_frame(R.coverage(DATA))
    assert list(cov["Kombinationen aus CV, ρ, L, p"]) == [10, 30, 44, 24] and cov.iloc[0, 0] == "genau gemessen" and cov.iloc[3, 0] == "3 Parameter abweichend"


def test_chart_table_height_scales_with_the_chart():
    assert UI.CHART_TABLE_HEIGHT == C.CHART_HEIGHT + 60 and UI.STAGE_LABELS == ["1 (Einzelhandel)", "2 (Großhandel)", "3 (Distributor)", "4 (Fabrik)"]


def test_z_hint_names_the_origin_of_z():
    assert UI.z_hint(LIVE["measures"][C.M_BASIS]) == ""
    far = live(rho=0.8, lead=4)
    assert UI.z_hint(far["measures"][C.M_BASIS]) == "z aus nächstliegender Konfiguration" and UI.z_hint(far["measures"][C.M_LEAD]) == ""              # L halbiert bei L = 4 ist die Konfiguration L = 2
    calibrated = LV.solve_live(0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL, SEED, z_override={C.M_BASIS: [1.0] * 4}, demand=FZ.preset_series(0.2, 0.0))
    assert UI.z_hint(calibrated["measures"][C.M_BASIS]) == "z grob nachkalibriert" and UI.z_hint(calibrated["measures"][C.M_SMOOTH]) == ""


def test_the_service_card_says_when_z_is_not_exact():
    far = live(rho=0.8, lead=4)
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, far, C.M_SMOOTH, pair_for(rho=0.8, lead=4))
    assert "z-Herkunft: z aus nächstliegender Konfiguration." in cols[3].calls[0][2]["help"]
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, far, C.M_LEAD, pair_for(C.M_LEAD, rho=0.8, lead=4))
    assert "z-Herkunft" not in cols[3].calls[0][2]["help"]


def test_service_warning_only_when_z_is_borrowed_and_the_service_is_far_from_the_target():
    far = live(rho=0.8, lead=4)
    text = UI.service_warning(far, C.M_SMOOTH)
    assert text.startswith("Achtung: das z je Stufe stammt hier aus der nächstliegenden gemessenen Konfiguration") and "Basis 80,7 bis 92,8 %" in text and "alle glätten 83,5 bis 91,5 %" in text and "live nachkalibrieren" in text
    assert UI.service_warning(far, C.M_BASIS).count("Basis 80,7") == 1                                                              # die Basis als gezeigte Maßnahme nur einmal genannt
    assert UI.service_warning(LIVE, C.M_SMOOTH) == "" and UI.service_warning(live(zmode=C.ZMODE_FIXED, rho=0.8, lead=4), C.M_SMOOTH) == ""
    assert UI.service_warning(far, C.M_LEAD) != "" and "L halbiert" not in UI.service_warning(far, C.M_LEAD)                          # die Basis ist weit weg, L halbiert (exaktes z) nicht


def test_service_warning_threshold_is_five_points():
    lv = copy.deepcopy(LIVE)
    for m in (C.M_BASIS, C.M_SMOOTH):
        lv["measures"][m]["z_source"] = "vorgerechnet (nächstliegende Konfiguration)"
        lv["measures"][m]["z_exact"] = False
        lv["measures"][m]["beta"] = [0.95, 0.95, 0.95, 0.95]
    assert UI.service_warning(lv, C.M_SMOOTH) == ""
    lv["measures"][C.M_BASIS]["beta"] = [0.95, 0.95, 0.95, 0.9001]
    assert UI.service_warning(lv, C.M_SMOOTH) == ""
    lv["measures"][C.M_BASIS]["beta"] = [0.95, 0.95, 0.95, 0.8999]
    assert "Basis 90,0 bis 95,0 %" in UI.service_warning(lv, C.M_SMOOTH)
    lv["measures"][C.M_BASIS]["beta"] = [0.95, 0.95, 0.95, 1.0001]
    assert "Basis 95,0 bis 100,0 %" in UI.service_warning(lv, C.M_SMOOTH)                                                             # zu viel Service zählt ebenso
    lv["measures"][C.M_BASIS]["z_source"] = "live nachkalibriert (grob)"
    assert UI.service_warning(lv, C.M_SMOOTH) == ""
