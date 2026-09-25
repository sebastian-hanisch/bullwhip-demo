"""Darstellung der Tabellen und Kennzahlen: genaue Zeichenketten (Stellenzahl, Vorzeichen, Trenner) für einen festen Verlauf (eingefrorene Nachfragereihe des Preset-Seeds) und die Messreihe. Diese Tests
legen die FORMATE fest (eine Nachkommastelle bei Prozent, zwei bei Verstärkungen und z, ganze Einheiten bei Lager, Vorzeichen nur bei Änderungen); die Zahlen selbst prüfen test_ui_panel.py und
test_claims.py."""
import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_results as R
import bw_ui_panel as UI

DATA = R.load_results()
LIVE = LV.solve_live(0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL, C.SEED_DEFAULT, demand=FZ.preset_series(0.2, 0.0))
PAIR = R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)


def records(df, rows):
    return [df.iloc[i].to_dict() for i in rows]


def test_measures_frame_formats():
    assert records(UI.measures_frame(LIVE), [0, 1]) == [
        {"Maßnahme": "Basis", "z je Stufe": "1,07 / 1,33 / 1,66 / 1,84", "Herkunft z": "vorgerechnet", "z gehört zur Konfiguration": "CV 0,2, ρ 0,0, L 2, p 10", "Verstärkung Stufe 4": "10,4",
         "Summe Lager": "284", "Änderung (%)": "Bezug", "Servicegrad je Stufe (%)": "95,7 / 96,0 / 96,4 / 96,1", "Kettenkosten": "426"},
        {"Maßnahme": "alle glätten (γ 0,5)", "z je Stufe": "1,25 / 2,27 / 3,44 / 3,36", "Herkunft z": "vorgerechnet", "z gehört zur Konfiguration": "CV 0,2, ρ 0,0, L 2, p 10, γ 0,5",
         "Verstärkung Stufe 4": "2,9", "Summe Lager": "239", "Änderung (%)": "-15,8", "Servicegrad je Stufe (%)": "95,3 / 94,8 / 94,6 / 94,5", "Kettenkosten": "423"}]


def test_payer_live_frame_formats():
    assert records(UI.payer_live_frame(LIVE, C.M_RETAIL), [0, 3]) == [
        {"Stufe": "1 (Einzelhandel)", "Lager Basis": "32", "Lager nur Einzelhandel glättet": "37", "Änderung (%)": "+18,0", "Rückstand Basis": "4,2", "Rückstand Maßnahme": "4,2"},
        {"Stufe": "4 (Fabrik)", "Lager Basis": "126", "Lager nur Einzelhandel glättet": "94", "Änderung (%)": "-25,2", "Rückstand Basis": "3,9", "Rückstand Maßnahme": "4,1"}]


def test_amp_frame_formats():
    assert records(UI.amp_frame(LIVE, C.M_SMOOTH), [0, 3]) == [
        {"Stufe": "1 (Einzelhandel)", "Basis (dieser Verlauf)": "1,53", "alle glätten (dieser Verlauf)": "0,65", "Kontrolle: Simulation z = 0": "1,49", "exakte Theorie (Faltung)": "1,48",
         "Produktformel (Untergrenze)": "1,48", "Bestellung = 0 (Basis)": "0 %"},
        {"Stufe": "4 (Fabrik)", "Basis (dieser Verlauf)": "10,37", "alle glätten (dieser Verlauf)": "2,92", "Kontrolle: Simulation z = 0": "6,35", "exakte Theorie (Faltung)": "6,33",
         "Produktformel (Untergrenze)": "4,80", "Bestellung = 0 (Basis)": "6 %"}]


def test_comparison_table_formats():
    assert records(UI.comparison_table(LIVE, C.M_SMOOTH, PAIR, DATA), range(5)) == [
        {"Kennzahl": "Summe Lager Basis", "Dieser Verlauf (Seed 202, 1 Verlauf)": "284", "Messreihe (500 Wiederholungen, Mittel ± SE)": "270 ± 0,3"},
        {"Kennzahl": "Summe Lager alle glätten", "Dieser Verlauf (Seed 202, 1 Verlauf)": "239", "Messreihe (500 Wiederholungen, Mittel ± SE)": "228 ± 0,4"},
        {"Kennzahl": "Änderung der Summe Lager (%)", "Dieser Verlauf (Seed 202, 1 Verlauf)": "-15,8", "Messreihe (500 Wiederholungen, Mittel ± SE)": "-15,7 ± 0,2"},
        {"Kennzahl": "Verstärkung Stufe 4, Basis", "Dieser Verlauf (Seed 202, 1 Verlauf)": "10,4", "Messreihe (500 Wiederholungen, Mittel ± SE)": "11,1 ± 0,0"},
        {"Kennzahl": "Verstärkung Stufe 4, alle glätten", "Dieser Verlauf (Seed 202, 1 Verlauf)": "2,9", "Messreihe (500 Wiederholungen, Mittel ± SE)": "2,9 ± 0,0"}]
    assert UI.comparison_table(LIVE, C.M_SMOOTH, PAIR, DATA).shape == (5, 3)
    other = UI.comparison_table(LIVE, C.M_INFO, R.resolve_pair(DATA, C.M_INFO, 0.5, 0.2, 0.0, 2, 10), DATA)
    assert list(other["Kennzahl"])[1] == "Summe Lager Info teilen" and list(other["Messreihe (500 Wiederholungen, Mittel ± SE)"])[1].startswith("197 ± ")


def test_main_frame_formats():
    assert records(UI.main_frame(R.main_rows(DATA)), [0, 5, 12]) == [
        {"Maßnahme": "Basis", "z je Stufe": "1,07 / 1,33 / 1,66 / 1,84", "Verstärkung Stufen 1 bis 4": "1,51 / 2,62 / 5,26 / 11,06", "Summe Lager (Mittel ± SE)": "270 ± 0,3", "Änderung (%)": "Bezug",
         "Servicegrad je Stufe (%)": "95,0 / 95,0 / 95,1 / 95,2", "Kettenkosten": "451"},
        {"Maßnahme": "nur Einzelhandel glättet, γ 0,3", "z je Stufe": "1,46 / 2,05 / 2,48 / 2,21", "Verstärkung Stufen 1 bis 4": "0,40 / 0,71 / 1,59 / 3,95", "Summe Lager (Mittel ± SE)": "187 ± 0,3",
         "Änderung (%)": "-30,8 ± 0,1", "Servicegrad je Stufe (%)": "95,1 / 95,0 / 95,1 / 95,1", "Kettenkosten": "366"},
        {"Maßnahme": "L halbiert + Info teilen", "z je Stufe": "0,84 / 1,04 / 1,23 / 1,00", "Verstärkung Stufen 1 bis 4": "1,23 / 1,52 / 1,89 / 2,32", "Summe Lager (Mittel ± SE)": "85 ± 0,1",
         "Änderung (%)": "-68,7 ± 0,0", "Servicegrad je Stufe (%)": "95,0 / 95,1 / 95,1 / 95,1", "Kettenkosten": "263"}]


def test_fixed_frame_formats():
    assert records(UI.fixed_frame(R.fixed_rows(DATA)), [0, 2]) == [
        {"Maßnahme": "Basis", "Summe Lager fest z → gleicher Servicegrad": "293 → 270", "Servicegrad je Stufe bei festem z (%)": "97,6 / 95,7 / 94,0 / 93,5",
         "Rückstand (Summe) fest z → gleicher Servicegrad": "19,9 → 20,1", "Kettenkosten fest z → gleicher Servicegrad": "472 → 451"},
        {"Maßnahme": "Glättung γ 0,5 (alle)", "Summe Lager fest z → gleicher Servicegrad": "152 → 228", "Servicegrad je Stufe bei festem z (%)": "92,3 / 85,2 / 83,3 / 87,8",
         "Rückstand (Summe) fest z → gleicher Servicegrad": "53,6 → 19,9", "Kettenkosten fest z → gleicher Servicegrad": "634 → 407"}]


def test_payer_frame_formats():
    assert records(UI.payer_frame(R.payer_rows(DATA, 0.3)), [0, 3]) == [
        {"Stufe": "1 (Einzelhandel)", "Lager Basis": "30", "alle glätten (γ 0,3)": "38 (+29,0 %)", "nur Einzelhandel glättet (γ 0,3)": "40 (+36,5 %)"},
        {"Stufe": "4 (Fabrik)", "Lager Basis": "122", "alle glätten (γ 0,3)": "77 (-37,3 %)", "nur Einzelhandel glättet (γ 0,3)": "70 (-42,3 %)"}]


def test_regime_frame_formats():
    assert records(UI.regime_frame(R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5), "rho0.8"), [0, 2]) == [
        {"Gruppe": "Hauptszenario", "Zelle": "Hauptszenario", "Summe Lager Basis": "270", "Glättung γ 0,5 (%)": "-15,7 ± 0,2", "Urteil Glättung": "Glättung senkt hier den Kettenbestand",
         "Info teilen (%)": "-26,9 ± 0,1", "Urteil Info teilen": "Info teilen senkt hier den Kettenbestand"},
        {"Gruppe": "Autokorrelation", "Zelle": "▶ ρ 0,8", "Summe Lager Basis": "555", "Glättung γ 0,5 (%)": "+6,9 ± 0,4", "Urteil Glättung": "Glättung kostet hier",
         "Info teilen (%)": "-36,0 ± 0,2", "Urteil Info teilen": "Info teilen senkt hier den Kettenbestand"}]
    assert list(UI.regime_frame(R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5))["Zelle"])[2] == "ρ 0,8"


def test_proxy_frame_formats():
    assert records(UI.proxy_frame(R.proxy_rows(DATA)), [0, 7]) == [
        {"Maßnahme": "Basis", "Verstärkung Stufe 4": "11,06", "Rang (kleinste Verstärkung = 1)": 13, "Summe Lager": "270", "Rang (kleinstes Lager = 1)": 12},
        {"Maßnahme": "nur Fabrik glättet, γ 0,3", "Verstärkung Stufe 4": "3,13", "Rang (kleinste Verstärkung = 1)": 8, "Summe Lager": "275", "Rang (kleinstes Lager = 1)": 13}]


def test_formula_frame_formats():
    assert records(UI.formula_frame(R.formula_rows(DATA)), [0, 5]) == [
        {"m (= L)": 1, "p": 3, "m/p": "0,33", "Simulation Stufe 4 (± SE)": "21,51 ± 0,04", "exakte Theorie Stufe 4": "21,42", "Produktformel Stufe 4": "12,73", "exakt / Produkt": "1,68"},
        {"m (= L)": 2, "p": 5, "m/p": "0,40", "Simulation Stufe 4 (± SE)": "37,85 ± 0,09", "exakte Theorie Stufe 4": "37,70", "Produktformel Stufe 4": "20,20", "exakt / Produkt": "1,87"}]


def test_window_jump_frame_formats():
    assert records(UI.window_jump_frame(R.window_jump_rows(DATA)), [0, 3]) == [
        {"Fenster p": 4, "Summe Lager i.i.d. (± SE)": "1807 ± 4,9", "Summe Lager mit Niveausprüngen (± SE)": "1844 ± 5,5", "Verstärkung Stufe 4 i.i.d. / Sprünge": "342,1 / 138,3",
         "β Stufe 1 bei festem z i.i.d. / Sprünge (%)": "93,9 / 93,6"},
        {"Fenster p": 32, "Summe Lager i.i.d. (± SE)": "123 ± 0,1", "Summe Lager mit Niveausprüngen (± SE)": "168 ± 0,2", "Verstärkung Stufe 4 i.i.d. / Sprünge": "1,8 / 1,4",
         "β Stufe 1 bei festem z i.i.d. / Sprünge (%)": "99,0 / 97,7"}]


def test_fixed_z_text_formats():
    assert UI.fixed_z_text(PAIR) == ("Bei **festem z = 1,645** (Servicegrad nicht festgehalten) sähe dieselbe Maßnahme in der Messreihe so aus: Summe Lager -48,3 ± 0,1 % gegenüber der Basis, "
                                     "aber mit einem anderen Servicegrad als die Basis (siehe Abschnitt „Fester z gegen gleicher Servicegrad“): ein Vergleich ohne festgehaltenen Servicegrad täuscht.")


class FakeColumn:
    def __init__(self):
        self.calls = []

    def metric(self, label, value, **kw):
        self.calls.append((label, value, kw.get("delta")))


def test_metric_card_formats():
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, LIVE, C.M_SMOOTH, PAIR)
    assert [c.calls[0] for c in cols] == [("Verstärkung Stufe 4 (alle glätten)", "2,9", "-72 % gegenüber Basis (10,4)"), ("Summe Lager (Basis)", "284", "Messreihe: 270"),
                                          ("Summe Lager (alle glätten)", "239", "-45 (-15,8 %)"), ("Servicegrad, kleinste Stufe", "94,5 %", "95,3 / 94,8 / 94,6 / 94,5")]


def test_service_warning_boundary_is_strictly_above_the_gap(monkeypatch):
    import copy
    lv = copy.deepcopy(LIVE)
    lv["measures"][C.M_BASIS]["z_source"] = "vorgerechnet (nächstliegende Konfiguration)"
    lv["measures"][C.M_BASIS]["beta"] = [0.95, 0.95, 0.95, 0.70]                                # 0,95 - 0,70 = 0,25 genau
    monkeypatch.setattr(C, "SERVICE_WARN_GAP", 0.25)
    assert UI.service_warning(lv, C.M_SMOOTH) == ""
    monkeypatch.setattr(C, "SERVICE_WARN_GAP", 0.24)
    assert UI.service_warning(lv, C.M_SMOOTH).startswith("Achtung")


def test_metric_and_comparison_show_a_plus_sign_when_smoothing_makes_things_worse():
    import copy
    worse = copy.deepcopy(LIVE)
    worse["measures"][C.M_SMOOTH]["amp"][3] = 2 * worse["measures"][C.M_BASIS]["amp"][3]
    cols = [FakeColumn() for _ in range(4)]
    UI.render_metrics(cols, worse, C.M_SMOOTH, PAIR)
    assert cols[0].calls[0][2] == "+100 % gegenüber Basis (10,4)"
    slow = LV.solve_live(0.2, 0.8, 2, 10, 0.5, C.ZMODE_EQUAL, C.SEED_DEFAULT, demand=FZ.preset_series(0.2, 0.8))
    pair = R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.8, 2, 10)
    df = UI.comparison_table(slow, C.M_SMOOTH, pair, DATA)
    assert list(df.iloc[2]) == ["Änderung der Summe Lager (%)", "+14,5", "+6,9 ± 0,4"]
