"""AppTest: Skelett und Footer, jedes Preset, Permalink, alle Regler an Min und Max, alle drei Meldungszustände, keine wirkungslosen Regler, ausgeblendete Maßnahmen, Servicegrad-Modus, Kernabschnitt
②, Ansichten, Schnellkalibrierung (mit kleinem Ansatz, ohne Wall-Clock-Assertion), PDF, Texte. Die Live-Rechnung läuft auf der EINGEFRORENEN Nachfragereihe des Preset-Seeds (die CI installiert
das neueste NumPy: der Zufallsstrom von gen_ar1 darf sich ändern); für Einstellungen ohne eingefrorene Reihe prüfen die Tests nur Struktur und Richtung."""
import pathlib
import re
import time

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_presets as P
import bw_results as R
import bw_ui_panel as UI
from bw_presets import PRESET_STATE_KEYS, SETTING_SPECS

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
DATA = R.load_results()
FOOTER = ("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
          "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
          "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)")
SLIDERS = ("cv_slider", "rho_slider", "lead_slider", "window_slider", "gamma_slider")
FROZEN_KEYS = set(FZ.preset_demand()["series"])


@pytest.fixture(autouse=True)
def frozen_demand(monkeypatch):
    """st.cache_data ist prozessweit: Tests dürfen keine Ergebnisse anderer Tests sehen. Der Verlauf des Preset-Seeds kommt aus der eingefrorenen Reihe (unabhängig von der NumPy-Version)."""
    st.cache_data.clear()
    real = LV.gen_demand

    def fake(cv, rho, seed, periods=C.LIVE_T):
        if int(seed) == C.SEED_DEFAULT and f"{float(cv)}/{float(rho)}" in FROZEN_KEYS and periods == C.LIVE_T:
            return FZ.preset_series(float(cv), float(rho))
        return real(cv, rho, seed, periods)
    monkeypatch.setattr(LV, "gen_demand", fake)
    yield
    st.cache_data.clear()


def fresh(**query):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def set_and_run(at, **values):
    for key, value in values.items():
        if key == "seed_input":
            at.number_input(key=key).set_value(value)
        elif key == "z_mode":
            at.radio(key=key).set_value(value)
        else:
            at.select_slider(key=key).set_value(value)
    at.run()
    assert not at.exception, at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def main_metrics(at):
    return [(m.label, m.value) for m in at.metric[:4]]


def box(at, kind, needle):
    for x in getattr(at, kind):
        if needle in x.value:
            return x.value
    return None


def state_values(at):
    return {key: at.session_state[key] for key in SETTING_SPECS}


def live_for(**over):
    p = dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=C.ZMODE_EQUAL, seed=C.SEED_DEFAULT)
    p.update(over)
    d = FZ.preset_series(p["cv"], p["rho"]) if f"{p['cv']}/{p['rho']}" in FROZEN_KEYS else None
    return LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], p["seed"], demand=d)


def frames(at):
    return [d.value for d in at.dataframe]


# ---------------------------------------------------------------------------------------------------
# Skelett
# ---------------------------------------------------------------------------------------------------
def test_skeleton_and_footer():
    at = fresh()
    assert [h.value for h in at.sidebar.header] == ["⚙️ Einstellungen"]                   # genau EIN Header
    assert len(at.title) == 1 and at.title[0].value == "🐂 Bullwhip-Effekt: Wer zahlt für die Glättung?"
    assert any(v.value.startswith("## 🐂 Wie stark schaukelt sich die Kette hoch – und was kostet es?") for v in at.markdown)
    assert any(v.value.startswith("### 📐 Was die Messreihe bei gleichem Servicegrad zeigt") for v in at.markdown)
    assert [e.label for e in at.expander] == ["🔧 Wie wir das erreichen – Maßnahmen im Vergleich", "Wie funktioniert diese Demo?", "📐 Mathematische Formulierung"]
    assert any(c.value == FOOTER for c in at.caption)
    presets = [b.label for b in at.button if b.label in C.PRESETS]
    assert presets == list(C.PRESETS) == ["Standard", "Kurze Lieferzeit", "Träger Bedarf", "Kleines Fenster", "Großes Fenster"] and all(len(n) <= 32 for n in presets)
    assert [s.label for s in at.sidebar.select_slider] == ["Nachfrageschwankung (CV)", "Autokorrelation ρ", "Wiederbeschaffungszeit L", "Prognosefenster p", "Glättung γ"]
    assert [r.label for r in at.sidebar.radio] == ["Servicegrad"] and [n.label for n in at.sidebar.number_input] == ["Seed"] and not at.sidebar.slider
    assert [b.label for b in at.sidebar.button] == ["🎲 Neuer Verlauf"]                     # letztes Sidebar-Element
    assert [type(e).__name__ for e in at.sidebar.children.values()][-1] == "Button"
    assert [t.label for t in at.tabs] == ["📈 Verlauf", "🧮 Theorie", "💶 Kosten und Servicegrad", "📊 Messreihe"]


def test_the_preset_buttons_have_help_and_the_sidebar_has_named_groups_and_no_second_header():
    at = fresh()
    buttons = [b for b in at.button if b.label in C.PRESETS]
    assert all(b.help and len(b.help) > 20 for b in buttons) and [b.help for b in buttons] == [C.PRESET_HELP[n] for n in C.PRESETS]
    assert len(at.sidebar.header) == 1 and len(at.sidebar.subheader) == 0
    assert [m.value for m in at.sidebar.markdown] == ["**Die Kette**", "**Die Maßnahme**", "**Der gezeigte Verlauf**"]


def test_intro_names_the_expanders_and_neighbours():
    at = fresh()
    intro = next(m.value for m in at.markdown if m.value.strip().startswith("Der Kundenbedarf schwankt leicht"))
    for needle in ("Bestellungen weiter oben in der Kette immer stärker", "Bullwhip-Effekt", "Servicegrad festhält", "wer zahlt", "ein Verlauf über 300 Perioden", "500 Wiederholungen je Zelle",
                   "naive-forecast-demo", "exponential-smoothing-demo", "inventory-routing-demo", "Wie funktioniert diese Demo?", "📐 Mathematische Formulierung"):
        assert needle in intro, needle


def test_there_are_no_dead_controls():
    """Jeder Regler wirkt: es werden immer alle Maßnahmen gerechnet. Nur diese Widgets gibt es, und der einzige Knopf mit Rechnung ist die Schnellkalibrierung."""
    at = fresh()
    assert not at.sidebar.checkbox and not at.sidebar.toggle and not at.checkbox and not at.toggle and not at.slider
    assert [s.key for s in at.selectbox] == ["measure_select", "formula_select", "map_rho"] and sorted(r.key for r in at.radio) == ["stage_select", "view_select", "z_mode"]
    assert sorted(b.label for b in at.button if b.label not in C.PRESETS) == sorted(["🎲 Neuer Verlauf", "Servicegrad live nachkalibrieren (grob, ca. 2 s je Maßnahme)"])
    assert [b.key for b in at.button if b.label.startswith("Servicegrad live")] == ["quickcal_button"]


def test_display_choices_are_labelled_as_display_only():
    at = fresh()
    for w in (at.selectbox(key="measure_select"), at.radio(key="stage_select"), at.radio(key="view_select"), at.selectbox(key="formula_select"), at.selectbox(key="map_rho")):
        assert "Reine Anzeigewahl" in w.help, w.key


def test_sliders_have_the_measured_stages_and_the_seed_bounds():
    at = fresh()
    assert list(at.select_slider(key="cv_slider").options) == ["0,1", "0,2", "0,3"] and list(at.select_slider(key="rho_slider").options) == ["0,0", "0,5", "0,8"]
    assert list(at.select_slider(key="lead_slider").options) == ["1 Periode", "2 Perioden", "4 Perioden"] and list(at.select_slider(key="window_slider").options) == ["6 Perioden", "10 Perioden", "20 Perioden", "40 Perioden"]
    assert list(at.select_slider(key="gamma_slider").options) == ["0,7", "0,5", "0,3"] and list(at.radio(key="z_mode").options) == ["gleicher Servicegrad (kalibriert)", "festes z = 1,645"]
    seed = at.number_input(key="seed_input")
    assert (seed.min, seed.max, seed.step, seed.value) == (0, 299, 1, C.SEED_DEFAULT)


def test_default_state_and_permalink_written_to_the_address_bar():
    at = fresh()
    assert state_values(at) == dict(cv_slider=0.2, rho_slider=0.0, lead_slider=2, window_slider=10, gamma_slider=0.5, z_mode="kalibriert", seed_input=C.SEED_DEFAULT)
    qp = {k: (v[0] if isinstance(v, list) else v) for k, v in dict(at.query_params).items()}
    assert qp == {"cv": "0.2", "rho": "0", "l": "2", "p": "10", "g": "0.5", "z": "kalibriert", "seed": str(C.SEED_DEFAULT)}


def test_main_metrics_are_2x2_with_the_right_labels_and_values():
    at = fresh()
    assert [m[0] for m in main_metrics(at)] == ["Verstärkung Stufe 4 (alle glätten)", "Summe Lager (Basis)", "Summe Lager (alle glätten)", "Servicegrad, kleinste Stufe"]
    live = live_for()
    x = UI.live_changes(live, C.M_SMOOTH)
    fmt = UI.F.fmt_num
    vals = [m[1] for m in main_metrics(at)]
    assert vals[0] == fmt(x["amp"], 1) and vals[1] == fmt(x["total_base"], 0) and vals[2] == fmt(x["total"], 0) and vals[3] == fmt(100 * min(x["beta"]), 1) + " %"
    delta = {m.label: m.delta for m in at.metric[:4]}
    assert delta["Summe Lager (alle glätten)"].startswith("-") and "%)" in delta["Summe Lager (alle glätten)"] and delta["Summe Lager (Basis)"] == "Messreihe: 270" and len(at.columns) >= 4
    assert delta["Servicegrad, kleinste Stufe"] == " / ".join(fmt(100 * b, 1) for b in x["beta"])


def test_live_caption_says_it_is_one_run():
    at = fresh()
    cap = next(c.value for c in at.caption if c.value.startswith("Live-Verlauf (ein Verlauf, 300 Perioden nach 100 Perioden Einschwingen)"))
    assert f"Seed {C.SEED_DEFAULT}" in cap and "Ein einzelner Verlauf" in cap and "CV 0,2" in cap and "L 2" in cap and "p 10" in cap and "gleicher Servicegrad (kalibriert)" in cap and "0,3 s" in cap


# ---------------------------------------------------------------------------------------------------
# Presets, Permalink, Regler
# ---------------------------------------------------------------------------------------------------
PRESET_STATES = {"Standard": ("success", "Glättung senkt hier den Kettenbestand", "✅"), "Kurze Lieferzeit": ("info", "Glättung bringt hier wenig", "ℹ️"),
                 "Träger Bedarf": ("warning", "Glättung kostet hier", "⚠️"), "Kleines Fenster": ("success", "Glättung senkt hier den Kettenbestand", "✅"),
                 "Großes Fenster": ("info", "Glättung bringt hier wenig", "ℹ️")}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_sets_all_controls_and_shows_its_message(name):
    at = fresh()
    click(at, name)
    p = C.PRESETS[name]
    assert state_values(at) == {PRESET_STATE_KEYS[k]: v for k, v in p.items()}
    kind, needle, icon = PRESET_STATES[name]
    assert box(at, kind, needle) is not None and next(x for x in getattr(at, kind) if needle in x.value).icon == icon
    text = box(at, kind, needle)
    cell, exact, _ = R.nearest_cell(DATA, p["cv"], p["rho"], p["lead"], p["window"])
    assert exact and R.cell_setting_text(DATA, cell) in text
    assert dict(at.query_params).get("seed") in (str(C.SEED_DEFAULT), [str(C.SEED_DEFAULT)])
    live = live_for(**p)
    assert at.metric[1].value == UI.F.fmt_num(live["measures"][C.M_BASIS]["total_onhand"], 0)


def test_presets_change_the_run_shown_and_standard_restores_it():
    at = fresh()
    base_values = main_metrics(at)
    click(at, "Kleines Fenster")
    assert main_metrics(at) != base_values and at.session_state["window_slider"] == 6
    click(at, "Träger Bedarf")
    assert next(c.value for c in at.caption if c.value.startswith("Live-Verlauf")).count("ρ 0,8") == 1
    click(at, "Standard")
    assert main_metrics(at) == base_values


def test_permalink_sets_the_controls_and_snaps_to_stages():
    at = fresh(cv="0.3", rho="0.8", l="4", p="20", g="0.3", z="fest", seed="17")
    assert state_values(at) == dict(cv_slider=0.3, rho_slider=0.8, lead_slider=4, window_slider=20, gamma_slider=0.3, z_mode="fest", seed_input=17)
    at = fresh(cv="0.27", rho="0.6", l="3", p="12", g="0.9", z="egal", seed="-4")
    assert state_values(at) == dict(cv_slider=0.3, rho_slider=0.5, lead_slider=2, window_slider=10, gamma_slider=0.7, z_mode="kalibriert", seed_input=0)
    at = fresh(cv="abc", l="", seed="x")
    assert state_values(at)["cv_slider"] == 0.2 and state_values(at)["lead_slider"] == 2 and state_values(at)["seed_input"] == C.SEED_DEFAULT


def test_all_controls_at_min_and_max_run_without_exception():
    at = fresh()
    for key in SLIDERS:
        options = list(SETTING_SPECS[key].options)
        for v in (options[0], options[-1]):
            set_and_run(at, **{key: v})
            assert state_values(at)[key] == v
    for v in ("fest", "kalibriert"):
        set_and_run(at, z_mode=v)
        assert state_values(at)["z_mode"] == v
    for v in (0, 299):
        set_and_run(at, seed_input=v)
        assert state_values(at)["seed_input"] == v
    set_and_run(at, cv_slider=0.1, rho_slider=0.0, lead_slider=1, window_slider=6, gamma_slider=0.7, seed_input=0)
    assert at.metric[0].value
    set_and_run(at, cv_slider=0.3, rho_slider=0.8, lead_slider=4, window_slider=40, gamma_slider=0.3, z_mode="fest", seed_input=299)
    assert at.metric[0].value


def test_new_run_button_rolls_a_new_seed(monkeypatch):
    frozen = iter([3, 5, C.SEED_DEFAULT, 3])
    monkeypatch.setattr(P.random, "randint", lambda lo, hi: next(frozen))
    at = fresh()
    seen = {at.session_state["seed_input"]}
    for _ in range(3):
        click(at, "🎲 Neuer Verlauf")
        seen.add(at.session_state["seed_input"])
        assert 0 <= at.session_state["seed_input"] <= 299
    assert seen == {C.SEED_DEFAULT, 3, 5}


# ---------------------------------------------------------------------------------------------------
# Meldung in drei Zuständen, Zell-Zuordnung, keine wirkungslosen Regler
# ---------------------------------------------------------------------------------------------------
def test_three_message_states_follow_the_controls():
    at = fresh()
    assert box(at, "success", "Glättung senkt hier den Kettenbestand") and not box(at, "info", "Glättung bringt hier wenig") and not box(at, "warning", "Glättung kostet hier")
    set_and_run(at, rho_slider=0.8)
    assert box(at, "warning", "Glättung kostet hier") and not box(at, "success", "Glättung senkt hier")
    set_and_run(at, rho_slider=0.0, window_slider=40)
    assert box(at, "info", "Glättung bringt hier wenig") and not box(at, "warning", "Glättung kostet hier")
    set_and_run(at, window_slider=6)
    assert box(at, "success", "Glättung senkt hier den Kettenbestand")
    set_and_run(at, window_slider=10, lead_slider=1)
    assert box(at, "info", "Glättung bringt hier wenig")


def test_the_message_uses_the_measurement_and_not_the_run_shown():
    at = fresh()
    text = box(at, "success", "Glättung senkt hier den Kettenbestand")
    assert "Summe Lager 270 → 228 Einheiten (-15,7 ± 0,2 %)" in text and "Auf diesem einen Verlauf:" in text and "500 Wiederholungen" in text and "β = 95 % je Stufe" in text
    set_and_run(at, seed_input=5)
    assert "(-15,7 ± 0,2 %)" in box(at, "success", "Glättung senkt hier den Kettenbestand")                       # anderer Verlauf, dieselbe Messreihe


def test_every_control_changes_the_live_result_no_dead_controls_in_effect():
    at = fresh()
    base = main_metrics(at)
    for key, value in (("cv_slider", 0.3), ("rho_slider", 0.8), ("lead_slider", 4), ("window_slider", 20), ("gamma_slider", 0.3), ("z_mode", "fest"), ("seed_input", 5)):
        other = fresh()
        set_and_run(other, **{key: value})
        assert main_metrics(other) != base, key


def test_smoothing_only_changes_the_measures_it_belongs_to():
    """γ wirkt in 'alle glätten', 'nur Einzelhandel' und 'Info teilen + Glättung', nicht in Basis, Info teilen, L halbiert, Fenster verdoppelt."""
    a, b = live_for(gamma=0.5), live_for(gamma=0.3)
    for m in C.MEASURES:
        same = a["measures"][m]["total_onhand"] == b["measures"][m]["total_onhand"]
        assert same == (m not in C.SMOOTH_MEASURES), m


def test_the_measure_choice_switches_the_metrics_and_the_message_subject():
    at = fresh()
    assert list(at.selectbox(key="measure_select").options) == ["alle glätten (γ 0,5)", "nur Einzelhandel glättet (γ 0,5)", "Info teilen", "Info teilen + Glättung (γ 0,5)", "L halbiert", "Fenster verdoppelt"]
    for m, subject in ((C.M_INFO, "Info teilen senkt hier den Kettenbestand"), (C.M_LEAD, "Die halbierte Wiederbeschaffungszeit senkt hier den Kettenbestand"),
                       (C.M_WINDOW, "Das verdoppelte Fenster senkt hier den Kettenbestand"), (C.M_RETAIL, "Glättung nur im Einzelhandel senkt hier den Kettenbestand")):
        at.selectbox(key="measure_select").set_value(m).run()
        assert not at.exception and box(at, "success", subject), m
        assert at.metric[2].label == f"Summe Lager ({C.MEASURE_SHORT[m]})" and at.metric[1].label == "Summe Lager (Basis)"
    fmt = UI.F.fmt_num
    live = live_for()
    assert at.metric[2].value == fmt(live["measures"][C.M_RETAIL]["total_onhand"], 0)                                 # zuletzt gewählt: nur Einzelhandel


def test_hidden_measures_disappear_with_a_note_and_the_display_choice_resets():
    at = fresh()
    at.selectbox(key="measure_select").set_value(C.M_LEAD).run()
    set_and_run(at, lead_slider=1)                                                                   # "L halbiert" gewählt, dann entfällt es
    assert at.session_state["measure_select"] == C.MEASURE_DEFAULT and C.M_LEAD not in list(at.selectbox(key="measure_select").options)
    assert "L halbiert" not in " ".join(at.selectbox(key="measure_select").options) and any(c.value.startswith("Ausgeblendet bei dieser Einstellung: „L halbiert“") for c in at.caption)
    set_and_run(at, lead_slider=2)
    assert "L halbiert" in list(at.selectbox(key="measure_select").options) and not any(c.value.startswith("Ausgeblendet") for c in at.caption)
    at.selectbox(key="measure_select").set_value(C.M_WINDOW).run()
    set_and_run(at, window_slider=40)
    assert at.session_state["measure_select"] == C.MEASURE_DEFAULT and any("„Fenster verdoppelt“" in c.value and c.value.startswith("Ausgeblendet") for c in at.caption)
    set_and_run(at, lead_slider=1)
    cap = next(c.value for c in at.caption if c.value.startswith("Ausgeblendet"))
    assert "„L halbiert“" in cap and "„Fenster verdoppelt“" in cap
    assert len(list(at.selectbox(key="measure_select").options)) == 4                              # ohne Basis, L halbiert und Fenster verdoppelt


def test_unmeasured_combination_shows_the_nearest_cell_with_a_note():
    at = fresh()
    set_and_run(at, cv_slider=0.3, lead_slider=4)
    note = "keine eigene Messreihe: die Messreihe variiert je Zelle nur einen Parameter"
    assert any(note in i.value for i in at.info) and any(note in c.value for c in at.caption) and any("Nachfrageschwankung CV" in c.value or "abweichend" in c.value for c in at.caption)
    assert any("z je Stufe: bei" in c.value and "aus der nächstliegenden gemessenen Konfiguration" in c.value for c in at.caption)
    set_and_run(at, cv_slider=0.2, lead_slider=2)
    assert not any(note in i.value for i in at.info) and not any(note in c.value for c in at.caption)      # exakt gemessen: kein Hinweis
    assert any("z je Stufe: für alle Maßnahmen genau vorgerechnet" in c.value for c in at.caption)


def test_the_smoothing_level_that_is_not_measured_is_named_in_the_message():
    at = fresh(g="0.7", rho="0.8")
    text = box(at, "warning", "Glättung kostet hier")
    assert text is not None and "nur mit γ 0,5 gemessen" in text


# ---------------------------------------------------------------------------------------------------
# Servicegrad-Modus
# ---------------------------------------------------------------------------------------------------
def test_fixed_z_mode_warns_that_the_service_is_not_held_and_keeps_the_measured_message():
    at = fresh(z="fest")
    warn = box(at, "warning", "Modus „festes z = 1,645“")
    assert warn is not None and "nicht** festgehalten" in warn and "festem z = 1,645" in warn
    assert box(at, "success", "Glättung senkt hier den Kettenbestand") is not None                          # die Meldung stützt sich weiter auf die Messreihe bei gleichem Servicegrad
    assert "Festes z = 1,645: der Servicegrad wird nicht festgehalten." in at.metric[3].help and not [b for b in at.button if b.label.startswith("Servicegrad live")]
    equal = fresh()
    assert box(equal, "warning", "Modus „festes z = 1,645“") is None


def test_the_service_mode_changes_the_live_numbers_and_the_service_of_the_stages():
    equal, fixed = fresh(), fresh(z="fest")
    assert [m.value for m in equal.metric[:4]] != [m.value for m in fixed.metric[:4]]
    lf = live_for(zmode=C.ZMODE_FIXED)
    assert fixed.metric[3].delta == " / ".join(UI.F.fmt_num(100 * b, 1) for b in lf["measures"][C.M_SMOOTH]["beta"])


# ---------------------------------------------------------------------------------------------------
# Schnellkalibrierung
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def small_calibration(monkeypatch):
    """Die Schnellkalibrierung mit kleinem Ansatz (4 x 250) und Aufrufzähler: kein Wall-Clock-Test."""
    calls = []
    real = LV.quick_calibrate

    def small(*args, **kw):
        calls.append(args)
        kw.update(reps=4, periods=250)
        return real(*args, **kw)
    monkeypatch.setattr(LV, "quick_calibrate", small)
    return calls


CAL = "Servicegrad live nachkalibrieren (grob, ca. 2 s je Maßnahme)"


def test_quick_calibration_replaces_z_for_all_measures_and_is_marked_rough(small_calibration):
    at = fresh()
    click(at, CAL)
    assert len(small_calibration) == 1 and small_calibration[0][:6] == (0.2, 0.0, 2, 10, 0.5, C.SEED_DEFAULT) and list(small_calibration[0][6]) == list(C.MEASURES)
    cal = at.session_state["quick_cal"]
    assert cal["key"] == (0.2, 0.0, 2, 10, 0.5) and set(cal["z"]) == set(C.MEASURES) and cal["skipped"] == [] and all(len(z) == 4 for z in cal["z"].values())
    assert any(i.value.startswith("Grobe Live-Kalibrierung aktiv: z je Stufe für 7 Maßnahmen neu bestimmt (20 Wiederholungen × 700 Perioden, ein Durchlauf") and "grob" in i.value for i in at.info)
    tab = next(f for f in frames(at) if "Herkunft z" in f.columns)
    assert set(tab["Herkunft z"]) == {"live nachkalibriert (grob)"}
    assert [b for b in at.button if b.label == "Vorgerechnetes z wieder verwenden"]


def test_quick_calibration_z_reaches_the_live_result(small_calibration):
    at = fresh()
    before = [f for f in frames(at) if "Herkunft z" in f.columns][0]["z je Stufe"].tolist()
    click(at, CAL)
    after = [f for f in frames(at) if "Herkunft z" in f.columns][0]["z je Stufe"].tolist()
    z_basis = at.session_state["quick_cal"]["z"][C.M_BASIS]
    assert after[0] == " / ".join(UI.F.fmt_num(z, 2) for z in z_basis) and after != before


def test_the_calibration_is_dropped_when_the_chain_changes_but_kept_for_a_new_seed(small_calibration):
    at = fresh()
    click(at, CAL)
    set_and_run(at, seed_input=7)
    assert any(i.value.startswith("Grobe Live-Kalibrierung aktiv") for i in at.info)                          # gilt für die Kette, nicht für den Seed
    set_and_run(at, cv_slider=0.3)
    assert not any(i.value.startswith("Grobe Live-Kalibrierung aktiv") for i in at.info) and any("gilt nicht mehr" in c.value for c in at.caption)
    set_and_run(at, cv_slider=0.2, seed_input=C.SEED_DEFAULT)
    assert any(i.value.startswith("Grobe Live-Kalibrierung aktiv") for i in at.info)                          # zurück auf dieselbe Kette: wieder gültig


def test_the_calibration_does_not_apply_in_fixed_z_mode(small_calibration):
    at = fresh()
    click(at, CAL)
    set_and_run(at, z_mode="fest")
    tab = next(f for f in frames(at) if "Herkunft z" in f.columns)
    assert set(tab["Herkunft z"]) == {"fest 1,645"} and not any(i.value.startswith("Grobe Live-Kalibrierung aktiv") for i in at.info)


def test_reset_button_returns_to_the_precalculated_z(small_calibration):
    at = fresh()
    click(at, CAL)
    click(at, "Vorgerechnetes z wieder verwenden")
    assert "quick_cal" not in at.session_state and not any(i.value.startswith("Grobe Live-Kalibrierung aktiv") for i in at.info)
    assert set(next(f for f in frames(at) if "Herkunft z" in f.columns)["Herkunft z"]) == {"vorgerechnet"}


def test_a_second_click_within_the_pause_is_refused(small_calibration):
    at = fresh()
    click(at, CAL)
    assert len(small_calibration) == 1
    at.session_state["quickcal_last"] = time.monotonic()                                                       # Pause läuft (Sitzungszustand: kein Test hängt an der Laufzeit)
    click(at, CAL)
    assert len(small_calibration) == 1 and any("Bitte noch" in w.value and "Pause von 4 s" in w.value for w in at.warning)
    at.session_state["quickcal_last"] = time.monotonic() - 1000.0                                             # Pause abgelaufen
    click(at, CAL)
    assert len(small_calibration) == 2


def test_the_time_limit_warning_names_the_skipped_measures(monkeypatch):
    monkeypatch.setattr(LV, "quick_calibrate", lambda *a, **k: {"z": {C.M_BASIS: [1.0] * 4}, "skipped": [C.M_SMOOTH, C.M_INFO], "seconds": 91.0})
    at = fresh()
    click(at, CAL)
    assert any("Zeitlimit von 90 s erreicht: 2 Maßnahmen behalten das vorgerechnete z" in w.value for w in at.warning)
    tab = next(f for f in frames(at) if "Herkunft z" in f.columns)
    assert tab["Herkunft z"].tolist()[0] == "live nachkalibriert (grob)" and tab["Herkunft z"].tolist()[1] == "vorgerechnet"


def test_the_calibration_covers_only_the_active_measures(small_calibration):
    at = fresh(l="1", p="40")
    click(at, CAL)
    assert list(small_calibration[0][6]) == [C.M_BASIS, C.M_SMOOTH, C.M_RETAIL, C.M_INFO, C.M_INFO_SMOOTH]


# ---------------------------------------------------------------------------------------------------
# Live-Rechnung, Zwischenspeicher
# ---------------------------------------------------------------------------------------------------
def test_only_the_live_settings_recompute_the_run_display_choices_do_not(monkeypatch):
    calls = []
    real = LV.solve_live
    monkeypatch.setattr(LV, "solve_live", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    at = fresh()
    assert len(calls) == 1
    at.selectbox(key="measure_select").set_value(C.M_INFO).run()
    at.radio(key="stage_select").set_value(2).run()
    at.radio(key="view_select").set_value("Lager").run()
    at.selectbox(key="formula_select").set_value("m = 4, p = 3").run()
    at.selectbox(key="map_rho").set_value(0.9).run()
    assert len(calls) == 1 and not at.exception                                      # alle Anzeigewahlen sind reine Anzeige (st.cache_data)
    set_and_run(at, gamma_slider=0.3)
    assert len(calls) == 2
    set_and_run(at, gamma_slider=0.5)
    assert len(calls) == 2                                                           # zurück auf eine schon gerechnete Einstellung: aus dem Zwischenspeicher


def test_the_seven_measures_are_always_computed_for_each_setting(monkeypatch):
    seen = []
    real = LV.run_measure
    monkeypatch.setattr(LV, "run_measure", lambda D, m, *a, **k: (seen.append(m), real(D, m, *a, **k))[1])
    fresh()
    assert seen == list(C.MEASURES)


def test_display_choices_keep_the_metrics_but_change_the_charts():
    at = fresh()
    before = main_metrics(at)
    at.radio(key="stage_select").set_value(1).run()
    at.radio(key="view_select").set_value("Lager").run()
    assert not at.exception and main_metrics(at) == before
    assert at.radio(key="stage_select").value == 1 and at.radio(key="view_select").value == "Lager"


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②, Ansichten, Texte
# ---------------------------------------------------------------------------------------------------
def test_core_section_two_shows_the_measured_numbers():
    at = fresh()
    text = " ".join(c.value for c in at.caption)
    assert "halbierte Wiederbeschaffungszeit **65,1 %** Kettenbestand" in text and "geteilte Bedarfsinformation **26,9 %**" in text and "nur **17,3 %**" in text
    assert "von 11,1 auf 1,8 drückt (−83 %)" in text and "„Nur Einzelhandel glättet, γ 0,3“ spart 30,8 %" in text
    assert "innerhalb -0,4 bis +0,5 %" in text and "Rangkorrelation 1,000" in text
    assert "Rangkorrelation der Verstärkung an Stufe 4 mit der Summe Lager über die 13 Maßnahmen: **0,45**" in text and "**0,27**" in text and "steigt aber um 1,6 %" in text
    assert "zwischen 1,03 (m 1, p 20) und 3,43 (m 4, p 3)" in text and "Simulation 37,9, exakt 37,7, Produktformel 20,2" in text
    assert "Minimum bei p = 32" in text and "+4,6 % gegen p 32" in text
    md = " ".join(m.value for m in at.markdown)
    assert "Summe Lager 293 → 152 bei γ 0,5" in md and "β je Stufe 92,3 / 85,2 / 83,3 / 87,8 % bei γ 0,5 gegen 97,6 / 95,7 / 94,0 / 93,5 % in der Basis" in md
    for needle in ("**1 · Maßnahmen bei gleichem Servicegrad**", "**2 · Fester z gegen gleicher Servicegrad**", "**3 · Wer zahlt?**", "**4 · Regime-Landkarte**", "**5 · Verstärkung als Kostenproxy**",
                   "**6 · Formel gegen Simulation**", "**7 · Fenster mit Niveausprüngen**", "in diesem Modell, bei diesen Reglern", "500 Wiederholungen je Zelle", "nie live"):
        assert needle in md, needle


def test_core_section_reacts_to_the_settings():
    at = fresh(rho="0.8")
    assert any(f"**{R.cell_setting_text(DATA, 'rho0.8')}**" in m.value for m in at.markdown)
    regime = next(f for f in frames(at) if "Urteil Glättung" in f.columns)
    assert len(regime) == 10 and [z for z in regime["Zelle"] if z.startswith("▶")] == ["▶ ρ 0,8"] and regime["Urteil Glättung"][2] == "Glättung kostet hier"
    payer = next(f for f in frames(at) if "Lager Basis" in f.columns and "alle glätten (γ 0,5)" in f.columns)
    assert len(payer) == 4
    at = fresh(g="0.3")
    assert any("alle glätten (γ 0,3)" in " ".join(map(str, f.columns)) for f in frames(at))


def test_the_formula_choice_changes_the_chart_only():
    at = fresh()
    assert at.selectbox(key="formula_select").value == "m = 2, p = 5" and len(at.selectbox(key="formula_select").options) == 12
    at.selectbox(key="formula_select").set_value("m = 4, p = 3").run()
    assert not at.exception and at.selectbox(key="formula_select").value == "m = 4, p = 3"


def test_theory_tab_shows_the_control_and_the_practice_table():
    at = fresh()
    fr = frames(at)
    amp = next(f for f in fr if "Kontrolle: Simulation z = 0" in f.columns)
    assert amp["exakte Theorie (Faltung)"].tolist() == ["1,48", "2,31", "3,75", "6,33"]
    practice = next(f for f in fr if "z = 1,645 mit Abschneidung" in f.columns)
    assert len(practice) == 9 and set(practice["m (= L)"]) == {1, 2, 4} and set(practice["p"]) == {5, 10, 20}
    row = practice[(practice["m (= L)"] == 2) & (practice["p"] == 10)].iloc[0]
    assert row["lineare Theorie"] == "6,33" and row["z = 1,645 mit Abschneidung"] == "12,56" and row["z = 1,645 ohne Abschneidung"] == "14,80"
    assert any("Kontrolle: dieselbe Kette ohne Sicherheitsbestand" in c.value for c in at.caption)


def test_measurement_tab_shows_the_coverage():
    at = fresh()
    cov = next(f for f in frames(at) if "Abstand zur nächstliegenden gemessenen Zelle" in f.columns)
    assert cov["Kombinationen aus CV, ρ, L, p"].tolist() == [10, 30, 44, 24]
    assert any("Nur 10 von 108 Kombinationen liegen genau auf einer gemessenen Zelle" in c.value and "46" in c.value and "1341" in c.value for c in at.caption)


def test_expander_texts_state_the_limits_and_the_findings():
    at = fresh()
    how = next(m.value for m in at.markdown if m.value.strip().startswith("**Kette.**"))
    for needle in ("AR(1)", "unbegrenzten externen Lieferanten", "Backorders", "Order-up-to", "Abschneidung bei 0", "Glättung", "Info teilen", "immer alle Maßnahmen", "Warum der Servicegrad festgehalten wird",
                   "unabhängigen Stichprobe", "Was die Live-Zahlen bedeuten", "senkt hier den Kettenbestand", "kostet hier", "bringt hier wenig", "nächstliegende gemessene Zelle", "Servicegrad live nachkalibrieren",
                   "Produktformel und exakte Theorie", "Grenzen dieses Modells", "Stark stilisiert", "erfunden, nicht kalibriert", "Lehrbuchthema", "Chen et al. wurden nicht nachgeschlagen", "Varianzkosten nicht bewertet",
                   "Kleine Fenster", "Schätzartefakt", "proportionaler Regler", "Nicht Teil dieser Demo", "Kettenlängen", "Losgrößen", "Rationierung", "Informationsverzögerung"):
        assert needle in how, needle
    math = next(m.value for m in at.markdown if m.value.strip().startswith("**Order-up-to mit gleitendem Durchschnitt.**"))
    for needle in ("S_k(t)", "\\max", "\\gamma", "FIR-Filter", "\\Sigma_\\rho", "1 + 2m/p + 2m^2/p^2", "M_k = L_1", "\\beta_k", "95\\,\\%", "bw_model.py", "bw_theory.py", "bw_live.py", "bw_results.py", "2\\,\\text{SE}"):
        assert needle in math, needle


def test_no_dead_file_links_in_markdown():
    at = fresh()
    for md in list(at.markdown) + list(at.caption):
        for target in re.findall(r"\]\(([^)]+)\)", md.value):
            assert target.startswith("https://"), (target, md.value[:80])                  # nur echte Links, kein [foo.py](foo.py)


def test_real_umlauts_and_no_ascii_replacements_in_texts():
    at = fresh()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for bad in (" fuer ", " ueber ", " waehlen ", "Aenderung", "Loesung", "Bestaende", "Moeglich", "Groesse", "Glaettung", "Verstaerkung", "Maßnahme".replace("ß", "ss")):
        assert bad not in text, bad
    assert "Glättung" in text and "Verstärkung" in text and "Wiederbeschaffungszeit" in text


def test_all_plotly_charts_render_with_unique_keys():
    at = fresh()
    charts = at.get("plotly_chart")
    assert len(charts) == 12
    ids = [c.proto.id for c in charts]
    assert len(set(ids)) == len(ids)


def test_dataframes_have_no_null_columns():
    """Keine Ergebnisspalte ist überall gleich (Nullspalten-Signal), außer den per Definition konstanten."""
    at = fresh()
    constant_ok = {"Herkunft z", "Bezug"}
    for d in at.dataframe:
        df = d.value
        if len(df) < 3:
            continue
        for col in df.columns:
            if str(col) in constant_ok or str(col).startswith(("Bestellung = 0",)):
                continue
            assert df[col].astype(str).nunique() > 1, (col, df[col].tolist()[:3])


def test_pdf_download_button_is_present_and_named():
    at = fresh()
    buttons = at.get("download_button")
    assert len(buttons) == 1 and buttons[0].proto.label == "📄 Auswertung als PDF herunterladen"


def test_app_survives_every_preset_with_every_display_choice():
    at = fresh()
    for name in C.PRESETS:
        click(at, name)
        for m in at.selectbox(key="measure_select").options:
            pass
        for stage in (1, 4):
            for view in ("Bestellungen", "Lager"):
                at.radio(key="stage_select").set_value(stage)
                at.radio(key="view_select").set_value(view)
                at.run()
                assert not at.exception, (name, stage, view)
        for m in list(R.active_measures(C.PRESETS[name]["lead"], C.PRESETS[name]["window"]))[1:]:
            at.selectbox(key="measure_select").set_value(m).run()
            assert not at.exception, (name, m)


def test_a_borrowed_z_far_from_the_target_service_gets_a_warning():
    at = fresh(rho="0.8", l="4")                                                                          # keine gemessene Zelle: z aus der nächstliegenden Konfiguration
    text = box(at, "warning", "Achtung: das z je Stufe stammt hier aus der nächstliegenden gemessenen Konfiguration")
    assert text is not None and "„Servicegrad live nachkalibrieren“" in text
    assert "z-Herkunft: z aus nächstliegender Konfiguration." in at.metric[3].help
    assert box(fresh(), "warning", "Achtung: das z je Stufe") is None                                     # exakt gemessen: keine Warnung
    assert box(fresh(rho="0.8", l="4", z="fest"), "warning", "Achtung: das z je Stufe") is None          # fester z: der Servicegrad wird ohnehin nicht festgehalten
