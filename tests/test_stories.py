"""Abnahmekriterien der Presets an KÜNSTLICHEN Werten: jedes Kriterium kippt einzeln an seiner Schwelle (Kopie der Ergebnisdatei mit verstellten Werten), Vorzeichen-Kriterien nur zusammen
mit der Standardfehler-Bedingung. Die Abnahme der echten Presets steht in tests/test_preset_stories.py."""
import copy

import pytest

import bw_constants as C
import bw_results as R
import bw_stories as ST

DATA = R.load_results()
CELL_OF = {"Standard": "main", "Kurze Lieferzeit": "L1", "Träger Bedarf": "rho0.8", "Kleines Fenster": "p6", "Großes Fenster": "p40"}


def texts(name):
    return [t for _, t in ST.criteria(name, DATA)]


def crit(name, *mods):
    d = copy.deepcopy(DATA)
    for m in mods:
        m(d)
    return [ok for ok, _ in ST.criteria(name, d)]


def set_delta(cell, job, delta, se=0.05):
    """Setzt die Änderung der Summe Lager `job` gegenüber der Basis der Zelle (in %) und ihren Standardfehler (Standardfehler der Basis = 0)."""
    def m(d):
        base = d["jobs"][f"{cell}/basis"]["equal"]
        base["total_onhand_se"] = 0.0
        e = d["jobs"][f"{cell}/{job}"]["equal"]
        e["total_onhand"] = base["total_onhand"] * (1 + delta / 100.0)
        e["total_onhand_se"] = se * base["total_onhand"] / 100.0
    return m


def smooth(name, delta, se=0.05):
    return set_delta(CELL_OF[name], "glaettung_0.5", delta, se)


def info(name, delta, se=0.05):
    return set_delta(CELL_OF[name], "info_teilen", delta, se)


def set_amp(cell, job, value):
    def m(d):
        d["jobs"][f"{cell}/{job}"]["equal"]["amp"][3][0] = value
    return m


def test_the_helper_sets_exactly_the_wanted_change_and_error():
    d = copy.deepcopy(DATA)
    set_delta("main", "glaettung_0.5", -12.0, 0.7)(d)
    r = R.resolve_pair(d, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)
    assert r["delta"] == pytest.approx(-12.0) and r["se"] == pytest.approx(0.7)


def test_all_real_criteria_hold_and_have_texts():
    for name in C.PRESETS:
        result = ST.criteria(name, DATA)
        assert result and all(ok for ok, _ in result), (name, result)
        assert all(isinstance(t, str) and len(t) > 10 for _, t in result)
    assert [len(ST.criteria(n, DATA)) for n in C.PRESETS] == [3, 2, 2, 2, 2]
    with pytest.raises(KeyError):
        ST.criteria("gibt es nicht", DATA)


# --- Standard ---------------------------------------------------------------------------------------------
def test_standard_smoothing_must_save_ten_percent_and_clear_two_standard_errors():
    assert crit("Standard", smooth("Standard", -10.02))[0] is True and crit("Standard", smooth("Standard", -9.98))[0] is False
    assert crit("Standard", smooth("Standard", -20.0, 10.1))[0] is False and crit("Standard", smooth("Standard", -20.0, 9.9))[0] is True          # 2 SE: nicht belastbar
    assert crit("Standard", smooth("Standard", 5.0))[0] is False and ST.STANDARD_SMOOTH_MAX == -10.0


def test_standard_amplification_must_fall_by_seventy_percent():
    base = DATA["jobs"]["main/basis"]["equal"]["amp"][3][0]
    assert crit("Standard", set_amp("main", "glaettung_0.5", 0.2999 * base))[1] is True
    assert crit("Standard", set_amp("main", "glaettung_0.5", 0.3001 * base))[1] is False and crit("Standard", set_amp("main", "glaettung_0.5", base * 1.2))[1] is False
    assert ST.STANDARD_AMP_DROP_MIN == 0.70


def test_standard_info_sharing_must_save_twenty_percent_and_clear_two_standard_errors():
    assert crit("Standard", info("Standard", -20.02))[2] is True and crit("Standard", info("Standard", -19.98))[2] is False
    assert crit("Standard", info("Standard", -30.0, 15.1))[2] is False and crit("Standard", info("Standard", -30.0, 14.9))[2] is True
    assert ST.STANDARD_INFO_MAX == -20.0


# --- Kurze Lieferzeit ---------------------------------------------------------------------------------------------
def test_short_lead_smoothing_must_change_less_than_two_percent_in_both_directions():
    for v, ok in ((1.99, True), (2.01, False), (-1.99, True), (-2.01, False), (0.0, True), (7.0, False)):
        assert crit("Kurze Lieferzeit", smooth("Kurze Lieferzeit", v))[0] is ok, v
    assert ST.SHORT_SMOOTH_ABS_MAX == 2.0


def test_short_lead_info_sharing_must_be_about_minus_ten_percent_and_clear_of_zero():
    for v, ok in ((-5.02, True), (-4.98, False), (-14.98, True), (-15.02, False), (-10.0, True), (0.0, False), (-40.0, False)):
        assert crit("Kurze Lieferzeit", info("Kurze Lieferzeit", v))[1] is ok, v
    assert crit("Kurze Lieferzeit", info("Kurze Lieferzeit", -8.0, 4.1))[1] is False and crit("Kurze Lieferzeit", info("Kurze Lieferzeit", -8.0, 3.9))[1] is True
    assert (ST.SHORT_INFO_LO, ST.SHORT_INFO_HI) == (-15.0, -5.0)


# --- Träger Bedarf -----------------------------------------------------------------------------------------------
def test_sluggish_demand_smoothing_must_cost_and_by_more_than_two_standard_errors():
    assert crit("Träger Bedarf", smooth("Träger Bedarf", 0.5, 0.2))[0] is True and crit("Träger Bedarf", smooth("Träger Bedarf", 0.5, 0.3))[0] is False        # 0,5 gegen 2 SE = 0,6
    assert crit("Träger Bedarf", smooth("Träger Bedarf", -0.5, 0.1))[0] is False and crit("Träger Bedarf", smooth("Träger Bedarf", 0.0, 0.0))[0] is False
    assert crit("Träger Bedarf", smooth("Träger Bedarf", 6.9, 0.4))[0] is True


def test_sluggish_demand_info_sharing_must_save_thirty_percent_and_clear_two_standard_errors():
    assert crit("Träger Bedarf", info("Träger Bedarf", -30.02))[1] is True and crit("Träger Bedarf", info("Träger Bedarf", -29.98))[1] is False
    assert crit("Träger Bedarf", info("Träger Bedarf", -40.0, 20.1))[1] is False and crit("Träger Bedarf", info("Träger Bedarf", -40.0, 19.9))[1] is True
    assert ST.SLUGGISH_INFO_MAX == -30.0


# --- Kleines Fenster ---------------------------------------------------------------------------------------------
def test_small_window_smoothing_and_info_sharing_must_each_save_forty_percent():
    assert crit("Kleines Fenster", smooth("Kleines Fenster", -40.02))[0] is True and crit("Kleines Fenster", smooth("Kleines Fenster", -39.98))[0] is False
    assert crit("Kleines Fenster", info("Kleines Fenster", -40.02))[1] is True and crit("Kleines Fenster", info("Kleines Fenster", -39.98))[1] is False
    assert crit("Kleines Fenster", smooth("Kleines Fenster", -60.0, 30.1))[0] is False and crit("Kleines Fenster", info("Kleines Fenster", -60.0, 30.1))[1] is False
    assert crit("Kleines Fenster", smooth("Kleines Fenster", -60.0, 29.9))[0] is True and (ST.SMALL_SMOOTH_MAX, ST.SMALL_INFO_MAX) == (-40.0, -40.0)


def test_small_window_criteria_are_independent_of_each_other():
    assert crit("Kleines Fenster", smooth("Kleines Fenster", -10.0)) == [False, True] and crit("Kleines Fenster", info("Kleines Fenster", -10.0)) == [True, False]


# --- Großes Fenster -----------------------------------------------------------------------------------------------
def test_large_window_both_changes_must_stay_under_three_percent():
    for v, ok in ((2.99, True), (3.01, False), (-2.99, True), (-3.01, False), (0.0, True)):
        assert crit("Großes Fenster", info("Großes Fenster", v))[0] is ok, v
        assert crit("Großes Fenster", smooth("Großes Fenster", v))[1] is ok, v
    assert ST.LARGE_ABS_MAX == 3.0
    assert crit("Großes Fenster", info("Großes Fenster", 8.0)) == [False, True] and crit("Großes Fenster", smooth("Großes Fenster", 8.0)) == [True, False]


def test_criteria_need_exactly_measured_cells():
    d = copy.deepcopy(DATA)
    del d["jobs"]["p40/glaettung_0.5"]
    with pytest.raises((ValueError, KeyError)):
        ST.criteria("Großes Fenster", d)


def test_criteria_texts_carry_the_measured_numbers():
    t = texts("Standard")
    assert "-15,7 ± 0,2 %" in t[0] and "11,1 auf 2,9 (-74 %)" in t[1] and "-26,9 ± 0,1 %" in t[2]
    assert "+0,6 ± 0,2" in texts("Kurze Lieferzeit")[0] and "-10,0 ± 0,1" in texts("Kurze Lieferzeit")[1]
    assert "+6,9 ± 0,4" in texts("Träger Bedarf")[0] and "-36,0 ± 0,2" in texts("Träger Bedarf")[1]
    assert "-47,4 ± 0,1" in texts("Kleines Fenster")[0] and "-52,6 ± 0,1" in texts("Kleines Fenster")[1]
    assert "-1,9 ± 0,1" in texts("Großes Fenster")[0] and "-1,8 ± 0,2" in texts("Großes Fenster")[1]


# --- gezeigter Verlauf ---------------------------------------------------------------------------------------------
def fake_live(smooth_delta=-10.0, info_delta=-10.0, smooth_amp=2.0, info_amp=5.0):
    """Künstlicher Live-Verlauf: nur die Felder, die day_criteria liest (Basis: Summe Lager 100, Verstärkung Stufe 4: 10)."""
    mk = lambda total, amp4: {"total_onhand": total, "amp": [1.0, 1.0, 1.0, amp4]}
    return {"measures": {C.M_BASIS: mk(100.0, 10.0), C.M_SMOOTH: mk(100.0 * (1 + smooth_delta / 100), smooth_amp), C.M_INFO: mk(100.0 * (1 + info_delta / 100), info_amp)}}


def dcrit(name, live):
    return [ok for ok, _ in ST.day_criteria(name, live)]


def test_live_changes_are_relative_to_the_basis():
    x = ST.live_changes(fake_live(-20.0, 10.0, 3.0, 5.0))
    assert x[C.M_BASIS] == {"delta": 0.0, "amp_ratio": 1.0} and x[C.M_SMOOTH]["delta"] == pytest.approx(-20.0) and x[C.M_SMOOTH]["amp_ratio"] == pytest.approx(0.3) and x[C.M_INFO]["delta"] == pytest.approx(10.0)


def test_day_criteria_standard():
    assert dcrit("Standard", fake_live(-10, -10.1, 4.99)) == [True, True, True] and dcrit("Standard", fake_live(-10, -10.1, 5.01)) == [False, True, True]
    assert dcrit("Standard", fake_live(0.1, -10.1)) == [True, False, True] and dcrit("Standard", fake_live(-10, -9.9)) == [True, True, False] and dcrit("Standard", fake_live(-10, -10.1))[2] is True
    assert dcrit("Standard", fake_live(-0.1))[1] is True and dcrit("Standard", fake_live(0.0))[1] is False


def test_day_criteria_short_lead():
    assert dcrit("Kurze Lieferzeit", fake_live(14.9, -10)) == [True, True] and dcrit("Kurze Lieferzeit", fake_live(15.1, -10)) == [False, True] and dcrit("Kurze Lieferzeit", fake_live(-15.1, -10)) == [False, True]
    assert dcrit("Kurze Lieferzeit", fake_live(0, -19.9))[1] is True and dcrit("Kurze Lieferzeit", fake_live(0, -20.1))[1] is False and dcrit("Kurze Lieferzeit", fake_live(0, 0.0))[1] is False and dcrit("Kurze Lieferzeit", fake_live(0, 5))[1] is False


def test_day_criteria_sluggish_demand():
    assert dcrit("Träger Bedarf", fake_live(0.1, -20.1)) == [True, True] and dcrit("Träger Bedarf", fake_live(0.0, -20.1)) == [False, True] and dcrit("Träger Bedarf", fake_live(-5, -20.1)) == [False, True]
    assert dcrit("Träger Bedarf", fake_live(5, -19.9))[1] is False and dcrit("Träger Bedarf", fake_live(5, -20.1))[1] is True


def test_day_criteria_small_window():
    assert dcrit("Kleines Fenster", fake_live(-25.1, -25.1)) == [True, True] and dcrit("Kleines Fenster", fake_live(-24.9, -25.1)) == [False, True] and dcrit("Kleines Fenster", fake_live(-25.1, -24.9)) == [True, False]


def test_day_criteria_large_window():
    assert dcrit("Großes Fenster", fake_live(9.9, -9.9)) == [True, True] and dcrit("Großes Fenster", fake_live(10.1, 0)) == [False, True] and dcrit("Großes Fenster", fake_live(0, -10.1)) == [True, False]
    assert dcrit("Großes Fenster", fake_live(-10.1, 0))[0] is False and dcrit("Großes Fenster", fake_live(0, 10.1))[1] is False


def test_day_criteria_unknown_preset_and_texts():
    with pytest.raises(KeyError):
        ST.day_criteria("gibt es nicht", fake_live())
    txt = [t for _, t in ST.day_criteria("Standard", fake_live(-12.3, -26.5, 2.0))]
    assert "Faktor 0,20" in txt[0] and "-12,3 %" in txt[1] and "-26,5 %" in txt[2]
    assert [len(ST.day_criteria(n, fake_live())) for n in C.PRESETS] == [3, 2, 2, 2, 2]


# ---------------------------------------------------------------------------------------------------
# Lücken aus dem Fehler-Einbau-Test: genaue Grenzen (mit darstellbaren Werten) und Textformate
# ---------------------------------------------------------------------------------------------------
def set_totals(cell, job, base, meas, se_delta=0.0):
    """Summen Lager der Basis und der Maßnahme genau setzen (darstellbare Verhältnisse wie 0,75) und den Standardfehler der Änderung (Standardfehler der Basis 0)."""
    def m(d):
        b = d["jobs"][f"{cell}/basis"]["equal"]
        b["total_onhand"], b["total_onhand_se"] = base, 0.0
        e = d["jobs"][f"{cell}/{job}"]["equal"]
        e["total_onhand"], e["total_onhand_se"] = meas, se_delta * base / 100.0
    return m


def patched(monkeypatch, **consts):
    for k, v in consts.items():
        monkeypatch.setattr(ST, k, v)


def test_exact_boundaries_of_the_measured_criteria(monkeypatch):
    patched(monkeypatch, STANDARD_SMOOTH_MAX=-25.0, STANDARD_INFO_MAX=-25.0, STANDARD_AMP_DROP_MIN=0.75, SHORT_SMOOTH_ABS_MAX=25.0, SHORT_INFO_LO=-25.0, SHORT_INFO_HI=-12.5,
            SLUGGISH_INFO_MAX=-25.0, SMALL_SMOOTH_MAX=-25.0, SMALL_INFO_MAX=-25.0, LARGE_ABS_MAX=25.0)
    down = lambda cell, job: set_totals(cell, job, 400.0, 300.0)                     # -25,0 % genau
    # inklusive Grenzen: genau auf der Schwelle gilt das Kriterium
    assert crit("Standard", down("main", "glaettung_0.5"))[0] is True and crit("Standard", down("main", "info_teilen"))[2] is True
    assert crit("Kleines Fenster", down("p6", "glaettung_0.5"))[0] is True and crit("Kleines Fenster", down("p6", "info_teilen"))[1] is True
    assert crit("Träger Bedarf", down("rho0.8", "info_teilen"))[1] is True
    assert crit("Kurze Lieferzeit", down("L1", "info_teilen"))[1] is True and crit("Kurze Lieferzeit", set_totals("L1", "info_teilen", 400.0, 350.0))[1] is True      # untere und obere Grenze inklusive
    def amp(d):
        d["jobs"]["main/basis"]["equal"]["amp"][3][0] = 8.0
        d["jobs"]["main/glaettung_0.5"]["equal"]["amp"][3][0] = 2.0
    assert crit("Standard", amp)[1] is True                                          # Abnahme genau 75 %
    # strenge Grenzen: genau auf der Schwelle gilt das Kriterium NICHT
    up = lambda cell, job: set_totals(cell, job, 400.0, 500.0)                        # +25,0 % genau
    assert crit("Kurze Lieferzeit", up("L1", "glaettung_0.5"))[0] is False and crit("Kurze Lieferzeit", down("L1", "glaettung_0.5"))[0] is False
    assert crit("Großes Fenster", up("p40", "info_teilen"))[0] is False and crit("Großes Fenster", down("p40", "info_teilen"))[0] is False
    assert crit("Großes Fenster", up("p40", "glaettung_0.5"))[1] is False and crit("Großes Fenster", down("p40", "glaettung_0.5"))[1] is False


def test_exact_two_standard_error_boundary_is_not_clear_for_every_sign_criterion(monkeypatch):
    patched(monkeypatch, STANDARD_SMOOTH_MAX=-25.0, STANDARD_INFO_MAX=-25.0, SLUGGISH_INFO_MAX=-25.0, SMALL_SMOOTH_MAX=-25.0, SMALL_INFO_MAX=-25.0, SHORT_INFO_LO=-30.0, SHORT_INFO_HI=-20.0)
    edge = lambda cell, job: set_totals(cell, job, 400.0, 300.0, 12.5)                # -25 mit Standardfehler 12,5: genau 2 SE
    assert crit("Standard", edge("main", "glaettung_0.5"))[0] is False and crit("Standard", edge("main", "info_teilen"))[2] is False
    assert crit("Kleines Fenster", edge("p6", "glaettung_0.5"))[0] is False and crit("Kleines Fenster", edge("p6", "info_teilen"))[1] is False
    assert crit("Träger Bedarf", edge("rho0.8", "info_teilen"))[1] is False and crit("Kurze Lieferzeit", edge("L1", "info_teilen"))[1] is False
    assert crit("Träger Bedarf", set_totals("rho0.8", "glaettung_0.5", 400.0, 500.0, 12.5))[0] is False                  # +25 mit 12,5: genau 2 SE
    assert crit("Träger Bedarf", set_totals("rho0.8", "glaettung_0.5", 400.0, 500.0, 12.4))[0] is True
    assert crit("Standard", set_totals("main", "glaettung_0.5", 400.0, 300.0, 12.4))[0] is True


def test_the_amplification_text_shows_a_plus_sign_when_the_fabric_gets_worse():
    def worse(d):
        d["jobs"]["main/glaettung_0.5"]["equal"]["amp"][3][0] = 2 * d["jobs"]["main/basis"]["equal"]["amp"][3][0]
    d = copy.deepcopy(DATA)
    worse(d)
    text = ST.criteria("Standard", d)[1][1]
    assert "(+100 %)" in text and ST.criteria("Standard", d)[1][0] is False


def test_exact_boundaries_of_the_day_criteria(monkeypatch):
    patched(monkeypatch, DAY_STANDARD_AMP_RATIO_MAX=0.5, DAY_STANDARD_INFO_MAX=-25.0, DAY_SHORT_SMOOTH_ABS_MAX=25.0, DAY_SHORT_INFO_MIN=-25.0, DAY_SLUGGISH_SMOOTH_MIN=25.0,
            DAY_SLUGGISH_INFO_MAX=-25.0, DAY_SMALL_MAX=-25.0, DAY_LARGE_ABS_MAX=25.0)
    assert dcrit("Standard", fake_live(-10, -10, smooth_amp=5.0))[0] is True and dcrit("Standard", fake_live(-10, -25.0))[2] is True and dcrit("Standard", fake_live(-10, -24.9))[2] is False
    assert dcrit("Kurze Lieferzeit", fake_live(25.0, -10))[0] is False and dcrit("Kurze Lieferzeit", fake_live(-25.0, -10))[0] is False and dcrit("Kurze Lieferzeit", fake_live(24.0, -10))[0] is True
    assert dcrit("Kurze Lieferzeit", fake_live(0, -25.0))[1] is False and dcrit("Kurze Lieferzeit", fake_live(0, -24.0))[1] is True
    assert dcrit("Träger Bedarf", fake_live(25.0, -25.0)) == [False, True] and dcrit("Träger Bedarf", fake_live(26.0, -24.0)) == [True, False]
    assert dcrit("Kleines Fenster", fake_live(-25.0, -25.0)) == [True, True] and dcrit("Kleines Fenster", fake_live(-24.0, -24.0)) == [False, False]
    assert dcrit("Großes Fenster", fake_live(25.0, 25.0)) == [False, False] and dcrit("Großes Fenster", fake_live(-25.0, -25.0)) == [False, False] and dcrit("Großes Fenster", fake_live(24.0, -24.0)) == [True, True]


def test_day_criteria_texts_show_sign_and_one_decimal():
    plus = {n: [t for _, t in ST.day_criteria(n, fake_live(12.345, -26.5678, 2.0))] for n in C.PRESETS}
    assert plus["Standard"] == ["Verstärkung der Fabrik mit Glättung höchstens halb so groß: Faktor 0,20", "Glättung senkt die Summe Lager: +12,3 %", "Info teilen spart mindestens 10 %: -26,6 %"]
    assert plus["Kurze Lieferzeit"] == ["|Änderung durch Glättung| unter 15 %: +12,3 %", "Info teilen spart etwas, aber weniger als 20 %: -26,6 %"]
    assert plus["Träger Bedarf"] == ["Glättung kostet Kettenbestand: +12,3 %", "Info teilen spart mindestens 20 %: -26,6 %"]
    assert plus["Kleines Fenster"] == ["Glättung spart mindestens 25 %: +12,3 %", "Info teilen spart mindestens 25 %: -26,6 %"]
    assert plus["Großes Fenster"] == ["|Änderung durch Glättung| unter 10 %: +12,3 %", "|Änderung durch Info teilen| unter 10 %: -26,6 %"]
    minus = {n: [t for _, t in ST.day_criteria(n, fake_live(-12.345, 26.5678, 2.0))] for n in C.PRESETS}
    assert minus["Kurze Lieferzeit"][1].endswith("+26,6 %") and minus["Träger Bedarf"][1].endswith("+26,6 %") and minus["Standard"][2].endswith("+26,6 %") and minus["Kleines Fenster"][0].endswith("-12,3 %")
    assert minus["Großes Fenster"][1].endswith("+26,6 %")


def test_day_criteria_texts_show_a_plus_sign_for_every_criterion_that_prints_a_positive_change():
    plus_info = {n: [t for _, t in ST.day_criteria(n, fake_live(-12.345, 26.5678, 2.0))] for n in C.PRESETS}
    assert plus_info["Kleines Fenster"][1].endswith("Info teilen spart mindestens 25 %: +26,6 %") and plus_info["Großes Fenster"][1].endswith(": +26,6 %")
    plus_smooth = {n: [t for _, t in ST.day_criteria(n, fake_live(12.345, -26.5678, 2.0))] for n in C.PRESETS}
    assert plus_smooth["Kleines Fenster"][0].endswith("Glättung spart mindestens 25 %: +12,3 %") and plus_smooth["Standard"][1].endswith(": +12,3 %")
