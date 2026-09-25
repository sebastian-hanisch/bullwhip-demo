"""Abnahme der ECHTEN Presets: die Kriterien der Messreihe (an data/bw_results.json, nicht an frisch gewürfelten Verläufen) und die qualitativen Kriterien am gezeigten Verlauf. Beides ist
unabhängig vom Zufallsgenerator (die CI installiert immer das neueste NumPy): die Tageskriterien laufen auf der EINGEFRORENEN Nachfragereihe des Preset-Seeds (tests/data/bw_preset_demand.json,
tools/freeze_reference.py). Dass gen_demand dieselbe Reihe erzeugt, wird nur mit der NumPy-Version geprüft, mit der eingefroren wurde."""
import numpy as np
import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_results as R
import bw_stories as ST
from toolload import load_tool

TP = load_tool("tune_presets")
DATA = R.load_results()
FROZEN_WITH = FZ.preset_demand()["numpy"]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_measured_criteria_hold_for_every_preset(name):
    result = ST.criteria(name, DATA)
    assert result and all(ok for ok, _ in result), result


def test_preset_settings_match_the_cells_the_criteria_read():
    p = C.PRESETS
    assert (p["Standard"]["cv"], p["Standard"]["rho"], p["Standard"]["lead"], p["Standard"]["window"], p["Standard"]["gamma"]) == (0.2, 0.0, 2, 10, 0.5)
    assert p["Kurze Lieferzeit"]["lead"] == 1 and p["Träger Bedarf"]["rho"] == 0.8 and p["Kleines Fenster"]["window"] == 6 and p["Großes Fenster"]["window"] == 40
    for name, preset in p.items():
        assert preset["gamma"] == 0.5 and preset["zmode"] == C.ZMODE_EQUAL and preset["cv"] == 0.2                     # die Kriterien lesen Glättung 0,5 bei gleichem Servicegrad
        cell, exact, _ = R.nearest_cell(DATA, preset["cv"], preset["rho"], preset["lead"], preset["window"])
        assert exact, name                                                                                              # jedes Preset liegt auf einer gemessenen Zelle
        assert R.resolve_pair(DATA, C.M_SMOOTH, preset["gamma"], preset["cv"], preset["rho"], preset["lead"], preset["window"])["exact"]
        assert R.resolve_pair(DATA, C.M_INFO, preset["gamma"], preset["cv"], preset["rho"], preset["lead"], preset["window"])["exact"]


def test_every_preset_shows_its_message_state():
    expected = {"Standard": (C.STATE_LOWER, C.STATE_LOWER), "Kurze Lieferzeit": (C.STATE_LITTLE, C.STATE_LOWER), "Träger Bedarf": (C.STATE_COSTS, C.STATE_LOWER),
                "Kleines Fenster": (C.STATE_LOWER, C.STATE_LOWER), "Großes Fenster": (C.STATE_LITTLE, C.STATE_LITTLE)}
    for name, preset in C.PRESETS.items():
        args = (preset["gamma"], preset["cv"], preset["rho"], preset["lead"], preset["window"])
        got = (R.resolve_pair(DATA, C.M_SMOOTH, *args)["state"], R.resolve_pair(DATA, C.M_INFO, *args)["state"])
        assert got == expected[name], name


def test_preset_seeds_lie_outside_the_seeds_of_the_measurement_series():
    """Die Messreihe würfelt mit Seeds ab 10000 (Kalibrierung s, Auswertung s + 1, Wiederholung s + 777); die Anzeige-Seeds liegen bei 200 bis 299."""
    used = set()
    for j in DATA["jobs"].values():
        s = j["scenario"]["seed"]
        used |= {s, s + 1}
    assert min(used) >= 10000
    for name, p in C.PRESETS.items():
        assert 200 <= p["seed"] <= C.SEED_RANGE[1] and p["seed"] not in used, name
    assert C.SEED_RANGE[1] < 10000


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_day_criteria_hold_on_the_frozen_preset_series(name):
    p = C.PRESETS[name]
    live = TP.solve_preset(p, p["seed"], demand=FZ.preset_series(p["cv"], p["rho"]))
    crit = ST.day_criteria(name, live)
    assert all(ok for ok, _ in crit), crit


@pytest.mark.skipif(np.__version__ != FROZEN_WITH, reason="der Zufallsstrom kann sich zwischen NumPy-Versionen ändern: gen_demand nur mit der Einfrier-Version bitgleich prüfen")
@pytest.mark.parametrize("cv,rho", [(0.2, 0.0), (0.2, 0.8)])
def test_the_generator_reproduces_the_frozen_preset_series_with_the_same_numpy(cv, rho):
    assert np.array_equal(LV.gen_demand(cv, rho, C.SEED_DEFAULT), FZ.preset_series(cv, rho))


def test_the_frozen_series_belong_to_the_preset_seed_and_cover_all_presets():
    fz = FZ.preset_demand()
    assert fz["seed"] == C.SEED_DEFAULT and set(fz["series"]) == {f"{p['cv']}/{p['rho']}" for p in C.PRESETS.values()} == {"0.2/0.0", "0.2/0.8"}


def test_the_live_run_of_every_preset_has_the_full_structure():
    for name, p in C.PRESETS.items():
        live = TP.solve_preset(p, p["seed"], demand=FZ.preset_series(p["cv"], p["rho"]))
        assert live["T"] == 400 and set(live["measures"]) == set(R.active_measures(p["lead"], p["window"])) and live["measures"][C.M_BASIS]["total_onhand"] > 0


def test_tune_presets_evaluate_uses_the_criteria_and_reports_the_smoothing_change():
    res = TP.evaluate(C.SEED_DEFAULT)
    assert list(res) == list(C.PRESETS) and all(isinstance(v[0], bool) and len(v[1]) >= 2 for v in res.values())
    ok, crit, delta = TP.evaluate(C.SEED_DEFAULT, {"Standard": C.PRESETS["Standard"]})["Standard"]
    assert isinstance(delta, float) and len(crit) == 3


def test_a_preset_in_the_search_range_is_valid_only_if_all_five_criteria_hold(monkeypatch):
    """Die Suche nimmt nur Seeds, deren fünf Presets ALLE Kriterien erfüllen."""
    fake = {n: (True, [], -10.0) for n in C.PRESETS}
    fake["Großes Fenster"] = (False, [], 0.0)
    monkeypatch.setattr(TP, "evaluate", lambda seed, presets=None: fake)
    assert TP._one(200)[1] == "++++-"
