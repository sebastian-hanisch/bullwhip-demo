"""Regler-Spezifikation, Permalink (Begrenzen und Einrasten, Textwahl), Presets, Seed-Knopf und Reglergrenzen."""
import random

import pytest
import streamlit as st

import bw_constants as C
import bw_presets as P
from bw_presets import PRESET_STATE_KEYS, SETTING_SPECS


@pytest.fixture
def fake_state(monkeypatch):
    """st.session_state und st.query_params durch einfache dicts ersetzen (ohne laufende App)."""
    state, params = {}, {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "query_params", params)
    return state, params


def test_setting_specs_cover_every_control_once():
    assert list(SETTING_SPECS) == ["cv_slider", "rho_slider", "lead_slider", "window_slider", "gamma_slider", "z_mode", "seed_input"]
    assert {s.url_param for s in SETTING_SPECS.values()} == {"cv", "rho", "l", "p", "g", "z", "seed"} and len({s.url_param for s in SETTING_SPECS.values()}) == 7
    assert set(PRESET_STATE_KEYS.values()) == set(SETTING_SPECS) and list(PRESET_STATE_KEYS) == ["cv", "rho", "lead", "window", "gamma", "zmode", "seed"]
    assert SETTING_SPECS["cv_slider"].options == C.CV_OPTIONS == (0.1, 0.2, 0.3) and SETTING_SPECS["rho_slider"].options == (0.0, 0.5, 0.8)
    assert SETTING_SPECS["lead_slider"].options == (1, 2, 4) and SETTING_SPECS["window_slider"].options == (6, 10, 20, 40) and SETTING_SPECS["gamma_slider"].options == (0.7, 0.5, 0.3)
    assert SETTING_SPECS["z_mode"].options == ("kalibriert", "fest")
    assert [SETTING_SPECS[k].default for k in SETTING_SPECS] == [0.2, 0.0, 2, 10, 0.5, "kalibriert", C.SEED_DEFAULT]
    assert P.bounds("seed_input") == (0, 299) and P.bounds("cv_slider") == (None, None)


def test_parse_setting_snaps_to_the_nearest_stage_and_the_smaller_one_on_a_tie():
    cv = SETTING_SPECS["cv_slider"]
    assert P.parse_setting(cv, "0.1") == 0.1 and P.parse_setting(cv, "0.24") == 0.2 and P.parse_setting(cv, "0.26") == 0.3 and P.parse_setting(cv, "0.15") == 0.1 and P.parse_setting(cv, "5") == 0.3
    lead = SETTING_SPECS["lead_slider"]
    assert P.parse_setting(lead, "3") == 2 and P.parse_setting(lead, "3.1") == 4 and P.parse_setting(lead, "0") == 1 and P.parse_setting(lead, "100") == 4
    win = SETTING_SPECS["window_slider"]
    assert P.parse_setting(win, "8") == 6 and P.parse_setting(win, "15") == 10 and P.parse_setting(win, "16") == 20 and P.parse_setting(win, "30") == 20 and P.parse_setting(win, "31") == 40
    gam = SETTING_SPECS["gamma_slider"]
    assert P.parse_setting(gam, "0.6") == 0.5 and P.parse_setting(gam, "0.38") == 0.3 and P.parse_setting(gam, "1") == 0.7 and P.parse_setting(gam, "0") == 0.3
    assert P.parse_setting(SETTING_SPECS["rho_slider"], "0.65") == 0.5 and P.parse_setting(SETTING_SPECS["rho_slider"], "0.66") == 0.8


def test_parse_setting_of_the_text_option_needs_an_exact_match():
    spec = SETTING_SPECS["z_mode"]
    assert P.parse_setting(spec, "fest") == "fest" and P.parse_setting(spec, "kalibriert") == "kalibriert"
    for junk in ("Fest", "", "fest ", "abc", None, "1"):
        assert P.parse_setting(spec, junk) is None, junk


def test_parse_setting_clamps_the_seed_and_rejects_garbage():
    seed = SETTING_SPECS["seed_input"]
    assert P.parse_setting(seed, "17") == 17 and P.parse_setting(seed, "999") == 299 and P.parse_setting(seed, "-4") == 0 and P.parse_setting(seed, "12.6") == 13 and P.parse_setting(seed, "299") == 299
    assert isinstance(P.parse_setting(seed, "12.6"), int)
    for name, spec in SETTING_SPECS.items():
        for junk in ("abc", "", "nan", "inf", "-inf", None):
            assert P.parse_setting(spec, junk) is None, (name, junk)


def test_encoders_write_short_numbers():
    assert P._num(0.3) == "0.3" and P._num(0.0) == "0" and P._num(0.8) == "0.8" and P._text(10) == "10"
    assert SETTING_SPECS["rho_slider"].encoder(0.0) == "0" and SETTING_SPECS["window_slider"].encoder(6) == "6" and SETTING_SPECS["z_mode"].encoder("fest") == "fest"


def test_permalink_is_loaded_once_and_snaps(fake_state):
    state, params = fake_state
    params.update({"cv": "0.27", "rho": "0.6", "l": "3", "p": "12", "g": "0.38", "z": "fest", "seed": "999"})
    P.load_permalink_settings()
    assert state["cv_slider"] == 0.3 and state["rho_slider"] == 0.5 and state["lead_slider"] == 2 and state["window_slider"] == 10 and state["gamma_slider"] == 0.3
    assert state["z_mode"] == "fest" and state["seed_input"] == 299 and state["permalink_loaded"] is True
    state["cv_slider"] = 0.1
    params["cv"] = "0.3"
    P.load_permalink_settings()
    assert state["cv_slider"] == 0.1                                                                  # nur beim ersten Lauf


def test_permalink_ignores_unusable_values_and_missing_keys(fake_state):
    state, params = fake_state
    params.update({"cv": "viel", "l": "4", "z": "egal"})
    P.load_permalink_settings()
    assert "cv_slider" not in state and state["lead_slider"] == 4 and "z_mode" not in state and "rho_slider" not in state


def test_defaults_fill_only_missing_state(fake_state):
    state, _ = fake_state
    state["lead_slider"] = 4
    P.init_session_state_defaults()
    assert state["lead_slider"] == 4 and state["cv_slider"] == 0.2 and state["seed_input"] == C.SEED_DEFAULT and state["z_mode"] == "kalibriert" and set(state) == set(SETTING_SPECS)


def test_sync_query_params_writes_the_encoded_values(fake_state):
    _, params = fake_state
    P.sync_query_params({"cv_slider": 0.3, "rho_slider": 0.0, "z_mode": "fest", "seed_input": 211})
    assert params == {"cv": "0.3", "rho": "0", "z": "fest", "seed": "211"}


def test_sync_query_params_survives_a_broken_query_string_api(monkeypatch):
    class Broken:
        def __setitem__(self, k, v):
            raise RuntimeError("kein Zugriff")
    monkeypatch.setattr(st, "query_params", Broken())
    P.sync_query_params({"seed_input": 3})                                                             # darf nicht abstürzen


def test_apply_preset_sets_every_control(fake_state):
    state, _ = fake_state
    for name, preset in C.PRESETS.items():
        P.apply_preset(name)
        assert {k: state[k] for k in SETTING_SPECS} == {PRESET_STATE_KEYS[f]: v for f, v in preset.items()}
    P.apply_preset("Träger Bedarf")
    assert (state["rho_slider"], state["lead_slider"], state["window_slider"]) == (0.8, 2, 10)
    with pytest.raises(KeyError):
        P.apply_preset("gibt es nicht")


def test_randomize_seed_stays_in_range(fake_state, monkeypatch):
    state, _ = fake_state
    seen = []
    monkeypatch.setattr(random, "randint", lambda lo, hi: (seen.append((lo, hi)), 123)[1])
    P.randomize_seed()
    assert state["seed_input"] == 123 and seen == [(0, 299)]
    monkeypatch.undo()
    for _ in range(50):
        P.randomize_seed()
        assert 0 <= state["seed_input"] <= 299


def test_presets_are_five_short_named_measured_stages_with_help():
    assert list(C.PRESETS) == ["Standard", "Kurze Lieferzeit", "Träger Bedarf", "Kleines Fenster", "Großes Fenster"] and list(C.PRESET_HELP) == list(C.PRESETS)
    for name, p in C.PRESETS.items():
        assert len(name) <= 32 and len(C.PRESET_HELP[name]) > 20 and set(p) == set(PRESET_STATE_KEYS)
        for field, spec_key in PRESET_STATE_KEYS.items():
            spec = SETTING_SPECS[spec_key]
            assert (p[field] in spec.options) if spec.options else (spec.lo <= p[field] <= spec.hi), (name, field)
    assert C.PRESETS["Standard"] == dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode="kalibriert", seed=C.SEED_DEFAULT)
    assert 200 <= C.SEED_DEFAULT <= 299 and len({p["seed"] for p in C.PRESETS.values()}) == 1
    assert all(C.PRESETS[n]["zmode"] == C.ZMODE_EQUAL for n in C.PRESETS)


def test_constants_of_the_model_match_the_measurement_series():
    assert (C.N_STAGES, C.MU, C.H_COST, C.P_COST, C.TARGET_SERVICE, C.FIXED_Z, C.BURN) == (4, 100.0, 1.0, 9.0, 0.95, 1.645, 100)
    assert C.LIVE_SHOWN == 300 and C.LIVE_T == 400 and len(C.STAGE_NAMES) == 4 and len(C.STAGE_COLORS) == 4
    assert (C.QUICK_R, C.QUICK_T, C.QUICK_PASSES, C.QUICK_ITERS) == (20, 700, 1, 9) and C.QUICK_TIME_LIMIT > 30 and C.QUICK_COOLDOWN > 0
    assert (C.LOWER_PCT, C.RAISE_PCT, C.SE_FACTOR) == (-5.0, 2.0, 2.0) and set(C.MEASURES) == set(C.MEASURE_LABELS) and len(C.MEASURES) == 7
    assert set(C.MEASURE_SHORT) == set(C.MEASURES) and all(len(v) <= 15 for v in C.MEASURE_SHORT.values()) and C.SERVICE_WARN_GAP == 0.05
    assert set(C.SMOOTH_MEASURES) <= set(C.MEASURES) and C.MEASURE_DEFAULT in C.MEASURES and C.MEASURE_DEFAULT != C.M_BASIS
    assert set(C.ZMODE_LABELS) == set(C.ZMODE_OPTIONS) and C.LEAD_AXIS == (1, 2, 4) and 12 in C.WINDOW_AXIS and set(C.WINDOW_OPTIONS) <= set(C.WINDOW_AXIS)


def test_setting_specs_are_immutable_and_the_constants_are_pinned():
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        SETTING_SPECS["cv_slider"].default = 0.3
    assert C.CACHE_ENTRIES == 24 and C.WINDOW_AXIS == (6, 10, 12, 20, 40, 80) and C.LEAD_AXIS == (1, 2, 4) and C.BURN == 100 and C.QUICK_SEED_OFFSET == 100000
    assert C.CHART_HEIGHT == 380 and C.QUICK_ZMAX == 10.0 and C.QUICK_TIME_LIMIT == 90.0 and C.QUICK_COOLDOWN == 4.0
