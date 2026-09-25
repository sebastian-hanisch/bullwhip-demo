"""Regler-Spezifikation, Permalink, Presets und Seed-Knopf (Standardmuster aus dem OR-Demo-Portfolio, siehe irp_presets.py in inventory-routing-demo).
Fünf Regler sind feste Stufen (Nachfrageschwankung, Autokorrelation, Wiederbeschaffungszeit, Prognosefenster, Glättung), einer ist eine Textwahl (Servicegrad-Modus), einer ein Zahlenbereich
(Seed): beim Permalink wird jeder Wert auf den Bereich begrenzt bzw. auf die nächste Stufe eingerastet, damit die Adresszeile nie einen Wert außerhalb des Rasters in den Regler schreibt."""
import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import bw_constants as C


def _text(value):
    return str(value)


def _num(value):
    return f"{value:g}"


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None
    options: Optional[tuple] = None          # feste Stufen: auf die nächste Stufe einrasten (Zahlen) bzw. genau übereinstimmen (Text)
    encoder: Callable = _text


SETTING_SPECS = {
    "cv_slider": SettingSpec("cv", float, C.CV_DEFAULT, options=C.CV_OPTIONS, encoder=_num),
    "rho_slider": SettingSpec("rho", float, C.RHO_DEFAULT, options=C.RHO_OPTIONS, encoder=_num),
    "lead_slider": SettingSpec("l", int, C.LEAD_DEFAULT, options=C.LEAD_OPTIONS),
    "window_slider": SettingSpec("p", int, C.WINDOW_DEFAULT, options=C.WINDOW_OPTIONS),
    "gamma_slider": SettingSpec("g", float, C.GAMMA_DEFAULT, options=C.GAMMA_OPTIONS, encoder=_num),
    "z_mode": SettingSpec("z", str, C.ZMODE_DEFAULT, options=C.ZMODE_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.SEED_DEFAULT, *C.SEED_RANGE),
}

PRESET_STATE_KEYS = {"cv": "cv_slider", "rho": "rho_slider", "lead": "lead_slider", "window": "window_slider", "gamma": "gamma_slider", "zmode": "z_mode", "seed": "seed_input"}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def parse_setting(spec, raw):
    """Wert aus der Adresszeile: umwandeln, auf den Bereich begrenzen bzw. auf die nächste Stufe einrasten. None, wenn er sich nicht auswerten lässt (kein Zahlenwert, nicht endlich, unbekannter Text)."""
    if spec.caster is str:
        return raw if raw in spec.options else None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    if spec.options:                                        # Zahlenstufen: die nächste Stufe, bei Gleichstand die kleinere
        return min(spec.options, key=lambda o: (abs(o - value), o))
    return int(round(max(spec.lo, min(spec.hi, value))))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = parse_setting(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """values: dict state_key -> aktueller Wert (aus den Widgets)."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = SETTING_SPECS[state_key].encoder(value)
    except Exception:
        pass


def apply_preset(name):
    for field, state_key in PRESET_STATE_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][field]


def randomize_seed():
    """Würfelt einen neuen Seed für den gezeigten Verlauf."""
    st.session_state["seed_input"] = random.randint(*C.SEED_RANGE)
