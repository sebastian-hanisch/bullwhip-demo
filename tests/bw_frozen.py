"""Test-Hilfsmodul: die eingefrorenen Nachfragereihen und Referenzwerte (tests/data/bw_reference.json, bw_preset_demand.json) laden.

Eine eingefrorene Nachfragereihe enthält den kompletten Bedarf (R x T) verlustfrei (float64 in Base64): kein Test, der sie benutzt, hängt vom Zufallsgenerator oder von der NumPy-Version
ab (die Ströme von default_rng dürfen sich zwischen Versionen ändern). Die Referenzwerte stammen aus messreihe_bullwhip/bull.py (tools/freeze_reference.py)."""
import base64
import functools
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent / "data"


def enc(a):
    a = np.ascontiguousarray(a, dtype="<f8")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def dec(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype="<f8").reshape(d["shape"]).copy()


@functools.lru_cache(maxsize=1)
def reference():
    return json.loads((DATA / "bw_reference.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def preset_demand():
    return json.loads((DATA / "bw_preset_demand.json").read_text(encoding="utf-8"))


def demand(name):
    """Die eingefrorene Nachfragereihe `name` (Form R x T)."""
    return dec(reference()["demands"][name])


def preset_series(cv, rho):
    """Die eingefrorene Nachfragereihe des Preset-Verlaufs (Form 1 x T) für CV und rho, erzeugt mit dem Preset-Seed (tools/freeze_reference.py)."""
    return dec(preset_demand()["series"][f"{cv}/{rho}"])
