"""Friert Nachfragereihen und Referenzwerte für die Tests ein (nur lokal, braucht die Messreihe messreihe_bullwhip/bull.py):

  tests/data/bw_reference.json     11 Fälle (Nachfragereihen R x T verlustfrei als float64, Kettenkonfiguration, z) mit den Ergebnissen von bull.py (Bestellungen, Lagerführung, Verstärkung)
                                   und zwei Kalibrierläufen. Beim Einfrieren wird ZUSÄTZLICH bitgleich geprüft, dass bw_model dasselbe liefert wie bull.py (die mechanische Übernahme).
  tests/data/bw_preset_demand.json die Nachfragereihen der Preset-Verläufe (Seed aus bw_constants.PRESETS, Länge 400) für die qualitativen Tageskriterien.

Warum: die CI installiert immer das neueste NumPy, und die Ströme von default_rng dürfen sich zwischen Versionen ändern; die Referenzwerte hängen deshalb an EINGEFRORENEN Reihen, nicht an
gen_ar1. Aufruf (im Projektordner):  python tools/freeze_reference.py [Ordner_der_Messreihe]   (Standard ../bestand-planung/messreihe_bullwhip)"""
import base64
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    sys.path.insert(0, str(p))

import bw_constants as C  # noqa: E402
import bw_model as BW  # noqa: E402
import bw_theory as TH  # noqa: E402

DEFAULT_SOURCES = ROOT.parent / "bestand-planung" / "messreihe_bullwhip"
OUT_DIR = ROOT / "tests" / "data"
BURN = 50
DEMANDS = {                                  # Name -> (Art, R, T, mu, sigma, rho, Seed)
    "iid": ("ar1", 3, 300, 100.0, 20.0, 0.0, 4101), "rho0.8": ("ar1", 3, 300, 100.0, 20.0, 0.8, 4102), "shift": ("shift", 2, 300, 100.0, 20.0, None, 4103),
    "cal": ("ar1", 6, 350, 100.0, 20.0, 0.0, 4104), "long": ("ar1", 1, 400, 100.0, 20.0, 0.0, 4105),
}
CASES = [                                    # (Name, Nachfrage, Ls, p, z, order_streams-Schalter)
    ("basis", "iid", [2] * 4, 10, [1.07, 1.33, 1.66, 1.84], {}),
    ("gamma0.5", "iid", [2] * 4, 10, [1.25, 2.27, 3.44, 3.36], {"gamma": 0.5}),
    ("einzelhandel0.5", "iid", [2] * 4, 10, [1.27, 1.8, 2.23, 2.23], {"gammas": [0.5, 1, 1, 1]}),
    ("info", "iid", [2] * 4, 10, [1.05, 1.48, 2.07, 2.38], {"share": True}),
    ("info_gamma0.5", "iid", [2] * 4, 10, [1.19, 1.39, 1.66, 1.19], {"share": True, "gamma": 0.5}),
    ("lead1", "iid", [1] * 4, 10, [0.88, 0.96, 1.09, 0.88], {}),
    ("window20", "iid", [2] * 4, 20, [0.98, 1.07, 1.17, 1.04], {}),
    ("linear", "iid", [2] * 4, 10, 0.0, {"nonneg": False}),
    ("unequal", "iid", [1, 3, 2, 4], 6, 1.645, {}),
    ("rho0.8", "rho0.8", [2] * 4, 10, [2.27, 3.0, 3.5, 4.08], {}),
    ("shift", "shift", [2] * 4, 10, 1.645, {}),
]
CALIBRATIONS = [("basis", {}), ("info", {"share": True})]
CAL_KW = dict(target=0.95, burn=BURN, passes=1, iters=6, zmax=10.0)


def enc(a):
    a = np.ascontiguousarray(a, dtype="<f8")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def make_demand(mod, spec):
    kind, R, T, mu, sigma, rho, seed = spec
    rng = np.random.default_rng(seed)
    return mod.gen_ar1(rng, R, T, mu, sigma, rho) if kind == "ar1" else mod.gen_shift(rng, R, T, mu, sigma)


def evaluate(mod, D, Ls, p, z, kw):
    q, S, IPb = mod.order_streams(D, Ls, p, z=z, **kw)
    ph = mod.physical(D, q, Ls, burn=BURN)
    vr = mod.var_ratio(q, D, BURN)
    return q, ph, vr


def case_record(name, dname, D, Ls, p, z, kw, q, ph, vr):
    n, r, t = q.shape
    return {"name": name, "demand": dname, "Ls": Ls, "p": p, "z": z, "kw": kw,
            "expected": {"q_sum": q.sum(axis=2).tolist(), "q_tail": q[:, :, -5:].tolist(), "q_zero_count": int((q <= 1e-12).sum()),
                         "cost": ph["cost"].tolist(), "alpha": ph["alpha"].tolist(), "beta": ph["beta"].tolist(), "onhand": ph["onhand"].tolist(), "backlog": ph["backlog"].tolist(),
                         "cust_shipped": ph["cust_shipped"].tolist(), "goods_end": ph["goods_end"].tolist(), "amp": [a for a, _ in vr]}}


def main(argv):
    src = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCES
    sys.path.insert(0, str(src))
    import bull  # noqa: E402  (die Messreihe: Referenz)
    demands = {name: make_demand(BW, spec) for name, spec in DEMANDS.items()}
    for name, spec in DEMANDS.items():                                      # dieselben Ströme in bull.py
        assert np.array_equal(demands[name], make_demand(bull, spec)), name
    cases, mism = [], 0
    for name, dname, Ls, p, z, kw in CASES:
        D = demands[dname]
        qb, phb, vrb = evaluate(bull, D, Ls, p, z, kw)
        qw, phw, vrw = evaluate(BW, D, Ls, p, z, kw)
        same = np.array_equal(qb, qw) and all(np.array_equal(phb[k], phw[k]) for k in phb) and vrb == vrw
        mism += 0 if same else 1
        cases.append(case_record(name, dname, D, Ls, p, z, kw, qb, phb, vrb))
    assert mism == 0, f"{mism} Fälle: bw_model weicht von bull.py ab"
    cals = []
    for name, kw in CALIBRATIONS:
        zb, qb, phb = bull.calibrate_z(demands["cal"], [2] * 4, 10, **CAL_KW, **kw)
        zw, qw, phw = BW.calibrate_z(demands["cal"], [2] * 4, 10, **CAL_KW, **kw)
        assert zb == zw and np.array_equal(qb, qw), name
        cals.append({"name": name, "demand": "cal", "Ls": [2] * 4, "p": 10, "kw": kw, "cal_kw": CAL_KW, "z": zb, "beta": phb["beta"].mean(axis=1).tolist()})
    # Linearer Fall: exakte Faltung (unabhängig von bull.py) - nur zur Kontrolle beim Einfrieren
    D = demands["iid"]
    q, _, _ = BW.order_streams(D, [2] * 4, 10, z=0.0, nonneg=False)
    for k, h in enumerate(TH.chain_filters([2] * 4, [10] * 4)):
        y = np.array([np.convolve(D[r], h)[:D.shape[1]] for r in range(D.shape[0])])
        assert np.abs(y[:, 41:] - q[k][:, 41:]).max() < 1e-8
    ref = {"numpy": np.__version__, "platform": sys.platform, "burn": BURN, "source": "messreihe_bullwhip/bull.py",
           "demands": {n: enc(d) for n, d in demands.items()}, "specs": {n: list(s) for n, s in DEMANDS.items()}, "cases": cases, "calibrations": cals}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "bw_reference.json").write_text(json.dumps(ref, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    # Preset-Verläufe: Länge LIVE_T, Seed der Presets, je (CV, rho) der Presets
    seed = C.PRESETS["Standard"]["seed"]
    series = {}
    for p in C.PRESETS.values():
        assert p["seed"] == seed
        key = f"{p['cv']}/{p['rho']}"
        if key not in series:
            series[key] = enc(BW.gen_ar1(np.random.default_rng(seed), 1, C.LIVE_T, C.MU, C.MU * p["cv"], p["rho"]))
    (OUT_DIR / "bw_preset_demand.json").write_text(json.dumps({"numpy": np.__version__, "seed": seed, "series": series}, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"eingefroren: {len(cases)} Fälle, {len(cals)} Kalibrierungen, {len(series)} Preset-Reihen (Seed {seed}); bw_model == bull.py bitgleich; NumPy {np.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
