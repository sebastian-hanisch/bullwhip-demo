"""Bitgleichheit zur Messreihe: die mechanisch aus bull.py abgeleiteten Module (bw_model, bw_theory) müssen auf den EINGEFRORENEN Nachfragereihen (tests/data/bw_reference.json, erzeugt von
tools/freeze_reference.py aus den Ergebnissen von bull.py) dieselben Kennzahlen liefern: 11 Fälle (alle Maßnahmen, ungleiche Wiederbeschaffungszeiten, lineare Kette, ρ = 0,8, Niveausprünge) und
zwei Kalibrierläufe.

Die Reihen sind eingefroren (Bedarf R x T verlustfrei als float64), damit kein Test vom Zufallsgenerator abhängt: die CI installiert immer das neueste NumPy, und die Ströme von default_rng dürfen
sich zwischen Versionen ändern. Ganzzahlige Zählungen müssen immer exakt stimmen; die Gleitkommasummen dürfen sich auf einer anderen Plattform in der letzten Stelle unterscheiden (NumPy
summiert Felder je nach Version paarweise oder blockweise): dort gilt ein relativer Abstand von 1e-9. Auf derselben Plattform und NumPy-Version, mit der eingefroren wurde, gilt strikte Gleichheit."""
import sys

import numpy as np
import pytest

import bw_frozen as FZ
import bw_model as M

REF = FZ.reference()
CASES = REF["cases"]
SAME_PLATFORM = np.__version__ == REF["numpy"] and sys.platform == REF["platform"]
BURN = REF["burn"]


def run(case):
    D = FZ.demand(case["demand"])
    q, _, _ = M.order_streams(D, case["Ls"], case["p"], z=case["z"], **case["kw"])
    ph = M.physical(D, q, case["Ls"], burn=BURN)
    vr = M.var_ratio(q, D, BURN)
    return D, q, ph, vr


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_chain_reproduces_the_frozen_measurement_series_results(case):
    D, q, ph, vr = run(case)
    e = case["expected"]
    np.testing.assert_allclose(q.sum(axis=2), e["q_sum"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q[:, :, -5:], e["q_tail"], rtol=1e-9, atol=1e-9)
    assert int((q <= 1e-12).sum()) == e["q_zero_count"]                                      # Zählung über die Rauschgrenze 1e-12: immer exakt
    for key in ("cost", "alpha", "beta", "onhand", "backlog", "cust_shipped", "goods_end"):
        np.testing.assert_allclose(ph[key], e[key], rtol=1e-9, atol=1e-9, err_msg=key)
    np.testing.assert_allclose([a for a, _ in vr], e["amp"], rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not SAME_PLATFORM, reason="strikte Gleichheit nur mit derselben NumPy-Version und Plattform wie beim Einfrieren")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_chain_is_bit_identical_on_the_freezing_platform(case):
    D, q, ph, vr = run(case)
    e = case["expected"]
    assert q.sum(axis=2).tolist() == e["q_sum"] and q[:, :, -5:].tolist() == e["q_tail"]
    for key in ("cost", "alpha", "beta", "onhand", "backlog", "cust_shipped", "goods_end"):
        assert ph[key].tolist() == e[key], key
    assert [a for a, _ in vr] == e["amp"]


@pytest.mark.parametrize("cal", REF["calibrations"], ids=[c["name"] for c in REF["calibrations"]])
def test_calibration_reproduces_the_frozen_z_and_service(cal):
    D = FZ.demand(cal["demand"])
    zs, q, ph = M.calibrate_z(D, cal["Ls"], cal["p"], **cal["cal_kw"], **cal["kw"])
    np.testing.assert_allclose(zs, cal["z"], rtol=0, atol=1e-12)                       # Gitterpunkte der Bisektion
    np.testing.assert_allclose(ph["beta"].mean(axis=1), cal["beta"], rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(np.__version__ != REF["numpy"], reason="der Zufallsstrom kann sich zwischen NumPy-Versionen ändern: gen_ar1 nur mit der Einfrier-Version bitgleich prüfen")
@pytest.mark.parametrize("name", list(REF["specs"]))
def test_gen_reproduces_the_frozen_demand_with_the_same_numpy(name):
    kind, R, T, mu, sigma, rho, seed = REF["specs"][name]
    rng = np.random.default_rng(seed)
    D = M.gen_ar1(rng, R, T, mu, sigma, rho) if kind == "ar1" else M.gen_shift(rng, R, T, mu, sigma)
    assert np.array_equal(D, FZ.demand(name))


def test_the_reference_covers_every_measure_and_the_case_variety():
    names = [c["name"] for c in CASES]
    assert names == ["basis", "gamma0.5", "einzelhandel0.5", "info", "info_gamma0.5", "lead1", "window20", "linear", "unequal", "rho0.8", "shift"]
    kws = [c["kw"] for c in CASES]
    assert {"gamma": 0.5} in kws and {"gammas": [0.5, 1, 1, 1]} in kws and {"share": True} in kws and {"share": True, "gamma": 0.5} in kws and {"nonneg": False} in kws
    assert {tuple(c["Ls"]) for c in CASES} == {(2, 2, 2, 2), (1, 1, 1, 1), (1, 3, 2, 4)} and {c["p"] for c in CASES} == {6, 10, 20}
    for name in REF["demands"]:
        D = FZ.demand(name)
        assert D.ndim == 2 and D.shape == tuple(REF["demands"][name]["shape"]) and np.isfinite(D).all()


def test_the_frozen_cases_pin_down_the_measured_mechanism():
    """Auf den eingefrorenen Reihen zeigt sich der Mechanismus: Glättung und Info teilen senken die Verstärkung an der Fabrik gegenüber der Basis."""
    e = {c["name"]: c["expected"] for c in CASES}
    amp4 = {k: float(np.mean(v["amp"][3:])) for k, v in e.items()}
    assert amp4["gamma0.5"] < 0.6 * amp4["basis"] and amp4["info"] < 0.9 * amp4["basis"] and amp4["einzelhandel0.5"] < amp4["basis"]


def test_the_exact_service_target_of_the_frozen_calibrations_is_met():
    for cal in REF["calibrations"]:
        assert np.allclose(cal["beta"], 0.95, atol=0.02), cal["name"]
        assert all(0.0 < z < 10.0 for z in cal["z"])
