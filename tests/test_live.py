"""Live-Verlauf (bw_live): Nachfrage, Maßnahmen-Aufbau, Lagerverlauf je Periode gegen bw_model.physical, Kontrollzeile gegen die exakte Theorie, z-Zuordnung (vorgerechnet, fest, live),
Schnellkalibrierung (mit Zeitlimit über eine Ersatzuhr) und Pause. Referenzwerte laufen über EINGEFRORENE Nachfragereihen (die CI installiert das neueste NumPy); die Schnellkalibrierung
selbst würfelt ihre Stichprobe und wird deshalb nur auf Eigenschaften geprüft."""
import itertools
import math

import numpy as np
import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_model as M
import bw_results as R
import bw_theory as T

DATA = R.load_results()
PRESET_D = FZ.preset_series(0.2, 0.0)                     # Preset-Verlauf: Seed 202, CV 0,2, ρ 0, 400 Perioden
SEED = C.PRESETS["Standard"]["seed"]


def live(**over):
    p = dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=C.ZMODE_EQUAL, seed=SEED, z_override=None, demand=PRESET_D)
    p.update(over)
    return LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], p["seed"], z_override=p["z_override"], demand=p["demand"])


# ---------------------------------------------------------------------------------------------------
# Nachfrage
# ---------------------------------------------------------------------------------------------------
def test_gen_demand_asks_the_generator_for_one_replication_of_400_periods(monkeypatch):
    seen = {}

    def fake(rng, R_, T_, mu, sigma, rho):
        seen.update(R=R_, T=T_, mu=mu, sigma=sigma, rho=rho)
        return np.zeros((R_, T_))
    monkeypatch.setattr(LV.M, "gen_ar1", fake)
    D = LV.gen_demand(0.3, 0.5, 17)
    assert D.shape == (1, 400) and seen == dict(R=1, T=400, mu=100.0, sigma=30.0, rho=0.5)
    assert LV.gen_demand(0.1, 0.0, 17, periods=50).shape == (1, 50)


def test_gen_demand_is_reproducible_per_seed_and_differs_between_seeds():
    a, b, c = LV.gen_demand(0.2, 0.0, 5), LV.gen_demand(0.2, 0.0, 5), LV.gen_demand(0.2, 0.0, 6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    means = [LV.gen_demand(0.2, 0.0, s).mean() for s in range(60)]
    assert abs(np.mean(means) - 100.0) < 1.0 and 1.0 < np.std(means) < 4.0                        # Mittel der Mittel, Streuung ≈ 20 / sqrt(400)


def test_the_frozen_preset_series_has_the_live_length_and_plausible_moments():
    assert PRESET_D.shape == (1, C.LIVE_T) and abs(PRESET_D.mean() - 100.0) < 6.0 and 14.0 < PRESET_D.std() < 26.0
    assert FZ.preset_series(0.2, 0.8).shape == (1, C.LIVE_T)


# ---------------------------------------------------------------------------------------------------
# Maßnahmen
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("measure,expect", [
    (C.M_BASIS, ([2] * 4, 10, dict(gamma=1.0, share=False, gammas=None))),
    (C.M_SMOOTH, ([2] * 4, 10, dict(gamma=0.5, share=False, gammas=None))),
    (C.M_RETAIL, ([2] * 4, 10, dict(gamma=1.0, share=False, gammas=[0.5, 1.0, 1.0, 1.0]))),
    (C.M_INFO, ([2] * 4, 10, dict(gamma=1.0, share=True, gammas=None))),
    (C.M_INFO_SMOOTH, ([2] * 4, 10, dict(gamma=0.5, share=True, gammas=None))),
    (C.M_LEAD, ([1] * 4, 10, dict(gamma=1.0, share=False, gammas=None))),
    (C.M_WINDOW, ([2] * 4, 20, dict(gamma=1.0, share=False, gammas=None))),
])
def test_measure_setup(measure, expect):
    assert LV.measure_setup(measure, 0.5, 2, 10) == expect


def test_measure_setup_uses_the_effective_lead_and_window():
    assert LV.measure_setup(C.M_LEAD, 0.5, 4, 10)[:2] == ([2] * 4, 10) and LV.measure_setup(C.M_WINDOW, 0.5, 2, 6)[:2] == ([2] * 4, 12) and LV.measure_setup(C.M_SMOOTH, 0.3, 1, 40)[2]["gamma"] == 0.3


def test_physical_series_reproduces_the_means_of_physical_for_every_lead_configuration():
    for name, Ls in (("iid", [2] * 4), ("iid", [1] * 4), ("iid", [1, 3, 2, 4]), ("rho0.8", [4] * 4)):
        D = FZ.demand(name)
        q, _, _ = M.order_streams(D, Ls, 6, z=1.2)
        ph = M.physical(D, q, Ls, burn=50)
        oh, bl = LV.physical_series(D, q, Ls)
        assert oh.shape == q.shape and bl.shape == q.shape and (oh >= -1e-9).all() and (bl >= -1e-9).all()
        np.testing.assert_allclose(oh[:, :, 50:].mean(axis=2), ph["onhand"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bl[:, :, 50:].mean(axis=2), ph["backlog"], rtol=1e-12, atol=1e-12)
        assert ((oh > 1e-9) & (bl > 1e-9)).sum() == 0                                             # Lager und Rückstand gleichzeitig gibt es nicht


def test_physical_series_hand_instance():
    D = np.full((1, 5), 10.0)
    q = np.zeros((1, 1, 5))
    oh, bl = LV.physical_series(D, q, [1])
    assert oh[0, 0].tolist() == [0.0] * 5 and bl[0, 0].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_amplification_equals_the_mean_of_var_ratio_on_one_replication():
    D = FZ.demand("long")
    q, _, _ = M.order_streams(D, [2] * 4, 10, z=1.2)
    amp = LV.amplification(q, D)
    assert amp == pytest.approx([a for a, _ in _var_ratio_no_se(q, D)])
    assert all(a > 0 for a in amp) and amp[3] > amp[0]


def _var_ratio_no_se(q, D, burn=100):
    vd = D[:, burn:].var(axis=1, ddof=1)
    return [(float((q[k, :, burn:].var(axis=1, ddof=1) / vd).mean()), 0.0) for k in range(q.shape[0])]


def test_run_measure_summarises_the_evaluated_periods_only():
    D = PRESET_D
    z = DATA["jobs"]["main/basis"]["z"]
    r = LV.run_measure(D, C.M_BASIS, 0.5, 2, 10, z)
    assert r["q"].shape == (4, 400) and r["onhand"].shape == (4, 400) and r["backlog"].shape == (4, 400) and r["Ls"] == [2] * 4 and r["p"] == 10 and r["label"] == "Basis"
    assert r["total_onhand"] == pytest.approx(sum(r["onhand_mean"])) and r["total_cost"] == pytest.approx(sum(r["cost"]))
    assert r["onhand_mean"] == pytest.approx(r["onhand"][:, 100:].mean(axis=1).tolist()) and r["backlog_mean"] == pytest.approx(r["backlog"][:, 100:].mean(axis=1).tolist())
    assert r["zero_share"] == pytest.approx(((r["q"][:, 100:]) <= 1e-12).mean(axis=1).tolist()) and r["z"] == pytest.approx(z)
    assert all(0.5 < b <= 1.0 for b in r["beta"]) and r["cost"][0] == pytest.approx(1.0 * r["onhand_mean"][0] + 9.0 * r["backlog_mean"][0])


def test_frozen_live_values_for_the_standard_preset():
    """Eingefrorene Referenz (Seed 202, Standard, z aus der Messreihe): Summe Lager und Verstärkung Stufe 4 je Maßnahme."""
    r = live()
    want = {"basis": (284.22579059713195, 10.37390308012885), "glaettung": (239.2945518722037, 2.9224701292803617), "einzelhandel": (229.79095609017486, 5.592187395106255),
            "info": (208.4866581965907, 4.979736368242261), "info_glaettung": (163.27260759093312, 0.989905602255774), "lead_halved": (98.81704927717936, 3.1733516949550964),
            "window_doubled": (156.77187488646896, 2.8833026306453613)}
    for m, (tot, amp4) in want.items():
        assert r["measures"][m]["total_onhand"] == pytest.approx(tot, rel=1e-9) and r["measures"][m]["amp"][3] == pytest.approx(amp4, rel=1e-9), m
    assert r["measures"]["basis"]["beta"][0] == pytest.approx(0.957305688994297, rel=1e-9)


def test_frozen_live_values_fixed_z_with_slow_demand():
    r = live(rho=0.8, zmode=C.ZMODE_FIXED, demand=FZ.preset_series(0.2, 0.8))
    want = {"basis": 254.9728274076511, "glaettung": 249.83775689717504, "einzelhandel": 232.14006375358315, "info": 166.28482815397422, "info_glaettung": 180.56054657292958,
            "lead_halved": 131.32031945432448, "window_doubled": 234.20932990194558}
    for m, tot in want.items():
        assert r["measures"][m]["total_onhand"] == pytest.approx(tot, rel=1e-9), m


# ---------------------------------------------------------------------------------------------------
# solve_live
# ---------------------------------------------------------------------------------------------------
def test_solve_live_returns_all_seven_measures_and_the_setting():
    r = live()
    assert list(r["measures"]) == list(C.MEASURES) and r["active"] == list(C.MEASURES) and r["T"] == 400 and r["burn"] == 100 and r["seed"] == SEED
    assert (r["cv"], r["rho"], r["lead"], r["window"], r["gamma"], r["zmode"]) == (0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL) and r["D"].shape == (400,) and np.array_equal(r["D"], PRESET_D[0])
    for m, v in r["measures"].items():
        assert v["measure"] == m and len(v["z"]) == 4 and v["z_exact"] and v["z_source"] == "vorgerechnet" and v["z_note"] == ""


def test_every_switch_changes_the_fabric_amplification_against_the_basis_by_more_than_one_percent():
    """Zweig-Test (Nullspalte): jede Maßnahme wirkt in jeder Einstellung, in der sie aktiv ist."""
    for lead, window in ((2, 10), (4, 10), (2, 6), (2, 20), (1, 10), (2, 40)):
        r = live(lead=lead, window=window)
        base = r["measures"][C.M_BASIS]["amp"][3]
        for m in r["active"]:
            if m != C.M_BASIS:
                assert abs(r["measures"][m]["amp"][3] / base - 1) > 0.01, (m, lead, window)


def test_smoothing_only_in_the_retail_stage_leaves_stage_one_of_the_others_untouched():
    # Stufe 1: mit Info teilen unverändert gegen die Basis (dieselbe Eingangsreihe); bei gleichem z sind die Bestellungen der Stufe 1 bitgleich
    a = LV.run_measure(PRESET_D, C.M_BASIS, 0.5, 2, 10, [1.2] * 4)
    b = LV.run_measure(PRESET_D, C.M_INFO, 0.5, 2, 10, [1.2] * 4)
    assert np.array_equal(a["q"][0], b["q"][0]) and not np.array_equal(a["q"][1], b["q"][1])
    c = LV.run_measure(PRESET_D, C.M_RETAIL, 0.5, 2, 10, [1.2] * 4)
    assert not np.array_equal(a["q"][0], c["q"][0])


def test_hidden_measures_are_not_computed():
    r = live(lead=1)
    assert C.M_LEAD not in r["measures"] and r["active"] == [C.M_BASIS, C.M_SMOOTH, C.M_RETAIL, C.M_INFO, C.M_INFO_SMOOTH, C.M_WINDOW]
    r = live(window=40)
    assert C.M_WINDOW not in r["measures"] and C.M_LEAD in r["measures"]
    r = live(lead=1, window=40)
    assert len(r["measures"]) == 5


def test_equal_service_takes_z_from_the_measurement_series_and_says_when_it_is_not_exact():
    r = live(rho=0.5, lead=4, cv=0.3)                                                       # keine gemessene Zelle
    assert r["notes"] and any(not v["z_exact"] for v in r["measures"].values())
    inexact = [v for v in r["measures"].values() if not v["z_exact"]]
    assert all(v["z_source"] == "vorgerechnet (nächstliegende Konfiguration)" and v["z_note"] for v in inexact) and len(r["notes"]) == len(inexact)
    for m, v in r["measures"].items():
        want = R.resolve_z(DATA, m, 0.3, 0.5, 4, 10, 0.5)
        assert v["z"] == pytest.approx(want["z"]) and v["z_exact"] == want["exact"]


def test_the_configuration_behind_each_z_is_named():
    r = live()
    assert r["measures"][C.M_BASIS]["z_config"] == "CV 0,2, ρ 0,0, L 2, p 10" and r["measures"][C.M_RETAIL]["z_config"] == "CV 0,2, ρ 0,0, L 2, p 10, γ 0,5" and r["measures"][C.M_LEAD]["z_config"] == "CV 0,2, ρ 0,0, L 1, p 10"
    assert live(zmode=C.ZMODE_FIXED)["measures"][C.M_BASIS]["z_config"] == "–" and live(z_override={C.M_BASIS: [1.0] * 4})["measures"][C.M_BASIS]["z_config"] == "20 Wiederholungen × 700 Perioden"


def test_fixed_service_mode_uses_the_same_z_everywhere():
    r = live(zmode=C.ZMODE_FIXED)
    for v in r["measures"].values():
        assert v["z"] == [1.645] * 4 and v["z_exact"] and v["z_source"] == "fest 1,645" and v["z_note"] == ""
    assert r["notes"] == []


def test_a_live_calibration_overrides_the_measurement_series_for_its_measures_only():
    z = [0.5, 0.6, 0.7, 0.8]
    r = live(z_override={C.M_SMOOTH: z})
    assert r["measures"][C.M_SMOOTH]["z"] == z and r["measures"][C.M_SMOOTH]["z_source"] == "live nachkalibriert (grob)" and not r["measures"][C.M_SMOOTH]["z_exact"]
    assert r["measures"][C.M_BASIS]["z_source"] == "vorgerechnet" and r["measures"][C.M_BASIS]["z"] == DATA["jobs"]["main/basis"]["z"]
    fixed = live(zmode=C.ZMODE_FIXED, z_override={C.M_SMOOTH: z})
    assert fixed["measures"][C.M_SMOOTH]["z"] == z                                            # der Aufrufer reicht die Live-Kalibrierung nur im Modus "kalibriert" durch


def test_more_service_costs_more_stock():
    lo = LV.run_measure(PRESET_D, C.M_BASIS, 0.5, 2, 10, [0.5] * 4)
    hi = LV.run_measure(PRESET_D, C.M_BASIS, 0.5, 2, 10, [2.5] * 4)
    assert hi["total_onhand"] > lo["total_onhand"] and min(hi["beta"]) > max(lo["beta"]) - 0.02 and hi["amp"][3] > lo["amp"][3]


# ---------------------------------------------------------------------------------------------------
# Kontrollzeile
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lead,window,rho", [(2, 10, 0.0), (1, 6, 0.5), (4, 20, 0.8), (2, 40, 0.0)])
def test_control_row_is_the_exact_linear_theory(lead, window, rho):
    D = M.gen_ar1(np.random.default_rng(11), 1, 400, 100.0, 20.0, rho) if window < 40 else FZ.demand("long")
    c = LV.control_row(D, lead, window, rho)
    assert c["max_deviation"] < 1e-8 and c["start"] == 1 + 4 * window
    hs = T.chain_filters([lead] * 4, [window] * 4)
    assert c["amp_exact"] == pytest.approx([T.ratio_from_filter(h, rho) for h in hs])
    assert c["amp_product"] == pytest.approx([T.chen_single_bound(lead, window) ** (k + 1) for k in range(4)])
    assert all(a > 0 for a in c["amp_run"]) and c["amp_exact"][3] >= c["amp_product"][3]


def test_control_row_detects_a_broken_chain():
    """Die Kontrolle ist scharf: mit Sicherheitsbestand (z > 0) ist die Kette nicht mehr linear, die Abweichung von der Faltung wäre groß."""
    D = PRESET_D
    q, _, _ = M.order_streams(D, [2] * 4, 10, z=1.0, nonneg=False)
    h = T.chain_filters([2] * 4, [10] * 4)[3]
    y = np.convolve(D[0], h)[:400]
    assert np.abs(y[41:] - q[3, 0, 41:]).max() > 1.0
    assert LV.control_row(D, 2, 10, 0.0)["max_deviation"] < 1e-8


def test_the_live_result_carries_the_control_row():
    r = live()
    assert r["control"]["max_deviation"] < 1e-8 and len(r["control"]["amp_run"]) == 4


# ---------------------------------------------------------------------------------------------------
# Schnellkalibrierung und Pause
# ---------------------------------------------------------------------------------------------------
def test_quick_calibration_returns_a_z_per_stage_for_each_measure_on_a_small_sample():
    res = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS, C.M_INFO], reps=4, periods=250)
    assert list(res["z"]) == [C.M_BASIS, C.M_INFO] and res["skipped"] == [] and res["seconds"] >= 0
    for z in res["z"].values():
        assert len(z) == 4 and all(0.0 < v <= C.QUICK_ZMAX for v in z) and all(isinstance(v, float) for v in z)


def test_quick_calibration_is_reproducible_and_depends_on_the_seed():
    a = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS], reps=4, periods=250)["z"]
    b = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS], reps=4, periods=250)["z"]
    c = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 8, [C.M_BASIS], reps=4, periods=250)["z"]
    assert a == b and a != c


def test_quick_calibration_passes_the_documented_settings_to_the_calibration(monkeypatch):
    calls = []

    def fake(D, Ls, p, **kw):
        calls.append((D.shape, list(Ls), p, kw))
        return [1.0] * 4, None, None
    monkeypatch.setattr(LV.M, "calibrate_z", fake)
    LV.quick_calibrate(0.3, 0.5, 4, 20, 0.7, 3, [C.M_SMOOTH, C.M_LEAD, C.M_RETAIL])
    assert [c[0] for c in calls] == [(20, 700)] * 3 and [c[1] for c in calls] == [[4] * 4, [2] * 4, [4] * 4] and [c[2] for c in calls] == [20, 20, 20]
    kw = calls[0][3]
    assert kw["target"] == 0.95 and kw["burn"] == 100 and kw["passes"] == 1 and kw["iters"] == 9 and kw["zmax"] == 10.0 and kw["gamma"] == 0.7 and kw["share"] is False and kw["gammas"] is None
    assert calls[2][3]["gammas"] == [0.7, 1.0, 1.0, 1.0] and calls[1][3]["gamma"] == 1.0


def test_quick_calibration_uses_one_common_sample_independent_of_the_shown_seed_offset(monkeypatch):
    samples = []
    monkeypatch.setattr(LV.M, "calibrate_z", lambda D, Ls, p, **kw: (samples.append(D.copy()) or ([1.0] * 4, None, None)))
    LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 5, [C.M_BASIS, C.M_SMOOTH], reps=3, periods=120)
    assert np.array_equal(samples[0], samples[1])
    expect = M.gen_ar1(np.random.default_rng(5 + C.QUICK_SEED_OFFSET), 3, 120, 100.0, 20.0, 0.0)
    assert np.array_equal(samples[0], expect)


def test_quick_calibration_stops_at_the_time_limit_and_reports_the_skipped_measures():
    ticks = iter([0.0, 1.0, 200.0, 201.0, 202.0, 203.0])
    res = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS, C.M_SMOOTH, C.M_INFO], time_limit=50.0, clock=lambda: next(ticks), reps=3, periods=200)
    assert list(res["z"]) == [C.M_BASIS] and res["skipped"] == [C.M_SMOOTH, C.M_INFO] and res["seconds"] == 202.0


def test_quick_calibration_reports_progress():
    seen = []
    LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS, C.M_INFO], reps=3, periods=200, on_progress=lambda i, n, m: seen.append((i, n, m)))
    assert seen == [(0, 2, C.M_BASIS), (1, 2, C.M_INFO)]


def test_the_quick_calibration_is_rough_but_close_to_the_full_calibration_on_the_main_scenario():
    """AP 0 (c): z aus 20 x 700 liegt nahe an der vollen Kalibrierung (Basis 1,07 / 1,33 / 1,66 / 1,84): höchstens 0,35 Abstand je Stufe."""
    res = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 300, [C.M_BASIS])
    full = DATA["jobs"]["main/basis"]["z"]
    assert all(abs(a - b) < 0.35 for a, b in zip(res["z"][C.M_BASIS], full)), (res["z"][C.M_BASIS], full)


def test_cooldown():
    assert LV.cooldown_left(None, 100.0) == 0.0 and LV.cooldown_left(100.0, 101.0) == pytest.approx(C.QUICK_COOLDOWN - 1.0)
    assert LV.cooldown_left(100.0, 100.0 + C.QUICK_COOLDOWN) == 0.0 and LV.cooldown_left(100.0, 200.0) == 0.0 and LV.cooldown_left(100.0, 102.0, cooldown=10.0) == 8.0


def test_the_live_calculation_of_all_measures_is_fast_enough_to_run_on_every_change():
    """Keine Wall-Clock-Assertion: nur die Zahl der Kettenläufe ist begrenzt (sieben Maßnahmen, je ein Lauf, plus ein Kontrolllauf)."""
    calls = []
    orig = LV.M.order_streams
    LV.M.order_streams = lambda *a, **k: (calls.append(1) or orig(*a, **k))
    try:
        live()
    finally:
        LV.M.order_streams = orig
    assert len(calls) == 7 + 1


def test_quick_calibration_measures_seconds_from_its_own_start_and_stops_strictly_after_the_limit():
    ticks = iter([10.0, 11.0, 210.0, 211.0, 212.0])
    res = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS, C.M_SMOOTH, C.M_INFO], time_limit=50.0, clock=lambda: next(ticks), reps=3, periods=200)
    assert res["seconds"] == 202.0 and res["skipped"] == [C.M_SMOOTH, C.M_INFO]
    exactly = iter([0.0, 50.0, 50.0])                                        # genau am Limit: noch kalibrieren (nur später als das Limit zählt)
    res = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, 7, [C.M_BASIS], time_limit=50.0, clock=lambda: next(exactly), reps=3, periods=200)
    assert list(res["z"]) == [C.M_BASIS] and res["skipped"] == []


def test_zero_orders_are_counted_over_a_noise_threshold():
    assert C.ZERO_TOL == 1e-12
    D = np.full((1, 400), 100.0)
    D[0, 150] = 400.0                                                     # Spitze: danach bindet die Abschneidung
    r = LV.run_measure(D, C.M_BASIS, 0.5, 2, 10, [1.0] * 4)
    exact = ((r["q"][:, 100:]) == 0).mean(axis=1)
    assert r["zero_share"] == pytest.approx(exact.tolist()) and max(r["zero_share"]) > 0
    noisy = np.array([[0.0, 5e-13, 2e-12, 1.0]])
    assert float((noisy <= C.ZERO_TOL).mean()) == 0.5                     # Rauschen unter 1e-12 zählt als 0, darüber nicht
