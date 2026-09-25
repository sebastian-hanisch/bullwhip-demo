"""Einzeltests des Kerns (bw_model, bw_theory): Nachfragegenerator gegen Handrechnung mit einem Zufallsstrom-Ersatz, gleitendes Fenster, Bestellregel (Hand-Instanzen für Order-up-to, Glättung,
Info teilen, Fenster je Stufe), Lagerführung (Hand-Instanz mit Rückstand), Verstärkung, Kalibrierung, exakte Theorie. Statistische Tests laufen mit Standardfehler-Toleranz und ohne feste
Zahlen des Zufallsstroms (die CI installiert immer das neueste NumPy); die Referenzwerte stehen in tests/test_frozen_reference.py."""
import math

import numpy as np
import pytest

import bw_frozen as FZ
import bw_model as M
import bw_theory as T


class StubRng:
    """Ersatz für np.random.Generator: liefert vorgegebene Felder der Reihe nach (die Reihenfolge der Aufrufe ist Teil des Vertrags)."""

    def __init__(self, normals=(), uniforms=()):
        self.normals, self.uniforms = list(normals), list(uniforms)

    def standard_normal(self, shape):
        return self.normals.pop(0)

    def random(self, shape):
        return self.uniforms.pop(0)


# ---------------------------------------------------------------------------------------------------
# Nachfrage
# ---------------------------------------------------------------------------------------------------
def test_gen_ar1_follows_the_recursion_by_hand():
    rng = StubRng([np.ones((1, 3)), np.array([2.0])])
    D = M.gen_ar1(rng, 1, 3, 100.0, 4.0, 0.5)
    eps = 4.0 * math.sqrt(0.75)
    d0 = 108.0
    d1 = 100.0 + 0.5 * (d0 - 100.0) + eps
    d2 = 100.0 + 0.5 * (d1 - 100.0) + eps
    assert D.shape == (1, 3) and D[0].tolist() == pytest.approx([d0, d1, d2], abs=1e-12)


def test_gen_ar1_with_rho_zero_is_white_noise_around_the_mean_and_uses_the_two_draws_in_order():
    rng = StubRng([np.array([[1.0, -1.0, 0.5]]), np.array([0.0])])
    D = M.gen_ar1(rng, 1, 3, 50.0, 10.0, 0.0)
    assert D[0].tolist() == pytest.approx([50.0, 40.0, 55.0])


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.8])
def test_gen_ar1_has_the_requested_mean_spread_and_autocorrelation(rho):
    D = M.gen_ar1(np.random.default_rng(3), 200, 1500, 100.0, 20.0, rho)
    assert D.shape == (200, 1500)
    assert abs(D.mean() - 100.0) < 4 * 20.0 / math.sqrt(200 * 1500 / (1 + 2 * rho / (1 - rho)) ) + 0.3
    assert abs(D.std() - 20.0) < 0.6
    x = D - D.mean(axis=1, keepdims=True)
    lag1 = float((x[:, 1:] * x[:, :-1]).sum() / (x ** 2).sum())
    assert abs(lag1 - rho) < 0.02


def test_gen_ar1_starts_stationary_with_the_marginal_spread():
    D = M.gen_ar1(np.random.default_rng(5), 4000, 3, 100.0, 20.0, 0.8)
    assert abs(D[:, 0].std() - 20.0) < 1.0 and abs(D[:, 0].mean() - 100.0) < 1.0 and abs(D[:, 2].std() - 20.0) < 1.0


def test_gen_shift_clips_the_level_to_half_and_one_and_a_half_times_the_mean():
    up = M.gen_shift(StubRng([np.zeros((1, 5))], [np.zeros((1, 5)), np.full((1, 5), 0.9)]), 1, 5, 100.0, 5.0, rate=1.0)
    assert up[0].tolist() == pytest.approx([120.0, 140.0, 150.0, 150.0, 150.0])
    down = M.gen_shift(StubRng([np.zeros((1, 5))], [np.zeros((1, 5)), np.full((1, 5), 0.1)]), 1, 5, 100.0, 5.0, rate=1.0)
    assert down[0].tolist() == pytest.approx([80.0, 60.0, 50.0, 50.0, 50.0])


def test_gen_shift_without_jumps_is_the_mean_plus_noise():
    D = M.gen_shift(StubRng([np.array([[1.0, -2.0]])], [np.ones((1, 2)), np.zeros((1, 2))]), 1, 2, 100.0, 3.0, rate=0.0)
    assert D[0].tolist() == pytest.approx([103.0, 94.0])


def test_gen_shift_jumps_about_every_fifty_periods_on_average():
    D = M.gen_shift(np.random.default_rng(7), 200, 1000, 100.0, 0.001)
    lvl = np.rint(D)
    jumps = (np.abs(np.diff(D, axis=1)) > 5).sum() / D.shape[0] / (D.shape[1] - 1)
    assert 0.010 < jumps < 0.019 and lvl.min() >= 50 and lvl.max() <= 150                  # Rate 1/50 (Sprünge am Rand werden abgeschnitten: etwas darunter)


# ---------------------------------------------------------------------------------------------------
# Gleitendes Fenster
# ---------------------------------------------------------------------------------------------------
def test_window_starts_filled_with_the_initial_value_and_has_zero_spread():
    w = M.Window(3, 4, 100.0)
    assert w.mean().tolist() == [100.0] * 3 and w.sd().tolist() == [0.0] * 3


def test_window_matches_numpy_mean_and_sample_sd_of_the_last_p_values():
    rng = np.random.default_rng(1)
    xs = rng.normal(100, 15, size=(30, 3))
    w = M.Window(3, 5, 100.0)
    hist = [np.full(3, 100.0)] * 5
    for x in xs:
        w.push(x)
        hist = hist[1:] + [x]
        arr = np.array(hist)
        np.testing.assert_allclose(w.mean(), arr.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(w.sd(), arr.std(axis=0, ddof=1), rtol=1e-8, atol=1e-8)


def test_window_of_two_values_has_the_hand_computed_sd():
    w = M.Window(1, 2, 100.0)
    w.push(np.array([100.0]))
    w.push(np.array([110.0]))
    assert w.mean()[0] == 105.0 and w.sd()[0] == pytest.approx(math.sqrt(50.0))


def test_window_sd_never_goes_negative_from_rounding():
    w = M.Window(1, 3, 1e6)
    for _ in range(9):
        w.push(np.array([1e6]))
    assert w.sd()[0] >= 0.0


# ---------------------------------------------------------------------------------------------------
# Bestellregel
# ---------------------------------------------------------------------------------------------------
def test_order_up_to_first_period_by_hand_constant_demand():
    D = np.full((1, 4), 50.0)
    q, S, IPb = M.order_streams(D, [2], 5, z=1.0)
    assert q[0, 0, 0] == 50.0 and S[0, 0, 0] == 100.0 and IPb[0, 0, 0] == 50.0                # sd = 0: S = m * mu; IP0 = L * mu0
    assert np.allclose(q, 50.0)


def test_order_up_to_with_safety_stock_by_hand():
    D = np.array([[100.0, 110.0, 100.0]])
    q, S, IPb = M.order_streams(D, [1], 2, z=1.0, mu0=100.0)
    assert q[0, 0, 0] == 100.0                                                                  # Fenster [100, 100]: mu 100, sd 0
    sd = math.sqrt(50.0)
    assert S[0, 0, 1] == pytest.approx(105.0 + sd) and IPb[0, 0, 1] == pytest.approx(-10.0)      # IP nach t=0 ist 100, Bedarf 110 -> -10
    assert q[0, 0, 1] == pytest.approx(105.0 + sd + 10.0)


def test_smoothing_corrects_only_the_share_gamma_of_the_gap():
    D = np.array([[100.0, 110.0]])
    full, _, _ = M.order_streams(D, [1], 2, z=0.0, mu0=100.0)
    half, _, _ = M.order_streams(D, [1], 2, z=0.0, gamma=0.5, mu0=100.0)
    mu = 105.0                                                                                   # Fenster [100, 110]
    gap = 1 * mu + 10.0 - mu * 0 - 0.0                                                            # S - IPb = 105 + 10
    assert full[0, 0, 1] == pytest.approx(gap) and half[0, 0, 1] == pytest.approx(mu + 0.5 * (gap - mu))


def test_gamma_one_equals_the_plain_gap():
    D = M.gen_ar1(np.random.default_rng(2), 5, 60, 100.0, 20.0, 0.0)
    a, _, _ = M.order_streams(D, [2, 2], 6, z=1.0, gamma=1.0)
    b, _, _ = M.order_streams(D, [2, 2], 6, z=1.0, gammas=[1.0, 1.0])
    assert np.array_equal(a, b)


def test_orders_are_never_negative_unless_truncation_is_switched_off():
    D = np.full((1, 30), 3.0)
    D[0, 15] += 30.0
    q, _, _ = M.order_streams(D, [1], 2, z=0.0, mu0=3.0)
    ql, _, _ = M.order_streams(D, [1], 2, z=0.0, nonneg=False, mu0=3.0)
    assert q.min() == 0.0 and ql.min() == pytest.approx(-12.0)


def test_share_makes_every_stage_forecast_from_the_end_customer_demand():
    D = M.gen_ar1(np.random.default_rng(3), 4, 80, 100.0, 20.0, 0.0)
    q, S, _ = M.order_streams(D, [2, 2, 2, 2], 6, z=1.0, share=True)
    for k in range(4):                                     # mu_hat und sigma_hat aller Stufen aus D: S je Periode gleich, weil m gleich ist
        assert np.allclose(S[k], S[0])
    qn, Sn, _ = M.order_streams(D, [2, 2, 2, 2], 6, z=1.0)
    assert not np.allclose(Sn[3], Sn[0])


def test_stage_specific_window_and_gamma_apply_only_to_their_stage():
    D = M.gen_ar1(np.random.default_rng(4), 3, 120, 100.0, 20.0, 0.0)
    base, _, _ = M.order_streams(D, [2, 2, 2], 8, z=1.0)
    other, _, _ = M.order_streams(D, [2, 2, 2], 8, z=1.0, ps=[8, 8, 4])
    assert np.array_equal(base[:2], other[:2]) and not np.array_equal(base[2], other[2])
    g, _, _ = M.order_streams(D, [2, 2, 2], 8, z=1.0, gammas=[1, 0.5, 1])
    assert np.array_equal(base[0], g[0]) and not np.array_equal(base[1], g[1])


def test_z_can_be_a_list_per_stage_and_a_scalar_is_the_same_for_all():
    D = M.gen_ar1(np.random.default_rng(5), 3, 100, 100.0, 20.0, 0.0)
    a, _, _ = M.order_streams(D, [2, 2, 2], 8, z=1.2)
    b, _, _ = M.order_streams(D, [2, 2, 2], 8, z=[1.2, 1.2, 1.2])
    c, _, _ = M.order_streams(D, [2, 2, 2], 8, z=[1.2, 1.2, 1.5])
    assert np.array_equal(a, b) and np.array_equal(a[:2], c[:2]) and not np.array_equal(a[2], c[2])


def test_default_start_value_is_the_mean_of_the_first_period_over_all_replications():
    D = np.array([[80.0, 90.0, 110.0, 100.0], [120.0, 95.0, 105.0, 100.0]])
    a, _, _ = M.order_streams(D, [1], 3, z=0.0)
    b, _, _ = M.order_streams(D, [1], 3, z=0.0, mu0=100.0)                                       # (80 + 120) / 2
    c, _, _ = M.order_streams(D, [1], 3, z=0.0, mu0=90.0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_ip_after_order_is_the_running_position():
    D = M.gen_ar1(np.random.default_rng(6), 2, 50, 100.0, 20.0, 0.0)
    q, _, IPb = M.order_streams(D, [2, 2], 5, z=1.0)
    ip = M.ip_after_order(D, q, [2, 2])
    for k in range(2):
        x = D if k == 0 else q[k - 1]
        np.testing.assert_allclose(ip[k], IPb[k] + q[k], atol=1e-9)
        assert np.allclose(ip[k][:, 0], 2 * D[:, 0].mean() - x[:, 0] + q[k][:, 0])


# ---------------------------------------------------------------------------------------------------
# Lagerführung
# ---------------------------------------------------------------------------------------------------
def test_physical_hand_instance_without_orders_builds_up_backlog():
    D = np.full((1, 5), 10.0)
    q = np.zeros((1, 1, 5))
    ph = M.physical(D, q, [1], h=1.0, pc=9.0, burn=0)
    assert ph["onhand"][0, 0] == 0.0 and ph["backlog"][0, 0] == pytest.approx(20.0)             # 0, 10, 20, 30, 40 Rückstand
    assert ph["beta"][0, 0] == pytest.approx(0.2) and ph["alpha"][0, 0] == pytest.approx(0.2)
    assert ph["cost"][0, 0] == pytest.approx(9.0 * 20.0)


def test_physical_delivers_after_the_lead_time_and_pays_the_backlog_off():
    D = np.zeros((1, 6))
    D[0, 1] = 10.0
    q = np.zeros((1, 1, 6))
    q[0, 0, 0] = 10.0                                                                            # Bestellung in t=0 kommt in t=1 an (L = 1)
    ph = M.physical(D, q, [1], burn=0)
    assert ph["beta"][0, 0] == 1.0 or ph["beta"][0, 0] == pytest.approx(1.0)                      # Bedarf 10 in t=1 wird aus der Lieferung gedeckt


def test_physical_burn_excludes_the_first_periods():
    D = np.full((1, 6), 10.0)
    q = np.zeros((1, 1, 6))
    ph0 = M.physical(D, q, [1], burn=0)
    ph3 = M.physical(D, q, [1], burn=3)
    assert ph3["beta"][0, 0] == 0.0 and ph0["beta"][0, 0] > 0.0                                  # ab t=3 wird nichts mehr sofort geliefert
    assert ph3["backlog"][0, 0] == pytest.approx((30 + 40 + 50) / 3)


def test_physical_service_is_one_and_the_stock_is_exactly_used_up_with_constant_demand():
    D = np.full((2, 40), 50.0)
    q, _, _ = M.order_streams(D, [2, 2], 5, z=2.0)
    ph = M.physical(D, q, [2, 2], burn=10)
    assert np.allclose(ph["beta"], 1.0) and np.allclose(ph["backlog"], 0.0) and np.allclose(ph["onhand"], 0.0)


# ---------------------------------------------------------------------------------------------------
# Verstärkung
# ---------------------------------------------------------------------------------------------------
def test_var_ratio_of_a_scaled_copy_is_the_square_of_the_factor():
    D = M.gen_ar1(np.random.default_rng(8), 30, 200, 100.0, 20.0, 0.0)
    q = np.stack([D * 2.0, D * 3.0])
    vr = M.var_ratio(q, D, 10)
    assert vr[0][0] == pytest.approx(4.0) and vr[1][0] == pytest.approx(9.0) and vr[0][1] == pytest.approx(0.0, abs=1e-12)


def test_var_ratio_uses_only_periods_after_the_burn_in_and_returns_the_standard_error():
    rng = np.random.default_rng(9)
    D = rng.normal(100, 20, size=(50, 300))
    q = D[None].copy()
    q[0, :, :50] += 1000.0                                                                      # nur im Einschwingen verstellt
    (a, s), = M.var_ratio(q, D, 60)
    assert a == pytest.approx(1.0) and s == pytest.approx(0.0, abs=1e-12)
    q2 = np.stack([D * (1 + 0.1 * np.arange(50))[:, None]])
    (a2, s2), = M.var_ratio(q2, D, 0)
    factors = (1 + 0.1 * np.arange(50)) ** 2
    assert a2 == pytest.approx(factors.mean()) and s2 == pytest.approx(factors.std(ddof=1) / math.sqrt(50))


# ---------------------------------------------------------------------------------------------------
# Kalibrierung
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cal_demand():
    return FZ.demand("cal")


def test_calibration_hits_the_target_service_on_the_calibration_sample(cal_demand):
    zs, q, ph = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, burn=50, passes=2, iters=8)
    assert np.all(np.abs(ph["beta"].mean(axis=1) - 0.95) < 0.01) and len(zs) == 2 and all(0.0 < z < 8.0 for z in zs)


def test_a_higher_service_target_needs_a_larger_z(cal_demand):
    low, _, _ = M.calibrate_z(cal_demand, [2, 2], 10, target=0.90, burn=50, passes=1, iters=7)
    high, _, _ = M.calibrate_z(cal_demand, [2, 2], 10, target=0.99, burn=50, passes=1, iters=7)
    assert all(h > l for h, l in zip(high, low))


def test_calibration_returns_grid_points_and_the_orders_belonging_to_them(cal_demand):
    zs, q, ph = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, burn=50, passes=1, iters=6, zmax=8.0)
    for z in zs:
        assert (z * 64) == pytest.approx(round(z * 64))                                          # zmax / 2**iters Gitter: 8 / 64 = 0,125
    q2, _, _ = M.order_streams(cal_demand, [2, 2], 10, z=zs)
    assert np.array_equal(q, q2)


def test_calibration_passes_smoothing_and_sharing_through(cal_demand):
    a, _, _ = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, burn=50, passes=1, iters=6)
    b, _, _ = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, burn=50, passes=1, iters=6, gamma=0.5)
    c, _, _ = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, burn=50, passes=1, iters=6, share=True)
    assert a != b and a != c


# ---------------------------------------------------------------------------------------------------
# Exakte lineare Theorie
# ---------------------------------------------------------------------------------------------------
def test_stage_filter_has_the_two_taps_and_unit_gain():
    h = T.stage_filter(2, 5)
    assert len(h) == 6 and h[0] == pytest.approx(1.4) and h[5] == pytest.approx(-0.4) and h[1:5].tolist() == [0.0] * 4 and h.sum() == pytest.approx(1.0)


def test_chain_filter_lengths_and_unit_gain():
    for k, h in enumerate(T.chain_filters([2, 2, 2, 2], [5] * 4)):
        assert len(h) == 1 + (k + 1) * 5 and h.sum() == pytest.approx(1.0)
    hs = T.chain_filters([1, 3, 2], [4, 4, 4], share=True)
    assert [len(h) for h in hs] == [5, 5, 5] and hs[2][0] == pytest.approx(1.0 + 6 / 4) and hs[2][4] == pytest.approx(-6 / 4)


def test_shared_information_requires_the_same_window_everywhere():
    with pytest.raises(AssertionError):
        T.chain_filters([2, 2], [4, 5], share=True)


def test_ratio_for_iid_demand_is_the_sum_of_squares():
    h = T.stage_filter(2, 5)
    assert T.ratio_from_filter(h, 0.0) == pytest.approx(1.4 ** 2 + 0.4 ** 2) == pytest.approx(T.chen_single_bound(2, 5))


def test_ratio_with_autocorrelation_matches_the_explicit_double_sum():
    h = np.array([1.4, 0.0, -0.4])
    rho = 0.6
    want = sum(h[i] * h[j] * rho ** abs(i - j) for i in range(3) for j in range(3))
    assert T.ratio_from_filter(h, rho) == pytest.approx(want)
    assert T.ratio_from_filter(h, 1.0) == pytest.approx(h.sum() ** 2)                            # vollständig korreliert: (Summe)^2


def test_chen_single_stage_bound_hand_values():
    assert T.chen_single_bound(2, 5) == pytest.approx(2.12) and T.chen_single_bound(1, 1) == 5.0 and T.chen_single_bound(0, 7) == 1.0


def test_the_amplification_depends_only_on_m_over_p():
    a = T.ratio_from_filter(T.chain_filters([2] * 4, [10] * 4)[3], 0.0)
    b = T.ratio_from_filter(T.chain_filters([4] * 4, [20] * 4)[3], 0.0)
    c = T.ratio_from_filter(T.chain_filters([1] * 4, [5] * 4)[3], 0.0)
    assert a == pytest.approx(b) and a == pytest.approx(c)


def test_stage_two_hand_formula():
    m, p = 2, 5
    a, b = 1 + m / p, m / p
    assert T.ratio_from_filter(T.chain_filters([m] * 2, [p] * 2)[1], 0.0) == pytest.approx(a ** 4 + 4 * a ** 2 * b ** 2 + b ** 4)


# ---------------------------------------------------------------------------------------------------
# Lücken aus dem Fehler-Einbau-Test: Standardwerte, Grenzfälle, Kontrollgrößen
# ---------------------------------------------------------------------------------------------------
def test_defaults_of_the_public_functions_are_the_documented_ones():
    D = np.array([[100.0, 110.0, 100.0, 90.0, 100.0]])
    q_default, _, _ = M.order_streams(D, [1], 2)
    q_explicit, _, _ = M.order_streams(D, [1], 2, z=0.0, gamma=1.0, share=False, nonneg=True)
    assert np.array_equal(q_default, q_explicit)                                                     # z = 0, gamma = 1, geteilt aus, mit Abschneidung
    hand = np.full((1, 5), 10.0)
    ph = M.physical(hand, np.zeros((1, 1, 5)), [1])
    assert ph["cost"][0, 0] == pytest.approx(9.0 * 20.0) and ph["beta"][0, 0] == pytest.approx(0.2)   # h = 1, pc = 9, burn = 0


def test_calibration_defaults_are_target_95_two_passes_nine_bisections_from_zero_to_eight(cal_demand):
    a = M.calibrate_z(cal_demand, [2, 2], 10)
    b = M.calibrate_z(cal_demand, [2, 2], 10, target=0.95, passes=2, burn=100, iters=9, zmax=8.0)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_the_position_check_reports_a_deviation_in_any_replication():
    D = M.gen_ar1(np.random.default_rng(12), 3, 60, 100.0, 20.0, 0.0)
    q, _, _ = M.order_streams(D, [2, 2], 5, z=1.0)
    ip = M.ip_after_order(D, q, [2, 2])
    ok = M.physical(D, q, [2, 2], burn=10, check_ip=ip)["ip_max_dev"]
    assert ok < 1e-9
    bent = ip.copy()
    bent[1, 2, 30] += 1.0                                                                            # nur in einer Replikation, einer Stufe und einer Periode falsch
    assert M.physical(D, q, [2, 2], burn=10, check_ip=bent)["ip_max_dev"] == pytest.approx(1.0)
    bent[0, 0, 5] -= 3.0
    assert M.physical(D, q, [2, 2], burn=10, check_ip=bent)["ip_max_dev"] == pytest.approx(3.0)      # das Maximum über alles, nicht das kleinste


def test_alpha_counts_periods_with_a_backlog_up_to_one_billionth():
    q = np.zeros((1, 1, 4))
    for backlog, alpha in ((1e-12, 1.0), (1e-9, 1.0), (1e-6, 0.25)):
        D = np.array([[10.0, backlog, 0.0, 0.0]])
        assert M.physical(D, q, [1], burn=0)["alpha"][0, 0] == pytest.approx(alpha), backlog


def test_shift_jumps_need_a_uniform_draw_strictly_below_the_rate_and_the_sign_needs_half():
    up = M.gen_shift(StubRng([np.zeros((1, 3))], [np.full((1, 3), 0.02), np.full((1, 3), 0.52)]), 1, 3, 100.0, 5.0, rate=0.02)
    assert up[0].tolist() == [100.0, 100.0, 100.0]                                                    # Draw == Rate: kein Sprung
    jump = M.gen_shift(StubRng([np.zeros((1, 3))], [np.full((1, 3), 0.019), np.full((1, 3), 0.52)]), 1, 3, 100.0, 5.0, rate=0.02)
    assert jump[0].tolist() == pytest.approx([120.0, 140.0, 150.0])                                   # 0,52 liegt über 0,5: Sprung nach oben
    down = M.gen_shift(StubRng([np.zeros((1, 3))], [np.full((1, 3), 0.019), np.full((1, 3), 0.49)]), 1, 3, 100.0, 5.0, rate=0.02)
    assert down[0].tolist() == pytest.approx([80.0, 60.0, 50.0])
    default_rate = M.gen_shift(StubRng([np.zeros((1, 2))], [np.full((1, 2), 0.019), np.full((1, 2), 0.9)]), 1, 2, 100.0, 5.0)
    assert default_rate[0].tolist() == pytest.approx([120.0, 140.0])                                  # Standard: Rate 1/50 und Größe 0,2


def test_calibration_evaluates_with_the_default_costs_and_the_default_burn_in(cal_demand):
    zs, q, ph = M.calibrate_z(cal_demand, [2, 2], 10, iters=5, passes=1)
    ref = M.physical(cal_demand, q, [2, 2], 1.0, 9.0, burn=100)
    for key in ref:
        np.testing.assert_allclose(ph[key], ref[key], rtol=0, atol=0, err_msg=key)
    assert not np.allclose(ph["beta"], M.physical(cal_demand, q, [2, 2], 1.0, 9.0, burn=101)["beta"], rtol=0, atol=0)
    assert not np.allclose(ph["cost"], M.physical(cal_demand, q, [2, 2], 1.1, 9.0, burn=100)["cost"], rtol=0, atol=0)


def test_a_uniform_draw_of_exactly_one_half_gives_a_jump_upwards():
    up = M.gen_shift(StubRng([np.zeros((1, 2))], [np.full((1, 2), 0.0), np.full((1, 2), 0.5)]), 1, 2, 100.0, 5.0, rate=1.0)
    assert up[0].tolist() == pytest.approx([120.0, 140.0])
