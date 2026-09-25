"""Die vorgerechnete Messreihe (data/bw_results.json) und ihre Auswertung (bw_results): Struktur, Klassifikation der Jobs, Zell-Zuordnung, Urteil in drei Zuständen mit Schwellen,
z-Zuordnung für die Live-Kette, Tabellen der Kernabschnitte, Abdeckung; dazu innere Konsistenz der Datei (Summen, Servicegrade) und Nachrechnung der Theorie-Blöcke E1/E2/E3 mit bw_theory."""
import copy
import itertools
import math

import pytest

import bw_constants as C
import bw_results as R
import bw_theory as T

DATA = R.load_results()
MAIN_KEYS = [k for k, _ in R.MAIN13]


# ---------------------------------------------------------------------------------------------------
# Datei
# ---------------------------------------------------------------------------------------------------
def test_the_results_file_has_the_expected_blocks_and_sizes():
    assert set(DATA) == {"_meta", "E1", "E2", "E3", "jobs"}
    assert len(DATA["jobs"]) == 63 and DATA["_meta"]["n_jobs"] == 63 and len(DATA["E1"]) == 12 and len(DATA["E2"]) == 27 and len(DATA["E3"]) == 80
    assert DATA["_meta"]["config"]["R_EVAL"] == 500 and DATA["_meta"]["config"]["T_EVAL"] == 1500 and DATA["_meta"]["config"]["TARGET"] == 0.95
    assert (R.DATA_PATH.stat().st_size) < 120 * 1024                                              # klein gehalten


def test_load_results_is_cached_and_accepts_a_path(tmp_path):
    assert R.load_results() is R.load_results()
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert R.load_results(p) == {"a": 1}


def test_every_job_has_consistent_numbers():
    for name, j in DATA["jobs"].items():
        assert len(j["z"]) == 4 and all(0.0 < z <= 10.0 for z in j["z"]) and not any(j["z_at_cap"]), name
        e = j["equal"]
        assert abs(sum(e["onhand"]) - e["total_onhand"]) < 1e-3 * max(1.0, e["total_onhand"]) * 1e-2 + 1e-2, name
        assert abs(sum(e["cost"]) - e["total_cost"]) < 1e-2 * max(1.0, e["total_cost"]) * 1e-2 + 1e-2, name
        assert all(0.947 < b < 0.955 for b in e["beta"]), name                                    # Ziel 95 %, auf der unabhängigen Stichprobe bewertet
        assert all(a[0] > 0 and a[1] > 0 for a in e["amp"]) and e["total_onhand_se"] > 0, name
        assert len(e["amp"]) == len(e["onhand"]) == len(e["backlog"]) == len(e["cost"]) == len(e["zero_share"]) == 4
        if j["fixed"] is not None:
            assert len(j["fixed"]["beta"]) == 4 and abs(sum(j["fixed"]["onhand"]) - j["fixed"]["total_onhand"]) < 1e-2 + 1e-4 * j["fixed"]["total_onhand"], name


def test_the_fixed_z_block_exists_for_all_jobs_except_the_seed_repetition():
    without = [n for n, j in DATA["jobs"].items() if j["fixed"] is None]
    assert without == [n for n in DATA["jobs"] if n.startswith("main2/")] and len(without) == 13


def test_service_is_within_the_reported_range():
    lo, hi = R.service_range(DATA)
    assert 0.9484 <= lo < 0.95 < hi <= 0.9533


def test_the_theory_blocks_agree_with_an_independent_recomputation():
    for r in DATA["E1"]:
        Ls, ps = [r["m"]] * 4, [r["p"]] * 4
        for got, want in zip(r["exact"], [T.ratio_from_filter(h, 0.0) for h in T.chain_filters(Ls, ps)]):
            assert got == pytest.approx(want, rel=1e-5)
        for got, want in zip(r["share_exact"], [T.ratio_from_filter(h, 0.0) for h in T.chain_filters(Ls, ps, share=True)]):
            assert got == pytest.approx(want, rel=1e-5)
        for k, got in enumerate(r["chen_product"]):
            assert got == pytest.approx(T.chen_single_bound(r["m"], r["p"]) ** (k + 1), rel=1e-5)
        assert all(abs(s - e) < 4 * se + 0.01 * e for s, e, se in zip(r["sim"], r["exact"], r["sim_se"]))     # Simulation == Theorie auf 4 Standardfehler + 1 %
    for r in DATA["E3"][::7]:
        Ls, ps = [r["m"]] * 4, [r["p"]] * 4
        for got, want in zip(r["local"], [T.ratio_from_filter(h, r["rho"]) for h in T.chain_filters(Ls, ps)]):
            assert got == pytest.approx(want, rel=1e-5)
        for got, want in zip(r["share"], [T.ratio_from_filter(h, r["rho"]) for h in T.chain_filters(Ls, ps, share=True)]):
            assert got == pytest.approx(want, rel=1e-5)
    for r in DATA["E2"][::4]:
        for got, want in zip(r["linear_exact"], [T.ratio_from_filter(h, 0.0) for h in T.chain_filters([r["m"]] * 4, [r["p"]] * 4)]):
            assert got == pytest.approx(want, rel=1e-5)


# ---------------------------------------------------------------------------------------------------
# Klassifikation und Zellen
# ---------------------------------------------------------------------------------------------------
def test_entries_cover_the_main_scenario_and_nine_cells_but_not_the_repetition_or_jump_series():
    es = R.entries(DATA)
    assert len(es) == 13 + 9 * 3 and {e["group"] for e in es} == set(R.CELLS)
    assert not any(e["group"] in ("main2", "ar1", "shift") for e in es)


def test_classification_of_the_main_jobs():
    got = {e["name"].split("/")[1]: (e["measure"], e["kind"], e["gamma"], e["L"], e["p"]) for e in R.entries(DATA) if e["group"] == "main"}
    assert got["basis"] == ("basis", "basis", None, 2, 10)
    assert got["glaettung_0.7"] == ("glaettung", "glaettung", 0.7, 2, 10) and got["glaettung_0.3"][2] == 0.3
    assert got["nur_einzelhandel_glaettet_0.5"] == ("einzelhandel", "einzelhandel", 0.5, 2, 10)
    assert got["nur_stufe2_glaettet_0.3"][:2] == ("stufe2", "stufe2") and got["nur_fabrik_glaettet_0.3"][:2] == ("fabrik", "fabrik")
    assert got["info_teilen"] == ("info", "info", None, 2, 10) and got["info_teilen_glaettung_0.5"] == ("info_glaettung", "info_glaettung", 0.5, 2, 10)
    assert got["L_halbiert"] == ("lead_halved", "basis", None, 1, 10) and got["Fenster_doppelt"] == ("window_doubled", "basis", None, 2, 20)
    assert got["L_halbiert_info_teilen"] == ("lead_info", "info", None, 1, 10)


def test_cell_settings_and_labels():
    assert [R.cell_setting(DATA, c) for c in R.CELLS] == [(0.2, 0.0, 2, 10), (0.2, 0.5, 2, 10), (0.2, 0.8, 2, 10), (0.2, 0.0, 1, 10), (0.2, 0.0, 4, 10), (0.2, 0.0, 2, 6), (0.2, 0.0, 2, 20),
                                                          (0.2, 0.0, 2, 40), (0.1, 0.0, 2, 10), (0.3, 0.0, 2, 10)]
    assert [R.cell_label(DATA, c) for c in R.CELLS] == ["Hauptszenario", "ρ 0,5", "ρ 0,8", "L 1", "L 4", "p 6", "p 20", "p 40", "CV 0,1", "CV 0,3"]
    assert [R.cell_group(c) for c in R.CELLS] == ["Hauptszenario", "Autokorrelation", "Autokorrelation", "Wiederbeschaffungszeit", "Wiederbeschaffungszeit", "Prognosefenster", "Prognosefenster",
                                                  "Prognosefenster", "Nachfrageschwankung", "Nachfrageschwankung"]
    assert R.cell_setting_text(DATA, "main") == "Hauptszenario (CV 0,2, ρ 0,0, L 2, p 10)" and R.cell_setting_text(DATA, "L4") == "CV 0,2, ρ 0,0, L 4, p 10"


def test_measured_cells_per_measure():
    assert R.measured_cells(DATA) == list(R.CELLS) and R.measured_cells(DATA, C.M_BASIS) == list(R.CELLS)
    assert R.measured_cells(DATA, C.M_SMOOTH) == list(R.CELLS) and R.measured_cells(DATA, C.M_INFO) == list(R.CELLS)
    for m in (C.M_RETAIL, C.M_INFO_SMOOTH, C.M_LEAD, C.M_WINDOW):
        assert R.measured_cells(DATA, m) == ["main"], m


@pytest.mark.parametrize("cell", R.CELLS)
def test_every_measured_cell_maps_to_itself(cell):
    cv, rho, lead, window = R.cell_setting(DATA, cell)
    got, exact, diff = R.nearest_cell(DATA, cv, rho, lead, window)
    assert (got, exact, diff) == (cell, True, [])


def test_all_108_combinations_have_a_nearest_cell_with_at_most_three_differing_parameters():
    n = 0
    for combo in itertools.product(C.CV_OPTIONS, C.RHO_OPTIONS, C.LEAD_OPTIONS, C.WINDOW_OPTIONS):
        cell, exact, diff = R.nearest_cell(DATA, *combo)
        assert cell in R.CELLS and exact == (not diff) and len(diff) <= 3
        n += 1
    assert n == 108


def test_ties_go_to_the_cell_that_matches_the_more_influential_parameter():
    assert R.nearest_cell(DATA, 0.3, 0.0, 4, 10)[:2] == ("L4", False)                       # L 4 passt, nur CV weicht ab
    assert R.nearest_cell(DATA, 0.2, 0.5, 2, 6)[0] == "p6"                                  # p vor ρ
    assert R.nearest_cell(DATA, 0.1, 0.8, 2, 10)[0] == "rho0.8"                             # ρ vor CV
    assert R.nearest_cell(DATA, 0.3, 0.5, 2, 10)[0] == "rho0.5" and R.nearest_cell(DATA, 0.2, 0.5, 4, 10)[0] == "L4"
    assert R.nearest_cell(DATA, 0.2, 0.0, 2, 10, C.M_RETAIL)[0] == "main" and R.nearest_cell(DATA, 0.1, 0.8, 4, 40, C.M_LEAD)[:2] == ("main", False)


def test_nearest_cell_reports_the_parameters_that_still_differ():
    cell, exact, diff = R.nearest_cell(DATA, 0.3, 0.8, 4, 40)
    assert not exact and len(diff) == 3 and set(diff) <= {"cv", "rho", "lead", "window"}
    assert R.differing_params((0.2, 0.0, 2, 10), (0.2, 0.0, 2, 10)) == [] and R.differing_params((0.3, 0.8, 4, 40), (0.2, 0.0, 2, 10)) == ["lead", "window", "rho", "cv"]
    assert R.differing_params((0.2, 0.0, 2, 12), (0.2, 0.0, 2, 10)) == ["window"]


def test_assignment_note_is_empty_when_exact_and_names_the_differing_parameters_otherwise():
    assert R.assignment_note(DATA, 0.2, 0.0, 2, 10, "main", []) == ""
    cell, exact, diff = R.nearest_cell(DATA, 0.3, 0.8, 4, 10)
    note = R.assignment_note(DATA, 0.3, 0.8, 4, 10, cell, diff)
    assert "CV 0,3, ρ 0,8, L 4, p 10" in note and "keine eigene Messreihe" in note and all(R.PARAM_LABELS[p] in note for p in diff) and R.cell_setting_text(DATA, cell) in note


def test_measure_activity():
    assert not R.measure_active(C.M_LEAD, 1, 10) and R.measure_active(C.M_LEAD, 2, 10) and R.measure_active(C.M_LEAD, 4, 10)
    assert not R.measure_active(C.M_WINDOW, 2, 40) and R.measure_active(C.M_WINDOW, 2, 20) and R.measure_active(C.M_WINDOW, 2, 6)
    assert all(R.measure_active(m, 1, 40) for m in (C.M_BASIS, C.M_SMOOTH, C.M_RETAIL, C.M_INFO, C.M_INFO_SMOOTH))
    assert R.active_measures(2, 10) == list(C.MEASURES) and R.active_measures(1, 40) == [C.M_BASIS, C.M_SMOOTH, C.M_RETAIL, C.M_INFO, C.M_INFO_SMOOTH]
    assert R.effective_config(C.M_LEAD, 4, 10) == (2, 10) and R.effective_config(C.M_WINDOW, 2, 6) == (2, 12) and R.effective_config(C.M_SMOOTH, 4, 20) == (4, 20)


# ---------------------------------------------------------------------------------------------------
# Urteil in drei Zuständen
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("delta,se,state", [
    (-5.0, 0.1, C.STATE_LOWER), (-4.99, 0.1, C.STATE_LITTLE), (2.0, 0.1, C.STATE_LITTLE), (2.01, 0.1, C.STATE_COSTS), (0.0, 0.1, C.STATE_LITTLE),
    (-50.0, 0.1, C.STATE_LOWER), (30.0, 0.1, C.STATE_COSTS),
    (-6.0, 3.1, C.STATE_LITTLE), (-6.0, 2.9, C.STATE_LOWER), (-6.0, 3.0, C.STATE_LITTLE),                     # genau 2 Standardfehler: nicht belastbar
    (3.0, 1.6, C.STATE_LITTLE), (3.0, 1.4, C.STATE_COSTS), (3.0, 1.5, C.STATE_LITTLE),
])
def test_state_thresholds_and_standard_error_condition(delta, se, state):
    assert R.state_of(delta, se) == state


def test_state_text_names_the_measure_and_the_verdict():
    assert R.state_text(C.M_SMOOTH, C.STATE_LOWER) == "Glättung senkt hier den Kettenbestand"
    assert R.state_text(C.M_SMOOTH, C.STATE_LITTLE) == "Glättung bringt hier wenig" and R.state_text(C.M_SMOOTH, C.STATE_COSTS) == "Glättung kostet hier"
    assert R.state_text(C.M_INFO, C.STATE_LOWER) == "Info teilen senkt hier den Kettenbestand" and R.state_text(C.M_LEAD, C.STATE_COSTS).startswith("Die halbierte Wiederbeschaffungszeit")


def test_change_pct_is_relative_to_the_basis_with_propagated_error():
    b, m = DATA["jobs"]["main/basis"], DATA["jobs"]["main/glaettung_0.5"]
    d, se = R.change_pct(m, b)
    tb, sb = R.total(b)
    tm, sm = R.total(m)
    assert d == pytest.approx(100 * (tm / tb - 1)) and se == pytest.approx(100 * tm / tb * math.hypot(sm / tm, sb / tb))
    assert R.change_pct(b, b)[0] == 0.0 and d < 0
    df, sf = R.change_pct(m, b, "fixed")
    assert df < d and sf > 0                                                                      # bei festem z sieht Glättung noch billiger aus


def test_amp_last_and_total_read_the_right_fields():
    j = DATA["jobs"]["main/basis"]
    assert R.amp_last(j) == (j["equal"]["amp"][3][0], j["equal"]["amp"][3][1]) and R.amp_last(j, "fixed")[0] == j["fixed"]["amp"][3][0]
    assert R.total(j) == (j["equal"]["total_onhand"], j["equal"]["total_onhand_se"]) and R.total(j, "fixed")[0] == j["fixed"]["total_onhand"]


# ---------------------------------------------------------------------------------------------------
# Meldung: Zelle, Glättung, Jobs
# ---------------------------------------------------------------------------------------------------
def test_pair_in_the_main_scenario_is_exact_and_reproduces_the_hook_numbers():
    r = R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)
    assert r["exact"] and r["cell"] == "main" and r["gamma"] == 0.5 and r["note"] == "" and r["state"] == C.STATE_LOWER
    assert r["delta"] == pytest.approx(-15.7, abs=0.05) and r["se"] < 0.5 and r["base_name"] == "main/basis" and r["name"] == "main/glaettung_0.5"
    assert r["total_base"] == pytest.approx(270.2, abs=0.1) and r["total"] == pytest.approx(227.7, abs=0.1) and "fixed_delta" in r and r["cell_text"].startswith("Hauptszenario")
    d = {m: R.resolve_pair(DATA, m, 0.5, 0.2, 0.0, 2, 10)["delta"] for m in C.MEASURES if m != C.M_BASIS}
    assert d[C.M_LEAD] == pytest.approx(-65.1, abs=0.1) and d[C.M_INFO] == pytest.approx(-26.9, abs=0.1) and d[C.M_INFO_SMOOTH] == pytest.approx(-42.9, abs=0.1)
    assert d[C.M_WINDOW] == pytest.approx(-45.4, abs=0.1) and d[C.M_RETAIL] == pytest.approx(-18.3, abs=0.1)


def test_pair_uses_the_nearest_measured_smoothing_level_and_says_so():
    r = R.resolve_pair(DATA, C.M_SMOOTH, 0.7, 0.2, 0.8, 2, 10)
    assert r["cell"] == "rho0.8" and r["cell_exact"] and r["gamma"] == 0.5 and not r["gamma_exact"] and not r["exact"] and "γ 0,5" in r["note"] and "γ 0,7" in r["note"] and r["state"] == C.STATE_COSTS
    r7 = R.resolve_pair(DATA, C.M_SMOOTH, 0.7, 0.2, 0.0, 2, 10)
    assert r7["gamma"] == 0.7 and r7["gamma_exact"] and r7["delta"] == pytest.approx(-14.7, abs=0.1)
    r3 = R.resolve_pair(DATA, C.M_INFO_SMOOTH, 0.3, 0.2, 0.0, 2, 10)                            # nur γ 0,5 gemessen
    assert r3["gamma"] == 0.5 and not r3["gamma_exact"]


def test_pair_for_measures_only_measured_in_the_main_scenario_moves_to_it_with_a_note():
    r = R.resolve_pair(DATA, C.M_LEAD, 0.5, 0.2, 0.0, 4, 10)
    assert r["cell"] == "main" and not r["cell_exact"] and "nur im Hauptszenario gemessen" in r["note"] and r["gamma"] is None and r["exact"] is False
    r2 = R.resolve_pair(DATA, C.M_INFO, 0.5, 0.3, 0.8, 4, 40)
    assert r2["cell"] in R.CELLS and not r2["cell_exact"] and "keine eigene Messreihe" in r2["note"]


def test_pair_compares_with_the_basis_of_the_same_cell():
    for cell in R.CELLS:
        cv, rho, lead, window = R.cell_setting(DATA, cell)
        r = R.resolve_pair(DATA, C.M_SMOOTH, 0.5, cv, rho, lead, window)
        assert r["base_name"] == f"{cell}/basis" and r["name"] == f"{cell}/glaettung_0.5" and r["exact"]


def test_gamma_choices_and_cell_entry():
    assert R.gamma_choices(DATA, "main", C.M_SMOOTH) == [0.7, 0.5, 0.3] and R.gamma_choices(DATA, "main", C.M_RETAIL) == [0.5, 0.3] and R.gamma_choices(DATA, "rho0.8", C.M_SMOOTH) == [0.5]
    assert R.cell_entry(DATA, "main", C.M_SMOOTH, 0.3)["name"] == "main/glaettung_0.3" and R.cell_entry(DATA, "p6", C.M_INFO)["name"] == "p6/info_teilen"
    with pytest.raises(KeyError):
        R.cell_entry(DATA, "L1", C.M_LEAD)


# ---------------------------------------------------------------------------------------------------
# z für die Live-Kette
# ---------------------------------------------------------------------------------------------------
def test_z_of_the_main_scenario_is_exact_and_equals_the_calibration():
    for m, name in ((C.M_BASIS, "basis"), (C.M_SMOOTH, "glaettung_0.5"), (C.M_RETAIL, "nur_einzelhandel_glaettet_0.5"), (C.M_INFO, "info_teilen"), (C.M_INFO_SMOOTH, "info_teilen_glaettung_0.5"),
                    (C.M_LEAD, "L_halbiert"), (C.M_WINDOW, "Fenster_doppelt")):
        r = R.resolve_z(DATA, m, 0.2, 0.0, 2, 10, 0.5)
        assert r["exact"] and r["z"] == DATA["jobs"][f"main/{name}"]["z"] and r["job"] == f"main/{name}" and r["note"] == "" and r["distance"] == 0, m


def test_z_belongs_to_the_configuration_not_to_the_measure():
    r = R.resolve_z(DATA, C.M_LEAD, 0.2, 0.0, 4, 10, 0.5)                                       # L halbiert bei L = 4 ist die Konfiguration L = 2
    assert r["exact"] and r["job"] == "main/basis" and r["config"]["lead"] == 2
    r = R.resolve_z(DATA, C.M_WINDOW, 0.2, 0.0, 2, 20, 0.5)                                     # p 20 verdoppelt = p 40
    assert r["exact"] and r["job"] == "p40/basis" and r["config"]["window"] == 40
    r = R.resolve_z(DATA, C.M_LEAD, 0.2, 0.0, 2, 10, 0.5)
    assert r["exact"] and r["job"] == "main/L_halbiert"                                         # gleiche Konfiguration wie L1/basis: die Maßnahme selbst gewinnt


def test_z_falls_back_to_the_nearest_measured_configuration_with_a_note():
    r = R.resolve_z(DATA, C.M_RETAIL, 0.3, 0.8, 4, 6, 0.3)
    assert not r["exact"] and r["job"] == "main/nur_einzelhandel_glaettet_0.3" and r["distance"] == 5 and "CV 0,3, ρ 0,8, L 4, p 6, γ 0,3" in r["note"] and r["z"] == DATA["jobs"][r["job"]]["z"]
    r = R.resolve_z(DATA, C.M_SMOOTH, 0.2, 0.0, 2, 10, 0.7)
    assert r["exact"] and r["config"]["gamma"] == 0.7
    r = R.resolve_z(DATA, C.M_RETAIL, 0.2, 0.0, 2, 10, 0.7)                                     # γ 0,7 nicht gemessen: die nächste Stufe 0,5
    assert not r["exact"] and r["config"]["gamma"] == 0.5 and "γ 0,7" in r["note"]


def test_z_prefers_the_candidate_that_matches_the_more_influential_parameter():
    r = R.resolve_z(DATA, C.M_BASIS, 0.2, 0.5, 4, 10, 0.5)                                       # ρ 0,5 / L 4: beide Nachbarn einen Schritt entfernt
    assert r["job"] == "L4/basis"                                                                # L stimmt, ρ weicht ab (nicht umgekehrt)
    r = R.resolve_z(DATA, C.M_BASIS, 0.3, 0.5, 2, 10, 0.5)                                       # ρ vor CV
    assert r["job"] == "rho0.5/basis"
    r = R.resolve_z(DATA, C.M_SMOOTH, 0.2, 0.0, 1, 10, 0.7)                                      # L vor γ: L 1 mit γ 0,5 statt L 2 mit γ 0,7
    assert r["job"] == "L1/glaettung_0.5"


def test_z_for_every_combination_and_measure_is_available():
    for cv, rho, lead, window, gamma in itertools.product(C.CV_OPTIONS, C.RHO_OPTIONS, C.LEAD_OPTIONS, C.WINDOW_OPTIONS, C.GAMMA_OPTIONS):
        for m in R.active_measures(lead, window):
            r = R.resolve_z(DATA, m, cv, rho, lead, window, gamma)
            assert len(r["z"]) == 4 and (r["exact"] == (r["note"] == "")) and r["job"] in DATA["jobs"]


def test_coverage_counts():
    cov = R.coverage(DATA)
    assert cov["combos"] == 108 and cov["cell_exact"] == 10 and cov["by_distance"] == {0: 10, 1: 30, 2: 44, 3: 24} and sum(cov["by_distance"].values()) == 108
    assert cov["z_cases"] == 1341 and cov["z_exact"] == 46 and cov["cells"] == list(R.CELLS)
    assert cov["z_per_measure"][C.M_BASIS] == [108, 10] and cov["z_per_measure"][C.M_INFO_SMOOTH][1] == 1 and cov["z_per_measure"][C.M_LEAD][0] == 72 and cov["z_per_measure"][C.M_WINDOW][0] == 81


# ---------------------------------------------------------------------------------------------------
# Tabellen der Kernabschnitte
# ---------------------------------------------------------------------------------------------------
def test_main_rows_are_the_13_measures_in_plan_order():
    rows = R.main_rows(DATA)
    assert [r["key"] for r in rows] == MAIN_KEYS and len(rows) == 13 and rows[0]["delta"] == 0.0 and rows[0]["label"] == "Basis"
    by = {r["key"]: r for r in rows}
    assert by["L_halbiert"]["delta"] == pytest.approx(-65.05, abs=0.02) and by["glaettung_0.3"]["delta"] == pytest.approx(-17.3, abs=0.1)
    assert by["nur_einzelhandel_glaettet_0.3"]["delta"] == pytest.approx(-30.8, abs=0.1) and by["nur_fabrik_glaettet_0.3"]["delta"] == pytest.approx(1.6, abs=0.1)
    assert by["basis"]["amp"][3] == pytest.approx(11.06, abs=0.01) and len(by["basis"]["z"]) == 4 and all(len(r["amp"]) == 4 and len(r["beta"]) == 4 for r in rows)
    assert by["basis"]["cost"] == pytest.approx(450.7, abs=0.1) and by["L_halbiert"]["L"] == 1 and by["Fenster_doppelt"]["p"] == 20


def test_fixed_rows_show_the_service_drop():
    rows = {r["key"]: r for r in R.fixed_rows(DATA)}
    assert rows["glaettung_0.5"]["total_fixed"] < rows["basis"]["total_fixed"] and min(rows["glaettung_0.3"]["beta_fixed"]) < 0.80 and min(rows["basis"]["beta_fixed"]) > 0.93
    assert rows["glaettung_0.5"]["cost_fixed"] > rows["basis"]["cost_fixed"] and rows["glaettung_0.5"]["cost_equal"] < rows["basis"]["cost_equal"]
    assert rows["glaettung_0.5"]["backlog_fixed"] > 2 * rows["basis"]["backlog_fixed"]


def test_payer_rows_show_that_the_smoothing_stage_pays():
    p = R.payer_rows(DATA, 0.3)
    assert p["gamma_all"] == 0.3 and p["gamma_retail"] == 0.3 and p["all_change"][0] == pytest.approx(29.0, abs=0.1) and p["retail_change"][0] == pytest.approx(36.5, abs=0.1)
    assert p["all_change"][3] == pytest.approx(-37.3, abs=0.1) and p["retail_change"][3] == pytest.approx(-42.3, abs=0.1) and p["retail_change"][1] < -30
    q = R.payer_rows(DATA, 0.7)
    assert q["gamma_all"] == 0.7 and q["gamma_retail"] == 0.5                                       # nur Einzelhandel: γ 0,7 nicht gemessen -> 0,5
    assert len(q["base"]) == 4 and q["all_change"][0] == pytest.approx(7.1, abs=0.1)


def test_regime_rows_for_smoothing_and_info():
    sm = {r["cell"]: r for r in R.regime_rows(DATA, C.M_SMOOTH, 0.5)}
    assert list(sm) == list(R.CELLS)
    assert {c: r["state"] for c, r in sm.items()} == {"main": "senkt", "rho0.5": "senkt", "rho0.8": "kostet", "L1": "wenig", "L4": "wenig", "p6": "senkt", "p20": "wenig", "p40": "wenig", "cv0.1": "wenig", "cv0.3": "senkt"}
    assert sm["rho0.8"]["delta"] == pytest.approx(6.9, abs=0.1) and sm["p6"]["delta"] == pytest.approx(-47.4, abs=0.1) and sm["cv0.3"]["delta"] == pytest.approx(-21.3, abs=0.1)
    inf = {r["cell"]: r for r in R.regime_rows(DATA, C.M_INFO, 0.5)}
    assert {c: r["state"] for c, r in inf.items()} == {c: ("wenig" if c == "p40" else "senkt") for c in R.CELLS}
    assert inf["L4"]["delta"] == pytest.approx(-49.3, abs=0.1) and inf["p40"]["delta"] == pytest.approx(-1.9, abs=0.1) and inf["main"]["gamma"] is None and sm["main"]["gamma"] == 0.5
    assert R.regime_rows(DATA, C.M_SMOOTH, 0.7)[0]["gamma"] == 0.7 and R.regime_rows(DATA, C.M_SMOOTH, 0.7)[1]["gamma"] == 0.5          # nur das Hauptszenario hat 0,7


def test_spearman_of_monotone_series():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0) and R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert -1.0 <= R.spearman([3, 1, 2, 5, 4], [1, 5, 2, 3, 4]) <= 1.0


def test_verstaerkung_is_a_weak_cost_proxy():
    p = R.proxy_rows(DATA)
    assert p["n_all"] == 13 and p["n_no_levers"] == 10 and p["rho_all"] == pytest.approx(0.451, abs=0.001) and p["rho_no_levers"] == pytest.approx(0.273, abs=0.001)
    assert {q["key"] for q in p["points"] if q["lever"]} == set(R.LEVER_JOBS) and all(q["amp4"] > 0 and q["total"] > 0 for q in p["points"])


def test_formula_rows_and_the_product_formula_is_only_a_lower_bound():
    rows = R.formula_rows(DATA)
    assert len(rows) == 12
    ratios = {(r["m"], r["p"]): r["ratio4"] for r in rows}
    assert min(ratios.values()) == pytest.approx(1.03, abs=0.005) and max(ratios.values()) == pytest.approx(3.43, abs=0.005) and ratios[(1, 20)] < ratios[(4, 3)]
    assert all(r["ratio4"] > 1.0 and r["exact"][0] == pytest.approx(r["product"][0], rel=1e-5) for r in rows)                          # Stufe 1 gleich, Stufe 4 echt größer
    c = R.formula_case(DATA, 2, 5)
    assert c["sim"][3] == pytest.approx(37.85, abs=0.01) and c["exact"][3] == pytest.approx(37.70, abs=0.01) and c["product"][3] == pytest.approx(20.20, abs=0.01)
    assert ratios[(2, 10)] == pytest.approx(ratios[(1, 5)], rel=1e-4)                                                                    # nur m/p zählt


def test_window_jump_rows_have_the_minimum_at_p32_for_jumps_and_fall_monotonically_without():
    rows = R.window_jump_rows(DATA)
    assert [r["p"] for r in rows] == [4, 8, 16, 32, 64]
    iid, shift = [r["iid"] for r in rows], [r["shift"] for r in rows]
    assert iid == sorted(iid, reverse=True) and shift.index(min(shift)) == 3 and shift[4] > shift[3] * 1.04 and rows[0]["iid"] > 1000


def test_robustness_of_the_repetition():
    r = R.robustness(DATA)
    assert r["n"] == 13 and r["spearman"] == 1.0 and -0.4 <= r["min"] < 0 < r["max"] <= 0.52


def test_data_modified_copies_change_the_verdict(monkeypatch=None):
    d = copy.deepcopy(DATA)
    d["jobs"]["main/glaettung_0.5"]["equal"]["total_onhand"] = d["jobs"]["main/basis"]["equal"]["total_onhand"] * 1.05
    r = R.resolve_pair(d, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)
    assert r["state"] == C.STATE_COSTS and r["delta"] == pytest.approx(5.0)
    d["jobs"]["main/glaettung_0.5"]["equal"]["total_onhand"] = d["jobs"]["main/basis"]["equal"]["total_onhand"] * 0.99
    assert R.resolve_pair(d, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)["state"] == C.STATE_LITTLE


# ---------------------------------------------------------------------------------------------------
# Lücken aus dem Fehler-Einbau-Test
# ---------------------------------------------------------------------------------------------------
def test_cell_entry_ignores_the_smoothing_level_for_measures_without_one():
    assert R.cell_entry(DATA, "main", C.M_BASIS, 0.5)["name"] == "main/basis" and R.cell_entry(DATA, "main", C.M_INFO, 0.3)["name"] == "main/info_teilen"
    assert R.cell_entry(DATA, "main", C.M_SMOOTH, 0.7)["name"] == "main/glaettung_0.7" and R.cell_entry(DATA, "main", C.M_RETAIL, 0.5)["name"] == "main/nur_einzelhandel_glaettet_0.5"
    with pytest.raises(KeyError):
        R.cell_entry(DATA, "main", C.M_SMOOTH, 0.99)


def test_nearest_smoothing_level_prefers_the_larger_one_on_a_tie():
    assert R._nearest_gamma([0.7, 0.3], 0.5) == 0.7 and R._nearest_gamma([0.3, 0.7], 0.5) == 0.7 and R._nearest_gamma([0.7, 0.5, 0.3], 0.3) == 0.3 and R._nearest_gamma([0.5], 0.7) == 0.5
    d = copy.deepcopy(DATA)
    del d["jobs"]["main/glaettung_0.5"]
    assert R.resolve_pair(d, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)["gamma"] == 0.7                          # 0,7 und 0,3 liegen gleich weit: die größere Stufe
    assert R.resolve_pair(d, C.M_SMOOTH, 0.3, 0.2, 0.0, 2, 10)["gamma"] == 0.3 and R.payer_rows(d, 0.5)["gamma_all"] == 0.7
    assert [r["gamma"] for r in R.regime_rows(d, C.M_SMOOTH, 0.5)][0] == 0.7


def test_pair_without_a_fixed_z_block_has_no_fixed_change():
    d = copy.deepcopy(DATA)
    d["jobs"]["main/glaettung_0.5"]["fixed"] = None
    assert "fixed_delta" not in R.resolve_pair(d, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)
    d2 = copy.deepcopy(DATA)
    d2["jobs"]["main/basis"]["fixed"] = None
    assert "fixed_delta" not in R.resolve_pair(d2, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10) and "fixed_delta" in R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)


def test_the_note_about_main_only_measures_appears_exactly_when_the_measure_sits_in_the_main_scenario_and_the_setting_is_not_exact():
    for combo in itertools.product(C.CV_OPTIONS, C.RHO_OPTIONS, C.LEAD_OPTIONS, C.WINDOW_OPTIONS):
        cv, rho, lead, window = combo
        for m in (C.M_SMOOTH, C.M_INFO, C.M_RETAIL, C.M_INFO_SMOOTH, C.M_LEAD, C.M_WINDOW):
            r = R.resolve_pair(DATA, m, 0.5, cv, rho, lead, window)
            has_note = "nur im Hauptszenario gemessen" in r["note"]
            assert has_note == (m in (C.M_LEAD, C.M_WINDOW, C.M_RETAIL, C.M_INFO_SMOOTH) and r["cell"] == "main" and not r["cell_exact"]), (m, combo)


def test_z_tie_break_prefers_the_candidate_whose_most_influential_differing_parameter_is_least_influential():
    r = R.resolve_z(DATA, C.M_BASIS, 0.3, 0.5, 4, 10, 0.5)              # L4 differs in (CV, rho), rho0.5 in (L, CV): the candidate that matches L wins
    assert r["job"] == "L4/basis" and r["distance"] == 2
    r = R.resolve_z(DATA, C.M_BASIS, 0.3, 0.5, 1, 10, 0.5)
    assert r["job"] == "L1/basis"


def test_window_jump_rows_carry_the_service_of_stage_one_at_fixed_z_and_the_values():
    rows = R.window_jump_rows(DATA)
    assert [round(100 * r["beta1_fixed_shift"], 1) for r in rows] == [93.6, 96.2, 97.5, 97.7, 96.9]
    assert [round(100 * r["beta1_fixed_iid"], 1) for r in rows] == [round(100 * DATA["jobs"][f"ar1/p{p}/basis"]["fixed"]["beta"][0], 1) for p in (4, 8, 16, 32, 64)]
    assert [round(r["shift"]) for r in rows] == [1844, 438, 202, 168, 176] and [round(r["iid"]) for r in rows] == [1807, 390, 169, 123, 107]
    assert rows[3]["shift_se"] > 0 and rows[0]["iid_se"] > 0 and rows[0]["amp4_iid"] == pytest.approx(342.1, abs=0.1) and rows[0]["amp4_shift"] == pytest.approx(138.3, abs=0.1)


def test_regime_rows_carry_the_base_total_and_the_amplification_of_stage_four():
    sm = {r["cell"]: r for r in R.regime_rows(DATA, C.M_SMOOTH, 0.5)}
    assert sm["main"]["base_total"] == pytest.approx(270.158, abs=0.001) and sm["L4"]["base_total"] == pytest.approx(997, abs=1) and sm["main"]["setting"] == (0.2, 0.0, 2, 10)
    assert sm["main"]["amp4_base"] == pytest.approx(11.06, abs=0.01) and sm["main"]["amp4"] == pytest.approx(2.89, abs=0.01) and sm["rho0.8"]["amp4_base"] == pytest.approx(30.26, abs=0.01)
    assert sm["main"]["group"] == "Hauptszenario" and sm["p6"]["group"] == "Prognosefenster" and sm["main"]["label"] == "Hauptszenario"


def test_the_smoothing_of_the_measured_cell_is_taken_from_that_cell_only():
    inf = R.regime_rows(DATA, C.M_INFO, 0.3)
    assert all(r["gamma"] is None for r in inf)
    sm = R.regime_rows(DATA, C.M_SMOOTH, 0.3)
    assert sm[0]["gamma"] == 0.3 and all(r["gamma"] == 0.5 for r in sm[1:])


def test_nearest_smoothing_level_on_an_exact_tie_prefers_the_level_closer_to_the_default_then_the_larger():
    assert R._nearest_gamma([0.5, 0.25], 0.375) == 0.5 and R._nearest_gamma([0.25, 0.5], 0.375) == 0.5              # gleich weit: die Stufe näher am Standard 0,5
    assert R._nearest_gamma([0.75, 0.25], 0.5) == 0.75 and R._nearest_gamma([0.25, 0.75], 0.5) == 0.75             # auch gleich weit vom Standard: die größere


def test_pair_at_the_lowest_smoothing_level_is_exact_and_the_default_level_wins_a_tie_only_by_distance():
    r = R.resolve_pair(DATA, C.M_SMOOTH, 0.3, 0.2, 0.0, 2, 10)
    assert r["gamma"] == 0.3 and r["gamma_exact"] and r["name"] == "main/glaettung_0.3" and r["delta"] == pytest.approx(-17.3, abs=0.1)
    r7 = R.resolve_pair(DATA, C.M_SMOOTH, 0.7, 0.2, 0.0, 2, 10)
    assert r7["gamma"] == 0.7 and r7["name"] == "main/glaettung_0.7"
