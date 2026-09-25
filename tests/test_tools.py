"""Werkzeuge in tools/: Ergebnisdatei bauen (Rundung, Struktur, Idempotenz), Testreferenz einfrieren, Preset-Suche, Schnellkalibrierungs-Prüfung, Bau-Gate-Hilfen, Sweep-Aufbau, Mutantenliste und
Mutationswerkzeug, CI-Datei und Abhängigkeiten. Läufe, die die Messreihe brauchen (messreihe_bullwhip/), werden ohne sie übersprungen (CI)."""
import json
import os
import pathlib

import numpy as np
import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_model as M
import bw_results as R
from toolload import load_tool

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = pathlib.Path(os.environ.get("BW_SOURCE") or ROOT.parent / "bestand-planung" / "messreihe_bullwhip")     # BW_SOURCE: vom Mutationswerkzeug für die Kopie gesetzt
HAVE_SOURCE = (SOURCE / "sweep_data.json").exists() and (SOURCE / "bull.py").exists()
BR = load_tool("build_results")
FR = load_tool("freeze_reference")
TP = load_tool("tune_presets")
QC = load_tool("quick_check")
CF = load_tool("check_full")
SW = load_tool("sweep")
GM = load_tool("gen_mutants")
MC = load_tool("mutation_check")
MUTATING = os.environ.get("BW_MUTATION_RUN") == "1"


# ---------------------------------------------------------------------------------------------------
# build_results
# ---------------------------------------------------------------------------------------------------
def test_rounding_keeps_six_significant_digits_and_leaves_other_types_alone():
    assert BR.rnd(270.15778009428266) == 270.158 and BR.rnd(0.000123456789) == 0.000123457 and BR.rnd(718126.55555) == 718127.0 and BR.rnd(1.0) == 1.0
    assert BR.rnd(7) == 7 and BR.rnd(True) is True and BR.rnd(None) is None and BR.rnd("text") == "text" and BR.rnd((1.23456789, 2)) == [1.23457, 2]
    assert BR.rnd({"a": [1.23456789, {"b": 2.34567891}]}) == {"a": [1.23457, {"b": 2.34568}]} and BR.rnd(1.23456789, 3) == 1.23 and BR.rnd(object) is object


def synthetic_sweep():
    stats = lambda base: {"amp": [[base + k, 0.01 * k, 9.9] for k in range(4)], "beta": [0.95] * 4, "alpha": [0.8] * 4, "onhand": [10.0 + k for k in range(4)], "backlog": [5.0] * 4,
                          "cost": [20.0] * 4, "total_onhand": 46.0, "total_onhand_se": 0.5, "total_cost": 80.0, "total_cost_se": 1.0, "zero_share": [0.0, 0.0, 0.01, 0.05]}
    job = lambda name, fixed: {"name": name, "scenario": {"kind": "ar1", "cv": 0.2, "rho": 0.0, "L": 2, "p": 10, "seed": 11464}, "policy": {}, "L": 2, "p": 10, "z": [1.0, 1.1, 1.2, 1.3],
                               "z_at_cap": [False] * 4, "insample_beta": [0.95] * 4, "equal_service": stats(1.0), **({"fixed_z": stats(2.0)} if fixed else {})}
    e2 = {"cv": 0.1, "m": 1, "p": 5, "linear_exact": [1.0, 2.0, 3.0, 4.0], **{k: {"amp": [1.0] * 4, "se": [0.1] * 4, "zero_share": [0.0] * 4, "extra": 1} for k in ("z0_nonneg", "z1645_nonneg", "z1645_linear")}}
    return {"config": {"N": 4}, "E1": [{"m": 1, "p": 3}], "E2": [e2], "E3": [{"rho": 0.0}], "jobs": [job("main/basis", True), job("main2/basis", False)]}


def test_build_selects_the_fields_the_app_shows():
    out = BR.build(synthetic_sweep())
    assert list(out) == ["_meta", "E1", "E2", "E3", "jobs"] and out["_meta"]["n_jobs"] == 2 and out["E1"] == [{"m": 1, "p": 3}] and out["E3"] == [{"rho": 0.0}]
    j = out["jobs"]["main/basis"]
    assert set(j) == {"scenario", "policy", "L", "p", "z", "z_at_cap", "equal", "fixed"} and set(j["scenario"]) == {"kind", "cv", "rho", "L", "p", "seed"}
    assert j["equal"]["amp"] == [[1.0, 0.0], [2.0, 0.01], [3.0, 0.02], [4.0, 0.03]] and "alpha" not in j["equal"] and set(j["equal"]) == {"amp", "beta", "onhand", "backlog", "cost", "zero_share", "total_onhand", "total_onhand_se", "total_cost", "total_cost_se"}
    assert j["fixed"]["amp"][0] == [2.0, 0.0] and out["jobs"]["main2/basis"]["fixed"] is None
    assert set(out["E2"][0]) == {"cv", "m", "p", "linear_exact", "z0_nonneg", "z1645_nonneg", "z1645_linear"} and "extra" not in out["E2"][0]["z0_nonneg"]


def test_dumps_is_compact_deterministic_and_ends_with_a_newline():
    text = BR.dumps({"b": 1, "a": [1.5, "ä"]})
    assert text == '{"b":1,"a":[1.5,"ä"]}\n' and BR.dumps({"b": 1, "a": [1.5, "ä"]}) == text


def test_the_shipped_results_file_is_in_the_canonical_format():
    text = (ROOT / "data" / "bw_results.json").read_text(encoding="utf-8")
    assert BR.dumps(json.loads(text)) == text and "\r" not in text
    assert BR.OUT == ROOT / "data" / "bw_results.json" and BR.SIGNIFICANT == 6


@pytest.mark.skipif(not HAVE_SOURCE, reason="Messreihe messreihe_bullwhip/ nicht vorhanden")
def test_the_shipped_results_file_is_exactly_the_build_of_the_measurement_series(tmp_path, monkeypatch):
    monkeypatch.setattr(BR, "OUT", tmp_path / "data" / "bw_results.json")
    assert BR.main([str(SOURCE)]) == 0
    assert (tmp_path / "data" / "bw_results.json").read_text(encoding="utf-8") == (ROOT / "data" / "bw_results.json").read_text(encoding="utf-8")
    assert BR.main([str(SOURCE / "sweep_data.json")]) == 0


# ---------------------------------------------------------------------------------------------------
# freeze_reference
# ---------------------------------------------------------------------------------------------------
def test_enc_and_dec_round_trip_bit_exactly():
    a = np.random.default_rng(1).normal(size=(3, 5))
    assert np.array_equal(FZ.dec(FR.enc(a)), a) and FR.enc(a) == FZ.enc(a) and FR.enc(a)["shape"] == [3, 5]


def test_freeze_cases_and_demands_are_consistent():
    names = [c[0] for c in FR.CASES]
    assert len(names) == len(set(names)) == 11 and all(c[1] in FR.DEMANDS for c in FR.CASES) and len(FR.DEMANDS) == 5
    assert FR.BURN == 50 and FR.CAL_KW["target"] == 0.95 and [c[0] for c in FR.CALIBRATIONS] == ["basis", "info"]
    assert names == [c["name"] for c in FZ.reference()["cases"]] and set(FR.DEMANDS) == set(FZ.reference()["demands"])


def test_make_demand_and_evaluate_use_the_given_module():
    D = FR.make_demand(M, FR.DEMANDS["iid"])
    assert D.shape == (3, 300) and np.array_equal(D, FR.make_demand(M, FR.DEMANDS["iid"])) and FR.make_demand(M, FR.DEMANDS["shift"]).shape == (2, 300)
    name, dname, Ls, p, z, kw = FR.CASES[1]
    q, ph, vr = FR.evaluate(M, D, Ls, p, z, kw)
    rec = FR.case_record(name, dname, D, Ls, p, z, kw, q, ph, vr)
    assert rec["name"] == "gamma0.5" and np.array(rec["expected"]["q_sum"]).shape == (4, 3) and np.array(rec["expected"]["q_tail"]).shape == (4, 3, 5) and len(rec["expected"]["amp"]) == 4
    assert rec["expected"]["q_zero_count"] == int((q <= 1e-12).sum()) and rec["kw"] == {"gamma": 0.5} and rec["Ls"] == [2] * 4


@pytest.mark.skipif(not HAVE_SOURCE or np.__version__ != FZ.reference()["numpy"], reason="Messreihe nicht vorhanden oder andere NumPy-Version")
def test_freezing_again_reproduces_the_shipped_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "OUT_DIR", tmp_path)
    assert FR.main([str(SOURCE)]) == 0
    for name in ("bw_reference.json", "bw_preset_demand.json"):
        assert (tmp_path / name).read_text(encoding="utf-8") == (ROOT / "tests" / "data" / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# tune_presets, quick_check, check_full
# ---------------------------------------------------------------------------------------------------
def test_tune_presets_main_prints_the_good_seeds_closest_to_the_measured_mean(monkeypatch, capsys):
    good = {"Standard": (True, [], -15.0), "Kurze Lieferzeit": (True, [], 0.0), "Träger Bedarf": (True, [], 0.0), "Kleines Fenster": (True, [], 0.0), "Großes Fenster": (True, [], 0.0)}
    bad = dict(good, Standard=(True, [], -30.0))
    worse = dict(good, **{"Großes Fenster": (False, [], 0.0)})
    table = {200: bad, 201: good, 202: worse}
    monkeypatch.setattr(TP, "evaluate", lambda seed, presets=None: table[seed])
    assert TP.main(["200", "203"]) == 0
    out = capsys.readouterr().out
    assert "Seed 200: +++++" in out and "Seed 202: ++++-" in out and "[201, 200]" in out and "(Messreihe Mittel -15,7 %)".replace(",", ".") in out


def test_tune_presets_default_range_and_jobs_option(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(TP, "_one", lambda seed: (seen.append(seed), (seed, "+++++", -15.7))[1])
    assert TP.main([]) == 0
    assert seen == list(range(200, 300))


def test_quick_check_summary_of_synthetic_results():
    data = R.load_results()
    full = data["jobs"]["main/basis"]["z"]
    results = [("Basis", 300, [z + 0.1 for z in full], [0.95, 0.96, 0.955, 0.95], 280.0, 2.0), ("Basis", 301, [z - 0.1 for z in full], [0.94, 0.95, 0.95, 0.96], 290.0, 4.0)]
    results += [(label, 300, data["jobs"][f"main/{job}"]["z"], [0.95] * 4, 100.0, 1.0) for label, (_, job) in QC.MEASURES.items() if label != "Basis"] * 2
    s = QC.summarize(results, data)["Basis"]
    assert s["n"] == 2 and s["z_mean"] == pytest.approx(full) and s["z_diff"] == pytest.approx([0.0] * 4, abs=1e-12) and s["z_max_abs_diff"] == pytest.approx(0.1)
    assert s["beta_min"] == 0.94 and s["beta_max"] == 0.96 and s["total_mean"] == 285.0 and s["seconds"] == 3.0 and s["z_sd"][0] == pytest.approx(0.1414213562, abs=1e-6)
    assert s["total_full"] == pytest.approx(270.158, abs=0.001) and list(QC.MEASURES) == ["Basis", "Glättung γ 0,5", "Info teilen"]


def test_quick_check_one_calibration_and_evaluation(monkeypatch):
    monkeypatch.setattr(QC, "EVAL_R", 5)
    monkeypatch.setattr(QC, "EVAL_T", 250)
    monkeypatch.setattr(QC.LV, "quick_calibrate", lambda cv, rho, lead, window, gamma, seed, measures, **k: {"z": {measures[0]: [1.0, 1.1, 1.2, 1.3]}, "skipped": [], "seconds": 0.1})
    label, seed, z, beta, total, seconds = QC.one(("Info teilen", 305))
    assert (label, seed, z) == ("Info teilen", 305, [1.0, 1.1, 1.2, 1.3]) and len(beta) == 4 and all(0.5 < b <= 1 for b in beta) and total > 0 and seconds >= 0


def test_check_full_flatten():
    assert CF.flatten({"a": [1, {"b": 2.5}], "c": "x"}) == {"/a[0]": 1, "/a[1]/b": 2.5, "/c": "x"} and CF.flatten([]) == {} and CF.flatten(3) == {"": 3}
    assert CF.MEASURED_WITH_NUMPY == FZ.reference()["numpy"]


# ---------------------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------------------
def test_the_sweep_defines_the_63_jobs_of_the_shipped_results_with_the_documented_seeds():
    jobs = SW.make_jobs()
    assert [j["name"] for j in jobs] == list(R.load_results()["jobs"]) and len(jobs) == 63
    main = next(j for j in jobs if j["name"] == "main/basis")
    assert main["scenario"]["seed"] == 11464 and main["fixed"] is True and next(j for j in jobs if j["name"] == "main2/basis")["scenario"]["seed"] == 11464 + 777
    assert next(j for j in jobs if j["name"] == "main2/basis")["fixed"] is False and min(j["scenario"]["seed"] for j in jobs) >= 10000
    assert (SW.N_STAGES, SW.MU, SW.TARGET, SW.BURN, SW.R_CAL, SW.T_CAL, SW.R_EVAL, SW.T_EVAL) == (4, 100.0, 0.95, 100, 200, 1200, 500, 1500)


def test_a_small_calibration_job_returns_the_result_fields(monkeypatch):
    monkeypatch.setattr(SW, "R_CAL", 6)
    monkeypatch.setattr(SW, "T_CAL", 250)
    monkeypatch.setattr(SW, "R_EVAL", 8)
    monkeypatch.setattr(SW, "T_EVAL", 300)
    r = SW.run_policy(next(j for j in SW.make_jobs() if j["name"] == "main/glaettung_0.5"))
    assert set(r) == {"scenario", "policy", "name", "L", "p", "z", "z_at_cap", "insample_beta", "equal_service", "fixed_z"} and len(r["z"]) == 4
    assert set(r["equal_service"]) == {"amp", "beta", "alpha", "onhand", "backlog", "cost", "total_onhand", "total_onhand_se", "total_cost", "total_cost_se", "zero_share"} and len(r["equal_service"]["amp"][0]) == 3


def test_sweep_helpers():
    assert SW.se_of([1.0, 3.0]) == pytest.approx(1.0) and SW.demand("ar1", 3, 2, 50, 0.2, 0.0).shape == (2, 50)
    with pytest.raises(ValueError):
        SW.demand("gibt es nicht", 1, 2, 3, 0.2, 0.0)


# ---------------------------------------------------------------------------------------------------
# Mutationswerkzeug und Mutantenliste
# ---------------------------------------------------------------------------------------------------
def test_candidates_find_operators_numbers_and_skip_unary_signs():
    cands = GM.candidates("x = a < b and c + 1\ny = -2\nz = max(a, b)\nw = True\n")
    ops = {(c[2]) for c in cands}
    assert "<=" in ops and "or" in ops and "-" in ops and "2" in ops and "min" in ops and "False" in ops
    src = "y = -2\n"
    assert all(src[c[0]:c[0] + c[1]] != "-" for c in GM.candidates(src))                               # unäres Minus wird nicht mutiert
    assert GM.candidates("m = max\n") == []                                                            # max ohne Aufruf zählt nicht
    assert any(c[2] == "0.11" for c in GM.candidates("a = 0.1\n")) and any(c[2] == "0.1" for c in GM.candidates("a = 0.0\n"))


def test_unique_window_returns_the_shortest_unique_surroundings():
    src = "alpha = beta + 1\ngamma = delta + 1\n"
    off = src.index("+ 1")
    old, new = GM.unique_window(src, off, 1, "-")
    assert src.count(old) == 1 and new == old.replace("+", "-", 1) and "\n" not in old
    assert GM.unique_window("aaaa\naaaa\n", 1, 1, "b") is None


def test_mutation_check_runs_the_module_tests_first_and_skips_slow_and_app_tests():
    core = MC.build_args("bw_model.py")
    assert core[0] == "tests/test_model_units.py" and "tests/test_checks.py" in core and "tests/test_frozen_reference.py" in core and "tests/test_app.py" not in core
    other = MC.build_args("bw_results.py")
    assert other[0] == "tests/test_results.py" and "tests/test_checks.py" not in other and "tests/test_frozen_reference.py" not in other and "tests/test_app.py" not in other
    assert MC.CORE == ("bw_model.py", "bw_theory.py")
    for name, first in MC.FIRST.items():
        assert (ROOT / name).exists() and all((ROOT / "tests" / f).exists() for f in first), name


@pytest.mark.skipif(MUTATING, reason="in der Mutantenkopie verändert sich der Quelltext")
def test_every_mutant_site_occurs_exactly_once_in_its_module():
    from mutants import MUTANTS
    assert len(MUTANTS) >= 500 and len(set(MUTANTS)) == len(MUTANTS)
    sources = {}
    for name, old, new in MUTANTS:
        assert old != new, (name, old)
        src = sources.setdefault(name, (ROOT / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8"))
        assert src.count(old) == 1, (name, old)
    assert set(n for n, _, _ in MUTANTS) <= set(GM.TARGETS) and {"bw_model.py", "bw_theory.py", "bw_live.py", "bw_results.py", "bw_stories.py", "bw_ui_panel.py"} <= {n for n, _, _ in MUTANTS}


def test_the_mutant_list_documents_its_equivalent_survivors():
    import mutants
    assert "überlebend" in mutants.EQUIVALENT_NOTES and "Einschätzung, kein Beweis" in mutants.EQUIVALENT_NOTES


# ---------------------------------------------------------------------------------------------------
# CI-Datei und Abhängigkeiten
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(MUTATING, reason="Projektdateien")
def test_ci_and_requirements_follow_the_portfolio_template():
    ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in ci and "pip install -r requirements-dev.txt" in ci and "python -m pytest tests/ -v --durations=10" in ci and "workflow_dispatch:" in ci
    assert 'cron: "17 6 * * 1"' in ci and "branches: [main]" in ci and "actions/checkout@v7" in ci and "actions/setup-python@v7" in ci
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").split()
    assert req == ["streamlit>=1.51", "numpy>=1.26", "pandas>=2.0", "plotly>=5.20", "fpdf2>=2.7"] and all(">=" in r and "==" not in r for r in req) and "ortools" not in " ".join(req)
    assert (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").split() == ["-r", "requirements.txt", "pytest>=8.0"]
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for line in ("venv/", "__pycache__/", ".pytest_cache/", "tools/_out/"):
        assert line in ignore.splitlines()
