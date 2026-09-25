"""Werkzeug-Rahmen (Lücken aus dem Fehler-Einbau-Test): Hauptprogramme von build_results, check_full, quick_check und tune_presets mit synthetischer Messreihe und Ersatz-Prozesspool, dazu die
Konstanten des Einfrier-Werkzeugs gegen die ausgelieferte Referenz. Ohne die echte Messreihe, ohne Wartezeiten."""
import json

import numpy as np
import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_model as M
import bw_results as R
from test_tools import BR, CF, FR, QC, TP, synthetic_sweep


class FakePool:
    """Ersatz für multiprocessing.Pool: rechnet nacheinander im selben Prozess und merkt sich die Größe und die Aufgaben."""
    created = []
    tasks = []

    def __init__(self, processes):
        FakePool.created.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items, chunksize=None):
        items = list(items)
        FakePool.tasks.extend(items)
        return [fn(i) for i in items]


@pytest.fixture
def fake_pool():
    FakePool.created, FakePool.tasks = [], []
    return FakePool


# ---------------------------------------------------------------------------------------------------
# build_results.main
# ---------------------------------------------------------------------------------------------------
def test_build_results_main_writes_the_file_creates_the_folder_and_takes_a_directory_or_a_file(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "sweep_data.json").write_text(json.dumps(synthetic_sweep()), encoding="utf-8")
    out = tmp_path / "data" / "bw_results.json"
    monkeypatch.setattr(BR, "OUT", out)
    assert BR.main([str(src)]) == 0
    assert out.read_text(encoding="utf-8") == BR.dumps(BR.build(synthetic_sweep())) and "geschrieben:" in capsys.readouterr().out
    assert BR.main([str(src / "sweep_data.json")]) == 0                                                # Ordner existiert schon
    monkeypatch.setattr(BR, "DEFAULT_SOURCE", src / "sweep_data.json")
    assert BR.main([]) == 0
    out.unlink()
    monkeypatch.setattr(BR, "DEFAULT_SOURCE", src)
    assert BR.main([]) == 0 and out.exists()


# ---------------------------------------------------------------------------------------------------
# check_full.main
# ---------------------------------------------------------------------------------------------------
def make_check_full_env(tmp_path, monkeypatch, checks=47, numpy_match=True, tamper=None, sweep=None):
    sweep = sweep or synthetic_sweep()
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "sweep_data.json").write_text(json.dumps(sweep), encoding="utf-8")
    results = tmp_path / "bw_results.json"
    results.write_text(BR.dumps(BR.build(sweep)), encoding="utf-8")
    got = json.loads(json.dumps(sweep))
    if tamper:
        tamper(got)
    monkeypatch.setattr(CF, "RESULTS_PATH", results)
    monkeypatch.setattr(CF.K, "ALL_CHECKS", [(f"c{i}", lambda: (True, ""), ()) for i in range(checks)])
    monkeypatch.setattr(CF.SW, "run_all", lambda n, lg, t0=None: (lg("gerechnet"), got)[1])
    monkeypatch.setattr(CF, "MEASURED_WITH_NUMPY", np.__version__ if numpy_match else "0.0.0")
    return src


def test_check_full_passes_when_the_repetition_is_identical(tmp_path, monkeypatch, capsys):
    src = make_check_full_env(tmp_path, monkeypatch)
    assert CF.main([str(src)]) == 0
    out = capsys.readouterr().out
    assert "wie bei der Messreihe: Bitgleichheit wird verlangt" in out and "47 von 47 bestanden" in out and "0 Zahlen abweichend (0 davon innerhalb 1e-6 relativ)" in out
    assert "z je Stufe: 2 von 2 Jobs bitgleich" in out and "== Ausgabe von tools/build_results.py aus der wiederholten Messreihe: True" in out and "alle Checks bestanden" in out and "gerechnet" in out


def nudge_total(factor):
    def tamper(got):
        got["jobs"][0]["equal_service"]["total_onhand"] *= factor
    return tamper


def test_check_full_demands_bit_equality_only_with_the_measured_numpy(tmp_path, monkeypatch, capsys):
    src = make_check_full_env(tmp_path, monkeypatch, tamper=nudge_total(1 + 1e-9))
    with pytest.raises(AssertionError):
        CF.main([str(src)])
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=nudge_total(1 + 1e-9))                 # andere NumPy-Version: Toleranz 1e-6 relativ
    assert CF.main([str(src)]) == 0
    out = capsys.readouterr().out
    assert "anders als bei der Messreihe: nur Aggregate mit Toleranz" in out and "1 Zahlen abweichend (1 davon innerhalb 1e-6 relativ)" in out
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=nudge_total(1.001))
    with pytest.raises(AssertionError):
        CF.main([str(src)])


def test_check_full_tolerance_is_relative_but_never_finer_than_one_millionth_of_one(tmp_path, monkeypatch):
    sweep = synthetic_sweep()
    sweep["E1"][0]["sim_se"] = [0.001, 0.0, 0.0, 0.0]

    def move(delta):
        def tamper(got):
            got["E1"][0]["sim_se"][0] = 0.001 + delta
        return tamper
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=move(0.9e-6), sweep=sweep)
    assert CF.main([str(src)]) == 0                                                                    # kleiner Wert: absolut 1e-6 (max(1, |x|) = 1)
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=move(1.1e-6), sweep=sweep)
    with pytest.raises(AssertionError):
        CF.main([str(src)])
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=nudge_total(1 + 0.99e-6))
    assert CF.main([str(src)]) == 0                                                                    # großer Wert: relativ 1e-6
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=nudge_total(1 + 1.1e-6))
    with pytest.raises(AssertionError):
        CF.main([str(src)])


def test_check_full_stops_on_a_failing_check_or_the_wrong_number_of_checks(tmp_path, monkeypatch):
    for n in (48, 46):
        src = make_check_full_env(tmp_path, monkeypatch, checks=n)
        with pytest.raises(AssertionError):
            CF.main([str(src)])
    src = make_check_full_env(tmp_path, monkeypatch)
    monkeypatch.setattr(CF.K, "ALL_CHECKS", [("gut", lambda: (True, ""), ())] * 46 + [("schlecht", lambda: (False, "kaputt"), ())])
    with pytest.raises(AssertionError) as info:
        CF.main([str(src)])
    assert "schlecht" in str(info.value)


def test_check_full_skips_the_sweep_without_a_reference(tmp_path, monkeypatch, capsys):
    make_check_full_env(tmp_path, monkeypatch)
    empty = tmp_path / "leer"
    empty.mkdir()
    assert CF.main([str(empty)]) == 0
    assert "übersprungen" in capsys.readouterr().out


def test_check_full_detects_a_results_file_that_is_not_the_build_of_the_repetition(tmp_path, monkeypatch):
    src = make_check_full_env(tmp_path, monkeypatch)
    (tmp_path / "bw_results.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        CF.main([str(src)])


def test_check_full_asks_for_the_processes_of_the_machine_up_to_twelve(tmp_path, monkeypatch):
    src = make_check_full_env(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(CF.SW, "run_all", lambda n, lg, t0=None: (seen.append(n), json.loads((src / "sweep_data.json").read_text(encoding="utf-8")))[1])
    monkeypatch.setattr(CF.mp, "cpu_count", lambda: 4)
    CF.main([str(src)])
    monkeypatch.setattr(CF.mp, "cpu_count", lambda: 64)
    CF.main([str(src)])
    assert seen == [4, 12]


def test_check_full_counts_the_bit_identical_jobs(tmp_path, monkeypatch, capsys):
    def other_z(got):
        got["jobs"][1]["z"][0] += 1e-9
    src = make_check_full_env(tmp_path, monkeypatch, numpy_match=False, tamper=other_z)
    CF.main([str(src)])
    assert "z je Stufe: 1 von 2 Jobs bitgleich" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# quick_check und tune_presets
# ---------------------------------------------------------------------------------------------------
def fake_quick_results(label, seed):
    z = R.load_results()["jobs"][f"main/{QC.MEASURES[label][1]}"]["z"]
    return label, seed, list(z), [0.95, 0.95, 0.95, 0.95], 200.0, 2.0


def test_quick_check_main_uses_the_documented_seeds_and_prints_a_summary_per_measure(monkeypatch, capsys, fake_pool):
    monkeypatch.setattr(QC.mp, "Pool", fake_pool)
    monkeypatch.setattr(QC, "one", lambda task: fake_quick_results(*task))
    assert QC.main(["2", "--jobs", "3"]) == 0
    assert fake_pool.created == [3] and fake_pool.tasks == [(label, 300 + s) for label in QC.MEASURES for s in range(2)]
    out = capsys.readouterr().out
    assert "== Basis (2 Seeds): volle Kalibrierung z 1.07 1.33 1.66 1.84; Schnellkalibrierung z 1.07 1.33 1.66 1.84" in out and "größte Einzel-Abweichung 0.00" in out
    assert "Servicegrad auf unabhängiger Stichprobe (Ziel 95 %): Mittel 95.0 95.0 95.0 95.0 %, kleinster 95.0 %, größter 95.0 %" in out and "Summe Lager 200 (Standardabweichung 0.0) gegen 270 der Messreihe; 2.0 s je Kalibrierung" in out
    assert "== Glättung γ 0,5 (2 Seeds)" in out and "== Info teilen (2 Seeds)" in out


def test_quick_check_defaults_are_twelve_seeds_and_twelve_processes(monkeypatch, fake_pool):
    monkeypatch.setattr(QC.mp, "Pool", fake_pool)
    monkeypatch.setattr(QC, "one", lambda task: fake_quick_results(*task))
    assert QC.main([]) == 0
    assert fake_pool.created == [12] and len(fake_pool.tasks) == 36 and fake_pool.tasks[0] == ("Basis", 300) and fake_pool.tasks[11] == ("Basis", 311) and fake_pool.tasks[12] == ("Glättung γ 0,5", 300)


def test_quick_check_calibrates_the_measure_and_evaluates_on_an_independent_sample(monkeypatch):
    calls = {}
    monkeypatch.setattr(QC.LV, "quick_calibrate", lambda *a, **k: (calls.setdefault("cal", a), {"z": {a[6][0]: [1.0, 1.1, 1.2, 1.3]}})[1])
    real = QC.M.gen_ar1

    def spy(rng, R_, T_, mu, sigma, rho):
        calls["gen"] = (rng.standard_normal(), (R_, T_, mu, sigma, rho))
        return real(np.random.default_rng(1), 3, 250, mu, sigma, rho)
    monkeypatch.setattr(QC.M, "gen_ar1", spy)
    orders = {}
    real_orders = QC.M.order_streams
    monkeypatch.setattr(QC.M, "order_streams", lambda D, Ls, p, **k: (orders.update(kw=k, Ls=Ls, p=p), real_orders(D, Ls, p, **k))[1])
    QC.one(("Info teilen", 305))
    assert calls["cal"] == (0.2, 0.0, 2, 10, 0.5, 305, [C.M_INFO])
    first, args = calls["gen"]
    assert first == np.random.default_rng(305 + QC.EVAL_OFFSET).standard_normal() and args == (QC.EVAL_R, QC.EVAL_T, C.MU, C.MU * 0.2, 0.0)
    assert orders["kw"]["share"] is True and orders["kw"]["z"] == [1.0, 1.1, 1.2, 1.3] and orders["Ls"] == [2] * 4 and orders["p"] == 10 and (QC.EVAL_R, QC.EVAL_T) == (300, 1200)


def test_tune_presets_main_options(monkeypatch, capsys, fake_pool):
    seen = []
    monkeypatch.setattr(TP.mp, "Pool", fake_pool)
    monkeypatch.setattr(TP, "_one", lambda seed: (seen.append(seed), (seed, "+++++", -15.7))[1])
    assert TP.main(["--jobs", "2", "200", "203"]) == 0
    assert fake_pool.created == [2] and seen == [200, 201, 202]
    fake_pool.created.clear()
    seen.clear()
    assert TP.main(["210", "212", "--jobs", "4"]) == 0
    assert fake_pool.created == [4] and seen == [210, 211]
    fake_pool.created.clear()
    seen.clear()
    assert TP.main(["300", "302", "--jobs", "1"]) == 0
    assert fake_pool.created == [] and seen == [300, 301]                                     # ein Prozess: kein Pool
    assert "Seeds, die alle fünf Tageskriterien erfüllen" in capsys.readouterr().out


def test_tune_presets_sorts_the_good_seeds_by_the_distance_to_the_measured_mean(monkeypatch, capsys):
    values = {200: 5.0, 201: -30.0, 202: -16.0}
    monkeypatch.setattr(TP, "_one", lambda seed: (seed, "+++++", values[seed]))
    TP.main(["200", "203"])
    assert "[202, 201, 200]" in capsys.readouterr().out                                     # |-16 + 15,7| < |-30 + 15,7| < |5 + 15,7|


# ---------------------------------------------------------------------------------------------------
# freeze_reference
# ---------------------------------------------------------------------------------------------------
def test_case_constants_of_the_freeze_tool_equal_the_shipped_reference():
    ref = FZ.reference()
    for (name, dname, Ls, p, z, kw), c in zip(FR.CASES, ref["cases"]):
        assert (name, dname, Ls, p, z, kw) == (c["name"], c["demand"], c["Ls"], c["p"], c["z"], c["kw"])
    assert {k: list(v) for k, v in FR.DEMANDS.items()} == ref["specs"] and FR.BURN == ref["burn"]
    assert [(n, kw) for n, kw in FR.CALIBRATIONS] == [(c["name"], c["kw"]) for c in ref["calibrations"]] and all(FR.CAL_KW == c["cal_kw"] for c in ref["calibrations"])


def test_case_records_reproduce_the_shipped_expected_values_from_the_frozen_demands():
    ref = FZ.reference()
    for c in ref["cases"]:
        D = FZ.demand(c["demand"])
        q, ph, vr = FR.evaluate(M, D, c["Ls"], c["p"], c["z"], c["kw"])
        rec = FR.case_record(c["name"], c["demand"], D, c["Ls"], c["p"], c["z"], c["kw"], q, ph, vr)
        assert rec["expected"]["q_zero_count"] == c["expected"]["q_zero_count"] and (rec["name"], rec["Ls"], rec["p"], rec["z"], rec["kw"]) == (c["name"], c["Ls"], c["p"], c["z"], c["kw"])
        for key in ("q_sum", "q_tail", "cost", "alpha", "beta", "onhand", "backlog", "cust_shipped", "goods_end", "amp"):
            np.testing.assert_allclose(rec["expected"][key], c["expected"][key], rtol=1e-9, atol=1e-9, err_msg=f"{c['name']} {key}")
    assert any(c["expected"]["q_zero_count"] > 0 for c in ref["cases"])


def test_tune_presets_jobs_option_alone_keeps_the_default_range(monkeypatch, fake_pool):
    seen = []
    monkeypatch.setattr(TP.mp, "Pool", fake_pool)
    monkeypatch.setattr(TP, "_one", lambda seed: (seen.append(seed), (seed, "+++++", -15.7))[1])
    assert TP.main(["--jobs", "4"]) == 0
    assert seen == list(range(200, 300)) and fake_pool.created == [4]


def test_quick_check_summary_takes_the_largest_single_deviation_of_z():
    data = R.load_results()
    full = data["jobs"]["main/basis"]["z"]
    results = [("Basis", 300, [z + 0.1 for z in full], [0.95] * 4, 280.0, 2.0), ("Basis", 301, [z - 0.3 for z in full], [0.95] * 4, 290.0, 4.0)]
    results += [(label, seed, data["jobs"][f"main/{job}"]["z"], [0.95] * 4, 100.0, 1.0) for label, (_, job) in QC.MEASURES.items() if label != "Basis" for seed in (300, 301)]
    s = QC.summarize(results, data)["Basis"]
    assert s["z_max_abs_diff"] == pytest.approx(0.3) and s["z_diff"] == pytest.approx([-0.1] * 4)
