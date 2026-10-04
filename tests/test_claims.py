"""JEDE Zahl aus dem README wird hier nachgerechnet: die Befunde aus data/bw_results.json (über bw_results), die Modellangaben aus den Konstanten, die Live-Zahlen des Preset-Verlaufs aus der
eingefrorenen Nachfragereihe. Nicht geprüft werden Läufe, die nur lokal möglich sind (Bau-Gate tools/check_full.py, Preset-Suche tools/tune_presets.py, tools/quick_check.py): sie stehen im README
ausdrücklich als lokal gemessen. Das README wird mit Minuszeichen (U+2212) geschrieben; die Tests vergleichen mit dem Bindestrich."""
import os
import pathlib
import re

import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_results as R

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("−", "-")
DATA = R.load_results()
ROWS = {r["key"]: r for r in R.main_rows(DATA)}
FIX = {r["key"]: r for r in R.fixed_rows(DATA)}
F = R.F.fmt_num


def has(text):
    assert text.replace("−", "-") in README, text


def de(x, digits=1, sign=False):
    return F(x, digits, sign)


def test_readme_has_the_playbook_sections_in_order_and_the_undeployed_marker():
    heads = re.findall(r"^## (.+)$", README, flags=re.M)
    want = ["Warum dieses Problem", "Befunde und Korrekturen gegenüber dem Plan", "Modell", "Methodik", "Befunde (gemessen, keine Behauptungen)", "Ehrliche Grenzen", "Tests", "Dateistruktur", "Bewusst nicht umgesetzt",
            "Reproduktion der Messreihe", "Lokal ausführen", "Verwandte Demos mit demselben mathematischen Modell"]
    assert heads == want
    assert README.splitlines()[0].startswith("# Bullwhip-Effekt: Wer zahlt für die Glättung? – Streamlit-Demo") and "[→ Demo live ausprobieren](https://sebastianhanisch-bullwhip-demo.streamlit.app/)" in README.splitlines()[2]
    paras = README.rstrip().split("\n\n")  # vorletzter Absatz: Bauhinweis, letzter: Portfolio-Footer (set_demo_footer.py)
    assert paras[-2] == "Gebaut mit Streamlit, Plotly, NumPy und fpdf2." and paras[-1].startswith("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch]")
    assert "Stand 2026-09-25" in README


@pytest.mark.skipif(os.environ.get("BW_MUTATION_RUN") == "1", reason="die Dateiliste prüft die echte Projektstruktur")
def test_the_file_table_lists_every_project_file_and_only_real_ones():
    section = README.split("## Dateistruktur")[1].split("## Bewusst nicht umgesetzt")[0]
    listed = set()
    for cell in re.findall(r"^\| (.+?) \|", section, flags=re.M):
        listed |= set(re.findall(r"`([^`]+)`", cell))
    for name in listed:
        target = ROOT / name.rstrip("/")
        assert target.exists() or name.endswith("/"), name
    on_disk = {p.name for p in ROOT.glob("*.py")} | {p.name for p in (ROOT / "tools").glob("*.py")}
    on_disk -= {"__init__.py"}
    listed_names = {n.split("/")[-1] for n in listed}
    assert on_disk <= listed_names, sorted(on_disk - listed_names)


def test_model_facts():
    for n in ("bw_constants", "bw_model", "bw_theory", "bw_live", "bw_results", "bw_stories", "bw_presets", "bw_visualization", "bw_ui_panel", "bw_pdf_export", "bw_format"):
        assert (ROOT / f"{n}.py").exists()
    lo, hi = R.service_range(DATA)
    has(f"dort liegt β je Stufe zwischen {de(100 * lo)} und {de(100 * hi)} %")
    has(f"({round(R.DATA_PATH.stat().st_size / 1024)} KB, erzeugt von `tools/build_results.py`")
    has("3 · 3 · 3 · 4 = 108 Kombinationen")
    assert len(C.CV_OPTIONS) * len(C.RHO_OPTIONS) * len(C.LEAD_OPTIONS) * len(C.WINDOW_OPTIONS) == 108
    has("Kalibrierung auf 200 Wiederholungen × 1200 Perioden")
    has("(500 × 1500;")
    assert DATA["_meta"]["config"]["R_CAL"] == 200 and DATA["_meta"]["config"]["T_CAL"] == 1200 and DATA["_meta"]["config"]["R_EVAL"] == 500 and DATA["_meta"]["config"]["T_EVAL"] == 1500
    has("Zeitlimit 90 s und Pause 4 s")
    assert (C.QUICK_TIME_LIMIT, C.QUICK_COOLDOWN) == (90.0, 4.0)
    assert (C.QUICK_TIME_LIMIT, C.QUICK_COOLDOWN) == (90.0, 4.0)
    has("Bestand h = 1 und Rückstand 9 je Einheit")
    assert (C.H_COST, C.P_COST) == (1.0, 9.0)
    has("400 Perioden") and C.LIVE_T == 400 and C.LIVE_SHOWN == 300
    has("Verlauf über 300 Perioden")
    has("sieben Maßnahmen") and len(C.MEASURES) == 7
    has("Seed 202") and C.SEED_DEFAULT == 202


def test_ap0_coverage_claims():
    cov = R.coverage(DATA)
    has(f"**{cov['cell_exact']} von {cov['combos']}** Kombinationen")
    d = cov["by_distance"]
    has(f"({d[1]} weichen in einem Parameter ab, {d[2]} in zweien, {d[3]} in dreien)")
    has(f"**{cov['z_exact']} von {cov['z_cases']}**")
    assert (cov["cell_exact"], cov["combos"], cov["z_exact"], cov["z_cases"]) == (10, 108, 46, 1341)
    assert len(R.CELLS) == 10 and len(R.CELLS) - 1 == 9
    has("**neun** (nicht acht, wie der Plan")
    has("nur dieses hat alle 13 Maßnahmen")
    assert len([e for e in R.entries(DATA) if e["group"] == "main"]) == 13 and all(len([e for e in R.entries(DATA) if e["group"] == c]) == 3 for c in R.CELLS[1:])


def test_ap0_preset_seed_claim_on_the_frozen_series():
    p = C.PRESETS["Standard"]
    live = LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], p["seed"], demand=FZ.preset_series(p["cv"], p["rho"]))
    b, m = live["measures"][C.M_BASIS]["total_onhand"], live["measures"][C.M_SMOOTH]["total_onhand"]
    has(f"mit {de(100 * (m / b - 1), 1, True)} % genau auf dem Messreihen-Mittel ({de(R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)['delta'], 1, True)} %)")
    has("liegen bei 0 bis 299")
    has("bei 10000 und darüber")
    assert min(j["scenario"]["seed"] for j in DATA["jobs"].values()) >= 10000 and C.SEED_RANGE == (0, 299)
    has("durchsucht 200 bis 299")
    has("82 von 100 Seeds")                                                                        # lokal gemessen (tools/tune_presets.py), siehe README


def test_finding_1_formula_against_simulation():
    c = R.formula_case(DATA, 2, 5)
    has(f"Simulation {de(c['sim'][3])}, exakt {de(c['exact'][3])}, Produktformel {de(c['product'][3])}")
    rows = R.formula_rows(DATA)
    lo, hi = min(rows, key=lambda r: r["ratio4"]), max(rows, key=lambda r: r["ratio4"])
    has(f"zwischen **{de(lo['ratio4'], 2)}** (m {lo['m']}, p {lo['p']}) und **{de(hi['ratio4'], 2)}** (m {hi['m']}, p {hi['p']})")
    e2 = next(r for r in DATA["E2"] if r["cv"] == 0.2 and r["m"] == 2 and r["p"] == 10)
    has(f"Stufe 4 linear {de(e2['linear_exact'][3], 2)}, mit z = 1,645 und Abschneidung {de(e2['z1645_nonneg']['amp'][3], 2)}, ohne Abschneidung {de(e2['z1645_linear']['amp'][3], 2)} ({de(100 * e2['z1645_nonneg']['zero_share'][3])} % der Stufe-4-Bestellungen sind 0)")
    has("Abweichung < 1e-12")


def test_finding_2_lead_and_window_dominate():
    t = lambda c: R.total(DATA["jobs"][f"{c}/basis"])[0]
    has(f"L = 1 / 2 / 4: **{t('L1'):.0f} / {t('main'):.0f} / {t('L4'):.0f}**")
    has(f"p = 6 / 10 / 20 / 40: **{t('p6'):.0f} / {t('main'):.0f} / {t('p20'):.0f} / {t('p40'):.0f}**")
    has(f"**{de(t('L4') / t('L1'))}-fache (L)**")
    has(f"**{de(t('p6') / t('p40'))}-fache (p)**")


def test_finding_3_amplification_is_a_weak_cost_proxy():
    p = R.proxy_rows(DATA)
    has(f"über 13 Maßnahmen **{de(p['rho_all'], 2)}**")
    has(f"**{de(p['rho_no_levers'], 2)}**")
    a = ROWS["basis"]["amp"][3], ROWS["glaettung_0.3"]["amp"][3]
    has(f"um {100 * (1 - a[1] / a[0]):.0f} % ({de(a[0])} → {de(a[1])}) und spart nur **{de(-ROWS['glaettung_0.3']['delta'])} %**")
    has(f"Verstärkung -{100 * (1 - ROWS['nur_fabrik_glaettet_0.3']['amp'][3] / a[0]):.0f} %, Kettenbestand **+{de(ROWS['nur_fabrik_glaettet_0.3']['delta'])} %**")
    r = [DATA["jobs"][f"{c}/basis"]["equal"]["total_onhand"] for c in ("main", "rho0.5", "rho0.8")]
    has(f"({r[0]:.0f} / {r[1]:.0f} / {r[2]:.0f} für ρ 0 / 0,5 / 0,8)")


def test_finding_4_fixed_against_equal_service():
    b, g5, g3 = FIX["basis"], FIX["glaettung_0.5"], FIX["glaettung_0.3"]
    has(f"(Summe Lager {b['total_fixed']:.0f} → {g5['total_fixed']:.0f} bei γ 0,5)")
    has(f"(β je Stufe {' / '.join(de(100 * x) for x in g5['beta_fixed'])} % gegen {' / '.join(de(100 * x) for x in b['beta_fixed'])} % in der Basis)")
    has(f"Kettenkosten {b['cost_fixed']:.0f} → {g5['cost_fixed']:.0f} (γ 0,5) → {g3['cost_fixed']:.0f} (γ 0,3)")
    has(f"(-{de(-ROWS['glaettung_0.5']['delta'])} % bei γ 0,5, -{de(-ROWS['glaettung_0.3']['delta'])} % bei γ 0,3)")
    assert g5["cost_equal"] < b["cost_equal"] and g5["total_fixed"] < b["total_fixed"]


def test_finding_5_who_pays():
    ch = {g: R.payer_rows(DATA, g) for g in (0.7, 0.5, 0.3)}
    has(f"**{de(ch[0.7]['all_change'][0], 1, True)} / {de(ch[0.5]['all_change'][0], 1, True)} / {de(ch[0.3]['all_change'][0], 1, True)} %** bei alle glätten mit γ 0,7 / 0,5 / 0,3")
    has(f"**{de(ch[0.3]['retail_change'][0], 1, True)} %** bei „nur Einzelhandel glättet, γ 0,3“")
    has(f"Fabrik {de(ch[0.7]['all_change'][3], 1, True)} / {de(ch[0.5]['all_change'][3], 1, True)} / {de(ch[0.3]['all_change'][3], 1, True)} % bzw. {de(ch[0.3]['retail_change'][3], 1, True)} %")
    has(f"**-{de(-ROWS['nur_einzelhandel_glaettet_0.3']['delta'])} %** gegen -{de(-ROWS['glaettung_0.3']['delta'])} % (γ 0,3)")
    z_all, z_one = ROWS["glaettung_0.3"]["z"], ROWS["nur_einzelhandel_glaettet_0.3"]["z"]
    has(f"({' / '.join(de(z, 2) for z in z_all)} gegen {' / '.join(de(z, 2) for z in z_one)})")


def test_finding_6_measures_overview():
    for key, needle in (("L_halbiert", "L halbiert **-{}**"), ("info_teilen", "Info teilen **-{}**")):
        has(needle.format(f"{de(-ROWS[key]['delta'])} %"))
    has(f"Fenster verdoppelt -{de(-ROWS['Fenster_doppelt']['delta'])} %")
    has(f"Info teilen + Glättung (γ 0,5) -{de(-ROWS['info_teilen_glaettung_0.5']['delta'])} %")
    has(f"L halbiert + Info teilen -{de(-ROWS['L_halbiert_info_teilen']['delta'])} %")
    has(f"(Zusatznutzen des Teilens bei kurzer L nur {de(ROWS['L_halbiert']['delta'] - ROWS['L_halbiert_info_teilen']['delta'])} Prozentpunkte)")
    rb = R.robustness(DATA)
    has(f"alle 13 Summen innerhalb {de(rb['min'], 1, True)} bis {de(rb['max'], 1, True)} %")
    has(f"(Rangkorrelation {de(rb['spearman'], 3)})")


def test_finding_7_regimes():
    sm = {r["cell"]: r["delta"] for r in R.regime_rows(DATA, C.M_SMOOTH, 0.5)}
    inf = {r["cell"]: r["delta"] for r in R.regime_rows(DATA, C.M_INFO, 0.5)}
    s = lambda cells, d: " / ".join(de(d[c], 1, True) for c in cells)
    has(f"ρ 0 / 0,5 / 0,8: {s(('main', 'rho0.5'), sm)} / **{de(sm['rho0.8'], 1, True)} %**")
    has(f"L 1 / 2 / 4: {s(('L1', 'main', 'L4'), sm)} %")
    has(f"p 6 / 10 / 20 / 40: **{de(sm['p6'], 1, True)}** / {de(sm['main'], 1, True)} / {de(sm['p20'], 1, True)} / {de(sm['p40'], 1, True)} %")
    has(f"CV 0,1 / 0,2 / 0,3: {s(('cv0.1', 'main', 'cv0.3'), sm)} %")
    has(f"ρ: {s(('main', 'rho0.5', 'rho0.8'), inf)} %")
    has(f"L: {s(('L1', 'main', 'L4'), inf)} %")
    has(f"p: **{de(inf['p6'], 1, True)}** / {de(inf['main'], 1, True)} / {de(inf['p20'], 1, True)} / **{de(inf['p40'], 1, True)} %**")
    has(f"CV: {s(('cv0.1', 'main', 'cv0.3'), inf)} %")
    has(f"(β {' / '.join(de(100 * b) for b in FIX['info_teilen']['beta_fixed'])} %)")
    has(f"(z an der Fabrik {de(ROWS['info_teilen']['z'][3], 2)} statt {de(ROWS['basis']['z'][3], 2)})")


def test_finding_8_window_with_jumps():
    w = R.window_jump_rows(DATA)
    has("**" + " / ".join(f"{r['iid']:.0f}" for r in w) + "** (monoton)")
    has("**" + " / ".join(f"{r['shift']:.0f}" for r in w) + "**")
    jmin = min(w, key=lambda r: r["shift"])
    has(f"Minimum bei p = {jmin['p']}, bei p = 64 wieder **{de(100 * (w[-1]['shift'] / jmin['shift'] - 1), 1, True)} %**")
    has(f"(p 4: Stufe 4 bis {w[0]['amp4_iid']:.0f})")
    has("gerundeten Summen 176 und 168")
    assert round(w[4]["shift"]) == 176 and round(jmin["shift"]) == 168


def test_ap0_c_quick_calibration_numbers_are_the_measured_full_calibration_ones():
    has("gegen 1,07 / 1,33 / 1,66 / 1,84 der vollen Kalibrierung")
    z = DATA["jobs"]["main/basis"]["z"]
    assert [round(v, 2) for v in z] == [1.07, 1.33, 1.66, 1.84]
    has("Basis 284 gegen 270")
    has("Glättung 224 gegen 228")
    has("Info teilen 201 gegen 197")
    assert round(ROWS["basis"]["total"]) == 270 and round(ROWS["glaettung_0.5"]["total"]) == 228 and round(ROWS["info_teilen"]["total"]) == 197
    has("3,11 gegen 3,36")
    assert round(ROWS["glaettung_0.5"]["z"][3], 2) == 3.36


def test_the_presets_of_the_readme_match_the_criteria():
    pair = lambda m, *a: R.resolve_pair(DATA, m, 0.5, *a)
    has(f"gemessen {de(pair(C.M_INFO, 0.2, 0.0, 1, 10)['delta'], 1, True)} ± {de(pair(C.M_INFO, 0.2, 0.0, 1, 10)['se'])} %")


def test_the_only_sources_of_numbers_are_the_data_and_the_frozen_series():
    """Das README nennt die Md5-Summe der Messreihe; sie steht auch im Quelltext von tools/check_full.py-Ausgabe nicht fest, deshalb hier gegen die Datei der Messreihe (wenn vorhanden)."""
    has("767a84bc5737e87b733513a31cf88e80")
    src = ROOT.parent / "bestand-planung" / "messreihe_bullwhip" / "sweep_data.json"
    if not src.exists():
        pytest.skip("Messreihe nicht vorhanden (CI)")
    import hashlib
    assert hashlib.md5(src.read_bytes()).hexdigest() == "767a84bc5737e87b733513a31cf88e80"
    import json
    def leaves(o):
        return sum(leaves(v) for v in (o.values() if isinstance(o, dict) else o)) if isinstance(o, (dict, list)) else 1
    has(f"alle **{leaves(json.loads(src.read_text(encoding='utf-8')))} Zahlen**")
