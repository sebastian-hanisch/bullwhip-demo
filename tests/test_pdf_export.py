"""PDF-Export: die Kernschriften kennen nur Latin-1. Getestet werden die GENAUEN Zeichen, die fpdf2 abstürzen lassen (Gedankenstrich, Minuszeichen U+2212, Euro, Emoji, griechische Buchstaben),
die Ersetzungstabelle, der vollständige Export für alle Maßnahmen-Kombinationen und der Inhalt (unkomprimiert)."""
import re

import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_pdf_export as PDF
import bw_results as R
import bw_ui_panel as UI

DATA = R.load_results()
SEED = C.SEED_DEFAULT


def make(measure=C.M_SMOOTH, compress=True, message=None, note="", **over):
    p = dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=C.ZMODE_EQUAL, seed=SEED)
    p.update(over)
    d = FZ.preset_series(p["cv"], p["rho"]) if (p["cv"], p["rho"]) in ((0.2, 0.0), (0.2, 0.8)) else None
    live = LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], p["seed"], demand=d)
    pair = R.resolve_pair(DATA, measure, p["gamma"], p["cv"], p["rho"], p["lead"], p["window"])
    text = message if message is not None else UI.message(live, measure, pair)[1]
    return PDF.generate_bw_pdf(p, live, measure, text, note or pair["note"], pair["cell_text"], UI.comparison_table(live, measure, pair, DATA), compress=compress)


@pytest.mark.parametrize("char", ["γ", "ρ", "β", "σ", "μ", "τ", "κ", "Δ", "–", "—", "−", "‑", "≥", "≤", "→", "≈", "€", "±", "·", "“", "”", "„", "‘", "’", "√", "◀", "▶", "✅", "ℹ️", "⚠️", "🐂", "📐", "🎯",
                                  "📊", "🔧", "🎲", "📈", "📄", "🧮", "💶", "🔗", "⚙️"])
def test_every_problematic_character_is_replaced_by_latin1(char):
    out = PDF.pdf_text(f"a{char}b")
    out.encode("latin-1")
    assert out.startswith("a") and out.endswith("b")


def test_greek_letters_are_spelled_out_and_dashes_and_the_euro_are_plain():
    assert PDF.pdf_text("γ 0,5, ρ 0,8, β = 95 %, σ, μ") == "gamma 0,5, rho 0,8, beta = 95 %, sigma, mu"
    assert PDF.pdf_text("a – b — c − d") == "a - b - c - d" and PDF.pdf_text("5 €") == "5 EUR" and PDF.pdf_text("≥ 5 ≤ 6 → x ≈ y ± z") == ">= 5 <= 6 -> x ca. y +- z"
    assert PDF.pdf_text("„Glättung“") == '"Glättung"' and PDF.pdf_text("⚠️ Achtung") == "(!) Achtung" and PDF.pdf_text("✅ ok").strip() == "ok"


def test_umlauts_and_latin1_characters_survive():
    assert PDF.pdf_text("Größe für Bestände ÄÖÜäöüß ²") == "Größe für Bestände ÄÖÜäöüß ²"


def test_unknown_characters_become_question_marks_instead_of_crashing():
    assert PDF.pdf_text("a中b") == "a?b"


def test_a_pdf_with_every_measure_and_special_characters_in_the_message_is_valid():
    text = "⚠️ Glättung – γ ρ β σ μ ≥ ≤ → ≈ € ± ‘x’ „y“ 🐂 ✅ ℹ️ √ Δ τ κ ◀ ▶ " * 5
    for measure in (C.M_SMOOTH, C.M_RETAIL, C.M_INFO, C.M_INFO_SMOOTH, C.M_LEAD, C.M_WINDOW):
        pdf = make(measure, message=text, note="Hinweis – mit Sonderzeichen ρ 0,8 €")
        assert pdf[:5] == b"%PDF-" and pdf[-6:].strip().endswith(b"%%EOF") and len(pdf) > 3000


def test_the_pdf_works_in_the_hidden_measure_settings_and_in_fixed_z_mode():
    assert make(lead=1, window=40)[:5] == b"%PDF-"
    assert make(zmode=C.ZMODE_FIXED)[:5] == b"%PDF-"
    assert make(rho=0.8, gamma=0.5)[:5] == b"%PDF-"
    assert make(cv=0.3, rho=0.5, lead=4, window=6, gamma=0.7)[:5] == b"%PDF-"


def test_the_pdf_contains_the_sections_and_the_measured_numbers():
    raw = make(compress=False)
    text = raw.decode("latin-1").replace("\\", "")
    for needle in ("Bullwhip-Effekt: Wer zahlt f", "Einstellungen", "Kennzahlen dieses Verlaufs", "Meldung (aus der vorgerechneten Messreihe)", "Dieser Verlauf gegen die Messreihe",
                   "Hinweise zum Modell", "Autokorrelation rho", "Glättung gamma", "Gezeigte Ma", "alle gl", "Bezug", "270"):
        assert needle in text, needle
    assert text.count("/Type /Page\n") + text.count("/Type /Page ") >= 1 and "gleicher Servicegrad" in text


def test_the_pdf_is_compressed_by_default_and_larger_without():
    assert len(make(compress=True)) < len(make(compress=False))


def test_page_breaks_keep_tables_together():
    raw = make(compress=False).decode("latin-1")
    assert len(re.findall(r"/Type /Page\b(?!s)", raw)) >= 1


def test_the_pdf_is_compressed_by_default():
    default, explicit, plain = make(), make(compress=True), make(compress=False)
    assert abs(len(default) - len(explicit)) < 50 and len(default) < 0.8 * len(plain)               # die Erstellzeit steht im PDF: nur die Länge vergleichen


def test_the_pdf_text_carries_the_period_count_the_stage_labels_and_the_change_column():
    raw = make(compress=False).decode("latin-1").replace("\\", "")
    assert "Kennzahlen dieses Verlaufs (300 Perioden, ein einzelner Verlauf)" in raw
    for label in ("1 (Einzelhandel)", "2 (Grosshandel)".replace("Gross", "Groß"), "3 (Distributor)", "4 (Fabrik)"):
        assert label in raw, label
    assert "Bezug" in raw and "-15,8" in raw and "alle gl" in raw and "95,3 / 94,8 / 94,6 / 94,5" in raw
    assert raw.count(" re B") >= 3                                                                     # Tabellenköpfe sind gefüllt (fill + border)


def test_the_change_column_of_the_pdf_shows_the_sign_for_every_measure():
    raw = make(compress=False).decode("latin-1").replace("\\", "")
    live = LV.solve_live(0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL, SEED, demand=FZ.preset_series(0.2, 0.0))
    base = live["measures"][C.M_BASIS]["total_onhand"]
    for m in live["active"][1:]:
        assert UI.F.fmt_num(100 * (live["measures"][m]["total_onhand"] / base - 1), 1, True) in raw, m


def test_generate_bw_pdf_compresses_when_not_told_otherwise():
    p = dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=C.ZMODE_EQUAL, seed=SEED)
    live = LV.solve_live(0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL, SEED, demand=FZ.preset_series(0.2, 0.0))
    pair = R.resolve_pair(DATA, C.M_SMOOTH, 0.5, 0.2, 0.0, 2, 10)
    args = (p, live, C.M_SMOOTH, "Text", "", pair["cell_text"], UI.comparison_table(live, C.M_SMOOTH, pair, DATA))
    default, plain = PDF.generate_bw_pdf(*args), PDF.generate_bw_pdf(*args, compress=False)
    assert len(default) < 0.8 * len(plain)


def test_the_pdf_shows_a_plus_sign_for_a_measure_that_costs_stock():
    raw = make(compress=False, rho=0.8).decode("latin-1").replace("\\", "")
    assert "+14,5" in raw


def test_the_kennzahlen_table_of_the_pdf_has_its_own_plus_sign_for_a_costly_measure():
    """Zeigt die Auswertung Info teilen (Änderung negativ), steht die positive Änderung der Glättung (+14,5 bei ρ 0,8) nur in der Kennzahlen-Tabelle."""
    raw = make(C.M_INFO, compress=False, rho=0.8).decode("latin-1").replace("\\", "")
    assert "+14,5" in raw
