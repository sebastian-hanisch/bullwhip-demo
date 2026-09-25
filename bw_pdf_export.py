"""PDF-Export der Auswertung eines Verlaufs (fpdf2, Helvetica-Kernschrift, nur Text und Tabellen).

Die Kernschriften kennen nur Latin-1: Umlaute sind erlaubt, aber Gedankenstrich (U+2013), Minuszeichen (U+2212), Euro-Zeichen, Emoji und die griechischen Buchstaben gamma, rho, beta,
sigma, mu usw. lassen fpdf2 abstürzen. Deshalb läuft jeder Text durch pdf_text(): griechische Buchstaben werden ausgeschrieben."""
import time

import bw_constants as C
import bw_format as F
import bw_ui_panel as UI

_REPLACEMENTS = {
    "–": "-", "—": "-", "‑": "-", "−": "-", "≥": ">=", "≤": "<=", "→": "->", "≈": "ca.", "€": "EUR", "±": "+-", "·": "-", "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "⚠️": "(!)", "⚠": "(!)", "✅": "", "ℹ️": "", "γ": "gamma", "ρ": "rho", "β": "beta", "σ": "sigma", "μ": "mu", "τ": "tau", "κ": "kappa", "Δ": "Delta",
    "◀": "<", "▶": ">", "√": "Wurzel", "🐂": "", "📐": "", "🎯": "", "📊": "", "🔧": "", "🎲": "", "📈": "", "📄": "", "🧮": "", "💶": "", "🔗": "", "⚙️": "", "⚙": "",
}


def pdf_text(text):
    """Text für die Helvetica-Kernschrift: bekannte Sonderzeichen ersetzen (griechische Buchstaben ausschreiben), den Rest Latin-1-sicher machen."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_bw_pdf(settings, live, measure, message_text, note, cell_text, comparison, compress=True):
    """Auswertung eines Verlaufs als PDF: Einstellungen, Kennzahlen aller Maßnahmen, Meldung, Verlauf gegen Messreihe, Verstärkung je Stufe, Hinweise.

    settings: dict der Reglerwerte (cv, rho, lead, window, gamma, zmode, seed); live: bw_live.solve_live; measure: die gezeigte Maßnahme; message_text: Text der Meldung; note: Hinweis
    zur Zell-Zuordnung (leer, wenn exakt); cell_text: Beschreibung der Vergleichszelle der Messreihe; comparison: DataFrame aus bw_ui_panel.comparison_table."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_compression(compress)
    pdf.add_page()

    def line(text, height=7, width=0):
        pdf.cell(width, height, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(text):
        pdf.set_font("Helvetica", "B", 12)
        line(text, 8)
        pdf.set_font("Helvetica", "", 10)

    def pairs(rows):
        for label, value in rows:
            pdf.cell(80, 6, pdf_text(label), border=0)
            line(value, 6)

    def table(headers, widths, rows, size=8):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_fill_color(230, 230, 230)
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, pdf_text(header), border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", size)
        for row in rows:                                    # der automatische Seitenumbruch von fpdf2 hält die Zeile zusammen
            for value, width in zip(row, widths):
                pdf.cell(width, 7, pdf_text(str(value)), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(7)

    def keep_together(height):
        if pdf.get_y() + height > pdf.h - pdf.b_margin:
            pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    line("Bullwhip-Effekt: Wer zahlt für die Glättung?", 10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    line(f"Erstellt: {time.strftime('%d.%m.%Y %H:%M')}  -  sebastianhanisch.net", 6)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    heading("Einstellungen")
    pairs([("Nachfrageschwankung (CV)", F.fmt_level(settings["cv"])), ("Autokorrelation rho", F.fmt_level(settings["rho"])), ("Wiederbeschaffungszeit L (Perioden)", str(settings["lead"])),
           ("Prognosefenster p (Perioden)", str(settings["window"])), ("Glättung gamma", F.fmt_level(settings["gamma"])), ("Servicegrad", C.ZMODE_LABELS[settings["zmode"]]),
           ("Seed (gezeigter Verlauf)", str(settings["seed"])), ("Gezeigte Maßnahme", UI.measure_label(measure, settings["gamma"]))])
    pdf.ln(3)

    heading(f"Kennzahlen dieses Verlaufs ({live['T'] - live['burn']} Perioden, ein einzelner Verlauf)")
    base = live["measures"][C.M_BASIS]
    rows = []
    for m in live["active"]:
        v = live["measures"][m]
        rows.append([UI.measure_label(m, settings["gamma"]), F.fmt_num(v["amp"][-1], 1), F.fmt_num(v["total_onhand"], 0),
                     "Bezug" if m == C.M_BASIS else F.fmt_num(100 * (v["total_onhand"] / base["total_onhand"] - 1), 1, True), " / ".join(F.fmt_num(100 * b, 1) for b in v["beta"]),
                     F.fmt_num(v["total_cost"], 0)])
    table(["Maßnahme", "Verstärkung 4", "Summe Lager", "Änderung (%)", "Servicegrad je Stufe (%)", "Kosten"], [56, 24, 24, 24, 42, 20], rows)
    pdf.ln(3)

    heading("Meldung (aus der vorgerechneten Messreihe)")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_text(message_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4.5, pdf_text(f"Vergleichszelle der Messreihe: {cell_text}. {note}".strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    keep_together(50)
    heading("Dieser Verlauf gegen die Messreihe")
    cols = list(comparison.columns)
    table(cols, [62, 60, 68], [[r[c] for c in cols] for _, r in comparison.iterrows()])
    pdf.ln(3)

    keep_together(50)
    heading(f"Verstärkung je Stufe: Basis und {C.MEASURE_LABELS[measure]}")
    c = live["control"]
    m = live["measures"][measure]
    table(["Stufe", "Basis", C.MEASURE_LABELS[measure], "Kontrolle z = 0", "exakte Theorie", "Produktformel"], [40, 30, 34, 30, 30, 26],
          [[f"{k + 1} ({C.STAGE_NAMES[k]})", F.fmt_num(base["amp"][k], 2), F.fmt_num(m["amp"][k], 2), F.fmt_num(c["amp_run"][k], 2), F.fmt_num(c["amp_exact"][k], 2), F.fmt_num(c["amp_product"][k], 2)]
           for k in range(C.N_STAGES)])
    pdf.ln(3)

    keep_together(60)
    heading("Hinweise zum Modell")
    pdf.set_font("Helvetica", "", 9)
    for text in [
        "Stilisierte Kette: vier Stufen mit gleicher Wiederbeschaffungszeit und gleichem Fenster, ein Produkt, unbegrenzter Lieferant, keine Kapazitäten, keine Losgrößen, keine Informationsverzögerung, "
        "keine Rationierung. Parameter erfunden, nicht kalibriert: die Aussagen gelten in diesem Modell, bei diesen Reglern.",
        f"Ein einzelner Verlauf streut; die Aussage tragen die vorgerechneten 500 Wiederholungen je Zelle der Messreihe. Zell-Zuordnung: {cell_text}. Kosten der Bestellvarianz (Kapazität, Umrüstung, "
        "Expresszuschläge) sind nicht bewertet, der Nutzen der Glättung für die oberen Stufen ist deshalb eher unterschätzt.",
        "Bei gleichem Servicegrad wird z je Stufe so gewählt, dass der Mengen-Servicegrad 95 % erreicht; bei festem z = 1,645 wird der Servicegrad nicht festgehalten, dann täuscht ein Vergleich. Kleine "
        "Fenster (p bis 6) erzeugen durch die Schätzung der Streuung extreme Verstärkung (ein Schätzartefakt der Regel, kein Naturgesetz).",
    ]:
        pdf.multi_cell(0, 5, pdf_text("- " + text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())
