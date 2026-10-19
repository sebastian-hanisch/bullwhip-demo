"""
Bullwhip-Effekt: Wer zahlt für die Glättung? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Stufen (Einzelhandel, Großhandel, Distributor, Fabrik) bestellen nach Order-up-to mit gleitendem Durchschnitt aus den Bestellungen der Stufe darunter: die Bestellungen schaukeln sich nach oben
auf. Live: EIN Verlauf über 300 Perioden, alle sieben Maßnahmen (Basis, alle glätten, nur Einzelhandel glättet, Info teilen, Info teilen + Glättung, L halbiert, Fenster verdoppelt) laufen über
denselben Endkundenbedarf. Vorgerechnet: die Messreihe über 500 Wiederholungen je Zelle bei GLEICHEM Servicegrad (data/bw_results.json), die die Kostenaussage trägt.

Lauffähig mit: streamlit run app.py
"""
import time

import pandas as pd
import streamlit as st

import bw_constants as C
import bw_format as F
import bw_live as LV
import bw_results as R
import bw_ui_panel as UI
import bw_visualization as V
from bw_pdf_export import generate_bw_pdf
from bw_presets import (SETTING_SPECS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                        sync_query_params)

st.set_page_config(page_title="Bullwhip – Sebastian Hanisch", layout="wide")

SCENARIO_KEYS = list(SETTING_SPECS)
DATA = R.load_results()
# Zahlen der Texte: alle aus der vorgerechneten Messreihe (kein Text-Zahlen-Widerspruch möglich)
_ROWS = {r["key"]: r for r in R.main_rows(DATA)}
_FIXED = {r["key"]: r for r in R.fixed_rows(DATA)}
_PAY = R.payer_rows(DATA, 0.3)
_PROXY = R.proxy_rows(DATA)
_ROBUST = R.robustness(DATA)
_BETA_MIN, _BETA_MAX = R.service_range(DATA)
_E1_MIN = min(R.formula_rows(DATA), key=lambda r: r["ratio4"])
_E1_MAX = max(R.formula_rows(DATA), key=lambda r: r["ratio4"])
_E1_CASE = R.formula_case(DATA, 2, 5)
_JUMP = R.window_jump_rows(DATA)
_COV = R.coverage(DATA)


def _amp_drop(key):
    return 100 * (1 - _ROWS[key]["amp"][3] / _ROWS["basis"]["amp"][3])


def _chg(key):
    return _ROWS[key]["delta"]


@st.cache_data(show_spinner=False, max_entries=C.CACHE_ENTRIES)
def _live(cv, rho, lead, window, gamma, zmode, seed, z_override):
    """Live-Verlauf mit allen Maßnahmen (Zehntelsekunden), je Einstellung zwischengespeichert. Der Aufruf geht über das Modul, damit die Tests die Rechnung zählen oder ersetzen können."""
    return LV.solve_live(cv, rho, lead, window, gamma, zmode, seed, z_override=dict(z_override) if z_override else None)


st.title("🐂 Bullwhip-Effekt: Wer zahlt für die Glättung?")
st.markdown(
    """
Der Kundenbedarf schwankt leicht, die **Bestellungen weiter oben in der Kette immer stärker**: Jede Stufe prognostiziert aus den Bestellungen der Stufe darunter und bestellt nach oben,
und die Schwankung schaukelt sich auf – der **Bullwhip-Effekt**. Bekannt ist die Formel; die Demo fragt nach dem, was sie nicht sagt: Was kosten Glättung, geteilte Bedarfsinformation und
kürzere Wiederbeschaffungszeit, wenn man den **Servicegrad festhält** – und **wer zahlt**? Live läuft **ein Verlauf über 300 Perioden** mit allen Maßnahmen, vorgerechnet die Messreihe über
**500 Wiederholungen je Zelle**, die die Aussage trägt. Die Prognose als gleitender Durchschnitt ist in `naive-forecast-demo` und `exponential-smoothing-demo` ein Verfahren, hier die
Rückkopplungsschleife; Bestand beim Kunden mit Touren zeigt `inventory-routing-demo`. Wer den Effekt zuerst ohne Fachbegriffe verstehen will, öffnet „💬 In einfachen Worten“. Wie das Modell funktioniert, steht im Expander „Wie funktioniert diese Demo?“ weiter unten, die formale
Beschreibung im Expander „📐 Mathematische Formulierung“.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/bullwhip-effekt-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich eine Kette vor: Laden, Großhandel, Distributor, Fabrik. Im Laden schwankt der Verkauf nur leicht. Jede Stufe schätzt aus den Bestellungen darunter, was sie braucht, und hält einen Sicherheitsbestand. Steigt der Bedarf kurz, bestellt sie mehr als nötig. Die nächste Stufe sieht nur diese Schwankung und verstärkt sie noch einmal. So schaukelt sich der Bedarf zur Fabrik hin auf wie bei einer Peitsche: der **Bullwhip-Effekt**. Die Demo prüft, was dagegen hilft und wer es bezahlt, wenn die Lieferfähigkeit gleich bleiben soll. Es ist ein stark vereinfachtes Modell mit erfundenen Parametern.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/bullwhip-effekt-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS)
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(3)
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Kette**")
    cv = st.select_slider("Nachfrageschwankung (CV)", options=list(C.CV_OPTIONS), key="cv_slider", format_func=F.fmt_level,
                          help="Variationskoeffizient des Endkundenbedarfs: Streuung in Teilen des Mittels (100). Die Stufen sind die gemessenen Zellen der Messreihe (0,1 / 0,2 / 0,3).")
    rho = st.select_slider("Autokorrelation ρ", options=list(C.RHO_OPTIONS), key="rho_slider", format_func=F.fmt_level,
                           help="Wie stark der Bedarf von einer Periode zur nächsten zusammenhängt (AR(1)). Bei ρ = 0,8 lässt sich die Glättung ins Negative kippen: ein träger Bedarf verträgt sie schlecht.")
    lead = st.select_slider("Wiederbeschaffungszeit L", options=list(C.LEAD_OPTIONS), key="lead_slider", format_func=lambda v: f"{v} Perioden" if v > 1 else "1 Periode",
                            help="Perioden zwischen Bestellung und Ankunft, in jeder Stufe gleich; zugleich das Schutzintervall der Order-up-to-Regel. Der stärkste Hebel der Kette.")
    window = st.select_slider("Prognosefenster p", options=list(C.WINDOW_OPTIONS), key="window_slider", format_func=lambda v: f"{v} Perioden",
                              help="Länge des gleitenden Durchschnitts (Mittel und Streuung der letzten p Werte). Kleine Fenster (p = 6) erzeugen durch die Schätzung der Streuung extreme Verstärkung: ein Schätzartefakt der Regel, kein Naturgesetz.")
    st.markdown("**Die Maßnahme**")
    gamma = st.select_slider("Glättung γ", options=list(C.GAMMA_OPTIONS), key="gamma_slider", format_func=F.fmt_level,
                             help="Anteil der Bestandslücke, der pro Periode korrigiert wird (1 = keine Glättung). Wirkt in „alle glätten“, „nur Einzelhandel glättet“ und „Info teilen + Glättung“; die übrigen Maßnahmen und die Basis laufen ohne Glättung.")
    zmode = st.radio("Servicegrad", list(C.ZMODE_OPTIONS), key="z_mode", format_func=lambda v: C.ZMODE_LABELS[v],
                     help="Gleicher Servicegrad: z je Stufe ist so gewählt, dass der Mengen-Servicegrad in jeder Stufe 95 % trifft (vorgerechnet aus der Messreihe). Festes z = 1,645 für alle: der Servicegrad wird nicht "
                          "festgehalten, dann täuscht ein Vergleich (Glättung sieht billiger aus, als sie ist).")
    st.markdown("**Der gezeigte Verlauf**")
    seed = st.number_input("Seed", *bounds("seed_input"), key="seed_input", step=1,
                           help="Nummer des gezeigten Verlaufs. Die Kennzahlen der Messreihe stehen auf 500 Wiederholungen mit anderen Nachfrage-Seeds (alle ab 10000); die Beispielszenarien zeigen einen Verlauf aus 200 bis 299.")
    st.button("🎲 Neuer Verlauf", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed.")

sync_query_params({key: st.session_state[key] for key in SCENARIO_KEYS})
lead, window, seed = int(lead), int(window), int(seed)
cv, rho, gamma = float(cv), float(rho), float(gamma)

# --- welche Maßnahmen gibt es bei dieser Einstellung ------------------------------------------------------------------------------------
active = R.active_measures(lead, window)
options = [m for m in active if m != C.M_BASIS]
if st.session_state.get("measure_select") not in options:                 # z. B. "L halbiert" bei L = 1: die Maßnahme entfällt (Widget in diesem Lauf gezeichnet: kein Desync)
    st.session_state["measure_select"] = C.MEASURE_DEFAULT

# --- Schnellkalibrierung: gilt für diese Kette (CV, ρ, L, p, γ), nicht für den Seed ---------------------------------------------------------
cal_key = (cv, rho, lead, window, gamma)
stored_cal = st.session_state.get("quick_cal")
cal_valid = stored_cal is not None and stored_cal["key"] == cal_key and zmode == C.ZMODE_EQUAL
z_override = tuple(sorted((m, tuple(z)) for m, z in stored_cal["z"].items())) if cal_valid else None

live = _live(cv, rho, lead, window, gamma, zmode, seed, z_override)

# ---------------------------------------------------------------------------------------------------
# Hauptansicht (Kernabschnitt ①, live: ein Verlauf)
# ---------------------------------------------------------------------------------------------------
st.markdown("## 🐂 Wie stark schaukelt sich die Kette hoch – und was kostet es?")
measure = st.selectbox("Maßnahme im Vergleich zur Basis (Anzeige)", options, key="measure_select", format_func=lambda m: UI.measure_label(m, gamma),
                       help="Reine Anzeigewahl im Ergebnisbereich (kein Regler der Rechnung): alle Maßnahmen sind für jede Einstellung schon gerechnet, hier wählen Sie nur, welche neben der Basis erscheint (Kennzahlen, Meldung, Zeitreihen).")
hidden = [m for m in C.MEASURES if m not in active]
if hidden:
    st.caption("Ausgeblendet bei dieser Einstellung: " + "; ".join({C.M_LEAD: "„L halbiert“ (bei L = 1 gibt es nichts zu halbieren)", C.M_WINDOW: "„Fenster verdoppelt“ (bei p = 40 ist die Messreihe zu Ende)"}[m] for m in hidden) + ".")

pair = R.resolve_pair(DATA, measure, gamma, cv, rho, lead, window)
cell = pair["cell"]

st.caption(f"Live-Verlauf (ein Verlauf, {live['T'] - live['burn']} Perioden nach {live['burn']} Perioden Einschwingen): Seed {live['seed']}, CV {F.fmt_level(cv)}, ρ {F.fmt_level(rho)}, L {lead}, p {window}, "
           f"γ {F.fmt_level(gamma)}, Servicegrad: {C.ZMODE_LABELS[zmode]}. Alle Maßnahmen laufen bei jeder Einstellung neu (zusammen etwa 0,3 s, je Einstellung zwischengespeichert). "
           "Ein einzelner Verlauf – die vorgerechnete Messreihe unten trägt die Aussage.")

metric_rows = [st.columns(2), st.columns(2)]
x = UI.render_metrics(metric_rows[0] + metric_rows[1], live, measure, pair)
msg_state = UI.render_message(live, measure, pair)
_, msg_text = UI.message(live, measure, pair)
_service_warning = UI.service_warning(live, measure)
if _service_warning:
    st.warning(_service_warning)
if zmode == C.ZMODE_FIXED:
    st.warning("Modus „festes z = 1,645“: der Servicegrad wird **nicht** festgehalten. Die Zahlen oben zeigen deshalb, was die Maßnahmen bei unverändertem Sicherheitsfaktor tun – nicht, was sie bei gleichem "
               "Service kosten. Die Meldung stützt sich weiter auf die Messreihe bei gleichem Servicegrad. " + UI.fixed_z_text(pair))

# z-Herkunft und Schnellkalibrierung
inexact = [m for m in live["active"] if not live["measures"][m]["z_exact"]]
if zmode == C.ZMODE_EQUAL:
    if cal_valid:
        st.info(f"Grobe Live-Kalibrierung aktiv: z je Stufe für {len(stored_cal['z'])} Maßnahmen neu bestimmt (20 Wiederholungen × 700 Perioden, ein Durchlauf, {F.fmt_num(stored_cal['seconds'], 1)} s). "
                "Sie ist **grob**: gegenüber der vollen Kalibrierung weicht z typisch um ±0,1 ab, der Servicegrad um etwa ±1 Prozentpunkt und die Summe Lager um einige Prozent (siehe README).")
    elif inexact:
        st.caption(f"z je Stufe: bei {len(live['active']) - len(inexact)} von {len(live['active'])} Maßnahmen genau vorgerechnet, bei {len(inexact)} aus der nächstliegenden gemessenen Konfiguration – welche, steht je Maßnahme "
                   "im Expander „🔧 Wie wir das erreichen“, Reiter „📈 Verlauf“ (Spalten „Herkunft z“ und „z gehört zur Konfiguration“).")
    else:
        st.caption("z je Stufe: für alle Maßnahmen genau vorgerechnet (Messreihe, Kalibrierung auf β = 95 % je Stufe).")
    b1, b2 = st.columns([3, 2])
    with b1:
        if st.button("Servicegrad live nachkalibrieren (grob, ca. 2 s je Maßnahme)", key="quickcal_button",
                     help="Bestimmt z je Stufe für alle Maßnahmen neu (20 Wiederholungen × 700 Perioden, ein Durchlauf, grob) – nützlich für Kombinationen, die die Messreihe nicht genau gemessen hat. Dauert etwa 2 s je Maßnahme."):
            left = LV.cooldown_left(st.session_state.get("quickcal_last"), time.monotonic())
            if left > 0:
                st.warning(f"Bitte noch {F.fmt_num(left, 0)} s warten: zwischen zwei Aufrufen liegt eine Pause von {F.fmt_num(C.QUICK_COOLDOWN, 0)} s.")
            else:
                st.session_state["quickcal_last"] = time.monotonic()
                bar = st.progress(0.0, text="Kalibriere z je Stufe (grob) …")
                res = LV.quick_calibrate(cv, rho, lead, window, gamma, seed, live["active"], on_progress=lambda i, n, m: bar.progress(i / n, text=f"Kalibriere z je Stufe (grob): {C.MEASURE_LABELS[m]} …"))
                bar.empty()
                st.session_state["quick_cal"] = {"key": cal_key, "z": res["z"], "seconds": res["seconds"], "skipped": res["skipped"]}
                st.rerun()
    with b2:
        if cal_valid and st.button("Vorgerechnetes z wieder verwenden", key="quickcal_reset", help="Verwirft die grobe Live-Kalibrierung und nimmt wieder das vorgerechnete z der Messreihe."):
            st.session_state.pop("quick_cal", None)
            st.rerun()
    if stored_cal is not None and stored_cal["key"] != cal_key:
        st.caption("Die Einstellungen der Kette haben sich seit der letzten Live-Kalibrierung geändert: sie gilt nicht mehr, es wird wieder das vorgerechnete z genommen.")
    if cal_valid and stored_cal["skipped"]:
        st.warning(f"Zeitlimit von {F.fmt_num(C.QUICK_TIME_LIMIT, 0)} s erreicht: {len(stored_cal['skipped'])} Maßnahmen behalten das vorgerechnete z.")

st.markdown("**Dieser Verlauf und die Messreihe nebeneinander**")
st.dataframe(UI.comparison_table(live, measure, pair, DATA), width="stretch", hide_index=True)
st.caption(f"Messreihe: {pair['cell_text']}, 500 Wiederholungen je Zelle bei gleichem Servicegrad, Mittel ± Standardfehler. Ein einzelner Verlauf streut um diese Mittelwerte (Summe Lager typisch ±5 % um die Messreihe); "
           "die Meldung stützt sich deshalb auf die Messreihe." + (" " + pair["note"] if pair["note"] else ""))

st.markdown("#### 📈 Die Peitsche: Bestellungen der vier Stufen")
st.plotly_chart(V.whip_figure(live, measure), width="stretch", key="main_whip")
st.caption("Schwarz der Endkundenbedarf, farbig die Bestellungen der vier Stufen (Stufe 1 Einzelhandel bis Stufe 4 Fabrik), links die Basis, rechts die gewählte Maßnahme auf derselben Achse. "
           "In der Basis schwanken die Bestellungen der Fabrik um ein Vielfaches stärker als der Kundenbedarf; Glättung und geteilte Information beruhigen sie.")

d1, d2 = st.columns(2)
with d1:
    stage = st.radio("Stufe im Detail", [1, 2, 3, 4], horizontal=True, key="stage_select", format_func=lambda k: f"Stufe {k} ({C.STAGE_NAMES[k - 1]})",
                     help="Reine Anzeigewahl im Ergebnisbereich (kein Regler der Rechnung): welche Stufe die Detailgrafik zeigt.")
with d2:
    view = st.radio("Anzeige", ["Bestellungen", "Lager"], horizontal=True, key="view_select",
                    help="Reine Anzeigewahl im Ergebnisbereich (kein Regler der Rechnung): Bestellungen oder Lager der gewählten Stufe. Alle Verläufe sind schon gerechnet; die Kennzahlen oben ändern sich nicht.")
st.plotly_chart(V.detail_figure(live, measure, stage, view), width="stretch", key="main_detail")
st.caption("Basis (grau) gegen die gewählte Maßnahme (blau) in der gewählten Stufe. Bei „Lager“ zeigt sich, wer für die Glättung zahlt: die glättende Stufe braucht mehr Bestand, die Stufen darüber weniger.")

st.markdown("**Verstärkung und Lager je Stufe (dieser Verlauf)**")
st.plotly_chart(V.stage_bars_figure(live, measure), width="stretch", key="main_stage_bars")
st.dataframe(UI.amp_frame(live, measure), width="stretch", hide_index=True)
st.caption(UI.control_text(live))

pdf_slot = st.container()

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ② (vorgerechnet): was die Messreihe bei gleichem Servicegrad zeigt
# ---------------------------------------------------------------------------------------------------
st.markdown("### 📐 Was die Messreihe bei gleichem Servicegrad zeigt")
st.markdown(
    f"""
Kernfrage: Was kosten die Maßnahmen an Bestand, wenn der **Servicegrad in jeder Stufe gleich bleibt** (β = 95 %, z je Stufe nachkalibriert, auf einer unabhängigen Stichprobe bewertet: β liegt je Stufe zwischen
{F.fmt_num(100 * _BETA_MIN, 1)} und {F.fmt_num(100 * _BETA_MAX, 1)} %)? Die Antwort steht auf **500 Wiederholungen je Zelle** über 1500 Perioden, vorgerechnet und **nie live** gerechnet: der Sweep braucht etwa zwei
Minuten auf 12 Prozessen. Alle Aussagen gelten **in diesem Modell, bei diesen Reglern**; Mittel ± Standardfehler, ein Vorzeichen gilt nur ab 2 Standardfehlern. Ihre Einstellung liegt auf der Zelle:
**{pair['cell_text']}**.
"""
)
if not pair["exact"]:
    st.info(pair["note"] or "Ihre Einstellung ist nicht ganz gemessen.")

st.markdown("**1 · Maßnahmen bei gleichem Servicegrad** – Summe Lager der vier Stufen (Hauptszenario CV 0,2, ρ 0, L 2, p 10)")
main_rows = R.main_rows(DATA)
st.plotly_chart(V.measures_figure(main_rows), width="stretch", key="core_measures")
st.dataframe(UI.main_frame(main_rows), width="stretch", hide_index=True)
st.caption(f"Bei gleichem Servicegrad spart die halbierte Wiederbeschaffungszeit **{F.fmt_num(-_chg('L_halbiert'), 1)} %** Kettenbestand, geteilte Bedarfsinformation **{F.fmt_num(-_chg('info_teilen'), 1)} %** und Glättung an allen Stufen "
           f"(γ 0,3) nur **{F.fmt_num(-_chg('glaettung_0.3'), 1)} %** – obwohl sie die Verstärkung an der Fabrik von {F.fmt_num(_ROWS['basis']['amp'][3])} auf {F.fmt_num(_ROWS['glaettung_0.3']['amp'][3])} drückt "
           f"(−{F.fmt_num(_amp_drop('glaettung_0.3'), 0)} %). „Nur Einzelhandel glättet, γ 0,3“ spart {F.fmt_num(-_chg('nur_einzelhandel_glaettet_0.3'), 1)} %: Glätten nur an der Quelle schlägt Glätten überall; jede glättende Stufe braucht "
           f"ein größeres z und zahlt diesen Preis, an vier Stellen also viermal. Standardfehler der Summen {F.fmt_num(min(r['se'] for r in main_rows), 1)} bis {F.fmt_num(max(r['se'] for r in main_rows), 1)}. Wiederholung mit anderen Seeds: alle "
           f"{_ROBUST['n']} Summen innerhalb {F.fmt_num(_ROBUST['min'], 1, True)} bis {F.fmt_num(_ROBUST['max'], 1, True)} % der ersten Läufe, Rangfolge identisch (Rangkorrelation {F.fmt_num(_ROBUST['spearman'], 3)}). "
           "Die Kosten der Maßnahmen selbst (Lieferzeit verkürzen, Daten austauschen) sind hier nicht enthalten, ebenso wenig die Kosten der Bestellvarianz.")

st.markdown("**2 · Fester z gegen gleicher Servicegrad** – warum ein Vergleich ohne festgehaltenen Servicegrad täuscht")
fixed_rows = R.fixed_rows(DATA)
f1, f2 = st.columns([3, 2])
with f1:
    st.plotly_chart(V.fixed_service_figure(fixed_rows), width="stretch", key="core_fixed")
with f2:
    st.markdown(f"Mit **festem z = 1,645** ohne Nachkalibrieren sieht Glättung billiger aus (Summe Lager {F.fmt_num(_FIXED['basis']['total_fixed'], 0)} → {F.fmt_num(_FIXED['glaettung_0.5']['total_fixed'], 0)} bei γ 0,5), "
                f"verschlechtert aber den Service drastisch (β je Stufe {' / '.join(F.fmt_num(100 * b, 1) for b in _FIXED['glaettung_0.5']['beta_fixed'])} % bei γ 0,5 gegen "
                f"{' / '.join(F.fmt_num(100 * b, 1) for b in _FIXED['basis']['beta_fixed'])} % in der Basis). Die Kettenkosten (Bestand plus 9 je Einheit Rückstand) steigen dann von {F.fmt_num(_FIXED['basis']['cost_fixed'], 0)} "
                f"auf {F.fmt_num(_FIXED['glaettung_0.5']['cost_fixed'], 0)} (γ 0,5) bzw. {F.fmt_num(_FIXED['glaettung_0.3']['cost_fixed'], 0)} (γ 0,3). **Bei gleichem Servicegrad** ist die Kette mit Glättung dagegen billiger.")
st.dataframe(UI.fixed_frame(fixed_rows), width="stretch", hide_index=True)
st.caption("Die naheliegende Erwartung „Glättung erhöht die Kosten“ stimmt nur, wenn man den Servicegrad nicht festhält. Umgekehrt verschlechtert geteilte Bedarfsinformation bei festem z den Service in den oberen Stufen: "
           "die Sicherheitsbestandsrechnung muss dem neuen Informationsstand folgen (z an der Fabrik 2,38 statt 1,84).")

st.markdown("**3 · Wer zahlt?** – Lager je Stufe bei gleichem Servicegrad")
payer = R.payer_rows(DATA, gamma)
st.plotly_chart(V.payer_figure(payer), width="stretch", key="core_payer")
st.dataframe(UI.payer_frame(payer), width="stretch", hide_index=True)
st.caption(f"Stufe 1 ist der Einzelhandel, Stufe 4 die Fabrik. Die glättende Stufe **zahlt selbst**: Lager des Einzelhändlers {F.fmt_num(payer['all_change'][0], 1, True)} % bei „alle glätten“ und {F.fmt_num(payer['retail_change'][0], 1, True)} % bei „nur Einzelhandel glättet“ "
           f"(γ {F.fmt_level(payer['gamma_all'])}), der Nutzen fällt bei den Stufen darüber an ({F.fmt_num(payer['retail_change'][3], 1, True)} % an der Fabrik). Nutzen und Kosten liegen in verschiedenen Unternehmen: "
           "ein **Anreizproblem**, das die reine Verstärkungskennzahl nicht zeigt. Die Kosten der Bestellvarianz (Kapazität, Umrüstung, Expresszuschläge) sind nicht bewertet, der Nutzen der Glättung oben ist deshalb eher unterschätzt.")

st.markdown("**4 · Regime-Landkarte** – wann hilft Glättung, wann Info teilen? (alle 10 gemessenen Zellen)")
smooth_rows, info_rows = R.regime_rows(DATA, C.M_SMOOTH, gamma), R.regime_rows(DATA, C.M_INFO, gamma)
st.plotly_chart(V.regime_figure(smooth_rows, info_rows, cell), width="stretch", key="core_regime")
st.dataframe(UI.regime_frame(smooth_rows, info_rows, cell), width="stretch", hide_index=True)
_sm = {r["cell"]: r for r in smooth_rows}
_in = {r["cell"]: r for r in info_rows}
st.caption(f"Urteil in drei Zuständen, wie die Meldung oben: „senkt hier den Kettenbestand“ (Änderung höchstens −5 % und unter −2 Standardfehlern), „kostet hier“ (über +2 % und über 2 Standardfehlern) und „bringt hier wenig“ (dazwischen). "
           f"**Glättung ist regimeabhängig:** ρ 0 / 0,5 / 0,8: {F.fmt_num(_sm['main']['delta'], 1, True)} / {F.fmt_num(_sm['rho0.5']['delta'], 1, True)} / {F.fmt_num(_sm['rho0.8']['delta'], 1, True)} %; "
           f"L 1 / 2 / 4: {F.fmt_num(_sm['L1']['delta'], 1, True)} / {F.fmt_num(_sm['main']['delta'], 1, True)} / {F.fmt_num(_sm['L4']['delta'], 1, True)} %; p 6 / 10 / 20 / 40: {F.fmt_num(_sm['p6']['delta'], 1, True)} / "
           f"{F.fmt_num(_sm['main']['delta'], 1, True)} / {F.fmt_num(_sm['p20']['delta'], 1, True)} / {F.fmt_num(_sm['p40']['delta'], 1, True)} % – bei stark autokorreliertem Bedarf kostet sie. **Info teilen hilft überall, unterschiedlich stark:** "
           f"p 6 / 10 / 20 / 40: {F.fmt_num(_in['p6']['delta'], 1, True)} / {F.fmt_num(_in['main']['delta'], 1, True)} / {F.fmt_num(_in['p20']['delta'], 1, True)} / {F.fmt_num(_in['p40']['delta'], 1, True)} %; "
           f"L 1 / 2 / 4: {F.fmt_num(_in['L1']['delta'], 1, True)} / {F.fmt_num(_in['main']['delta'], 1, True)} / {F.fmt_num(_in['L4']['delta'], 1, True)} %. Jede Zelle wurde einmal gerechnet (das Hauptszenario zusätzlich mit anderen Seeds): "
           "kleine Unterschiede nicht deuten. Die Zelle Ihrer Einstellung ist hervorgehoben.")

st.markdown("**5 · Verstärkung als Kostenproxy** – sagt die Verstärkung den Bestand voraus?")
st.plotly_chart(V.proxy_figure(_PROXY["points"]), width="stretch", key="core_proxy")
st.dataframe(UI.proxy_frame(_PROXY), width="stretch", hide_index=True, height=UI.CHART_TABLE_HEIGHT)
st.caption(f"Rangkorrelation der Verstärkung an Stufe 4 mit der Summe Lager über die {_PROXY['n_all']} Maßnahmen: **{F.fmt_num(_PROXY['rho_all'], 2)}**; ohne die {_PROXY['n_all'] - _PROXY['n_no_levers']} Maßnahmen, die L oder p ändern, "
           f"**{F.fmt_num(_PROXY['rho_no_levers'], 2)}** – das Ranking nach Verstärkung sagt das Ranking nach Bestand kaum voraus. Beispiel: „nur Fabrik glättet“ drückt die Verstärkung an Stufe 4 um {F.fmt_num(_amp_drop('nur_fabrik_glaettet_0.3'), 0)} %, "
           f"der Kettenbestand steigt aber um {F.fmt_num(_chg('nur_fabrik_glaettet_0.3'), 1)} %: die geglätteten Bestellungen gehen an den externen Lieferanten, niemand in der Kette profitiert.")

st.markdown("**6 · Formel gegen Simulation** – die Produktformel ist nur eine Untergrenze")
e1_rows = R.formula_rows(DATA)
e1_labels = {f"m = {r['m']}, p = {r['p']}": r for r in e1_rows}
e1_pick = st.selectbox("Fall (Wiederbeschaffungszeit m = L, Fenster p)", list(e1_labels), index=list(e1_labels).index("m = 2, p = 5"), key="formula_select",
                       help="Reine Anzeigewahl: welchen der zwölf gerechneten Fälle die Grafik zeigt (i.i.d.-Bedarf, ohne Sicherheitsbestand und ohne Abschneidung).")
st.plotly_chart(V.formula_figure(e1_labels[e1_pick]), width="stretch", key="core_formula")
st.dataframe(UI.formula_frame(e1_rows), width="stretch", hide_index=True, height=UI.CHART_TABLE_HEIGHT)
st.caption(f"Ohne Sicherheitsbestand und Abschneidung ist die Verstärkung einer Stufe exakt 1 + 2m/p + 2m²/p² und die der Kette eine Faltung; die bekannte Produktformel bleibt ab Stufe 2 darunter. Beispiel m = 2, p = 5, Stufe 4: "
           f"Simulation {F.fmt_num(_E1_CASE['sim'][3])}, exakt {F.fmt_num(_E1_CASE['exact'][3])}, Produktformel {F.fmt_num(_E1_CASE['product'][3])}. Das Verhältnis exakt/Produkt an Stufe 4 liegt zwischen {F.fmt_num(_E1_MIN['ratio4'], 2)} (m {_E1_MIN['m']}, p {_E1_MIN['p']}) "
           f"und {F.fmt_num(_E1_MAX['ratio4'], 2)} (m {_E1_MAX['m']}, p {_E1_MAX['p']}); die Verstärkung hängt nur von m/p ab. Die Zuordnung zur Originalarbeit (Chen et al.) ist durch Herleitung und Simulation belegt, das Paper selbst wurde nicht nachgeschlagen.")

st.markdown("**7 · Fenster mit Niveausprüngen** – länger hilft, aber nicht unbegrenzt")
st.plotly_chart(V.window_jump_figure(_JUMP), width="stretch", key="core_jump")
st.dataframe(UI.window_jump_frame(_JUMP), width="stretch", hide_index=True)
_jmin = min(_JUMP, key=lambda r: r["shift"])
st.caption(f"Bei i.i.d.-Bedarf sinkt der Kettenbestand (bei gleichem Servicegrad) mit dem Fenster monoton; mit zufälligen Niveausprüngen (±20 % im Mittel alle 50 Perioden) liegt das Minimum bei p = {_jmin['p']} "
           f"und bei p = 64 steigt er wieder ({F.fmt_num(100 * (_JUMP[-1]['shift'] / _jmin['shift'] - 1), 1, True)} % gegen p {_jmin['p']}). Der Zielkonflikt zwischen Rauschunterdrückung und Nachlaufen ist real, im gewählten Sprungmodell aber mild. "
           "Kleine Fenster (p 4 bis 6) erzeugen extreme Verstärkung: ein Schätzartefakt der Regel, kein Naturgesetz.")

# PDF (nach der Live-Rechnung)
with pdf_slot:
    st.download_button(
        "📄 Auswertung als PDF herunterladen",
        data=generate_bw_pdf(dict(cv=cv, rho=rho, lead=lead, window=window, gamma=gamma, zmode=zmode, seed=seed), live, measure, msg_text, pair["note"], pair["cell_text"],
                             UI.comparison_table(live, measure, pair, DATA)),
        file_name="bullwhip_auswertung.pdf", mime="application/pdf", key="primary_pdf_download",
        help="Einstellungen, Kennzahlen aller Maßnahmen auf diesem Verlauf, Meldung, Verlauf gegen Messreihe und Verstärkung je Stufe.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Ansichten
# ---------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – Maßnahmen im Vergleich"):
    tabs = st.tabs(["📈 Verlauf", "🧮 Theorie", "💶 Kosten und Servicegrad", "📊 Messreihe"])
    with tabs[0]:
        st.markdown(
            "Der **Verlauf** ist eine Simulation von 400 Perioden (die ersten 100 sind Einschwingen): Jede Periode kommt die Lieferung an, der Endkundenbedarf trifft die Stufe 1, jede Stufe liefert aus dem Lager "
            "(offener Rückstand zuerst, Rest wird Rückstand) und bestellt am Periodenende nach **Order-up-to** mit gleitendem Durchschnitt. Die Maßnahmen: **alle glätten** und **nur Einzelhandel glättet** "
            "(nur der Anteil γ der Bestandslücke wird korrigiert), **Info teilen** (alle Stufen prognostizieren aus dem Endkundenbedarf), **Info teilen + Glättung**, **L halbiert**, **Fenster verdoppelt**. "
            "Alle laufen über denselben Bedarf; z je Stufe kommt im Modus „gleicher Servicegrad“ aus der Messreihe."
        )
        st.dataframe(UI.measures_frame(live), width="stretch", hide_index=True)
        st.caption("Alle Maßnahmen dieses einen Verlaufs. Die Spalte „Herkunft z“ sagt, ob z genau vorgerechnet, aus der nächstliegenden gemessenen Konfiguration oder grob live nachkalibriert ist, die Spalte „z gehört zur Konfiguration“ "
                   "welche Konfiguration der Messreihe (bzw. welche Stichprobe) dahinter steht. Ein Verlauf streut: die Änderungen der Summe Lager liegen typisch bei ±5 Prozentpunkten um die Messreihe.")
        st.markdown("**Wer zahlt auf diesem Verlauf** (mittleres Lager je Stufe):")
        st.dataframe(UI.payer_live_frame(live, measure), width="stretch", hide_index=True)
    with tabs[1]:
        st.markdown(
            "Ohne Sicherheitsbestand (z = 0) und ohne Abschneidung bei 0 ist die Bestellkette **linear**: die Bestellung einer Stufe ist eine feste Linearkombination des Endkundenbedarfs, "
            "die Verstärkung ein Varianzquotient des Kettenfilters (Faltung). Die Kontrollzeile rechnet genau diese Kette auf **diesem** Verlauf nach."
        )
        st.plotly_chart(V.theory_figure(live["control"]), width="stretch", key="tab_theory")
        st.caption(UI.control_text(live))
        rho_pick = st.selectbox("Autokorrelation für die m/p-Karte", [0.0, 0.3, 0.6, 0.9], key="map_rho", format_func=F.fmt_level,
                                help="Reine Anzeigewahl: Autokorrelation der exakten Theorie in der Karte (die Karte gehört nicht zu den Reglern der Rechnung).")
        st.plotly_chart(V.linear_map_figure(DATA, rho_pick), width="stretch", key="tab_linear_map")
        st.caption("Exakte lineare Theorie, Stufe 4, lokal bestellt (jede Stufe aus der Bestellung darunter): die Verstärkung wächst mit m/p exponentiell durch die Kette. Bei ρ = 0,9 sinkt sie, weil ein träger Bedarf "
                   "sich leichter fortschreiben lässt – aber die Kosten bei gleichem Servicegrad steigen (Abschnitt 4).")
        e2 = [r for r in DATA["E2"] if r["cv"] == 0.2 and r["m"] in (1, 2, 4) and r["p"] in (5, 10, 20)]
        st.markdown("**Praxis: mit Sicherheitsbestand und Abschneidung** (CV 0,2, Stufe 4):")
        st.dataframe(pd.DataFrame([{"m (= L)": r["m"], "p": r["p"], "lineare Theorie": F.fmt_num(r["linear_exact"][3], 2), "z = 0 mit Abschneidung": F.fmt_num(r["z0_nonneg"]["amp"][3], 2),
                                    "z = 1,645 mit Abschneidung": F.fmt_num(r["z1645_nonneg"]["amp"][3], 2), "z = 1,645 ohne Abschneidung": F.fmt_num(r["z1645_linear"]["amp"][3], 2),
                                    "Bestellung = 0 (z = 1,645)": F.fmt_share(r["z1645_nonneg"]["zero_share"][3])} for r in e2]), width="stretch", hide_index=True)
        st.caption("Der Sicherheitsbestand-Term z·σ̂ verstärkt das Rauschen zusätzlich; die Abschneidung (keine negativen Bestellungen) ist ein realer Dämpfer, den die lineare Theorie nicht kennt.")
    with tabs[2]:
        st.markdown("Kosten und Servicegrad je Maßnahme auf **diesem** Verlauf (Bestand h = 1, Rückstand 9 je Einheit und Periode) und die Messreihe für das Hauptszenario.")
        st.dataframe(UI.measures_frame(live), width="stretch", hide_index=True)
        st.markdown("**Fester z gegen gleicher Servicegrad (Messreihe, Hauptszenario):**")
        st.dataframe(UI.fixed_frame(fixed_rows), width="stretch", hide_index=True)
        st.caption("Bei festem z sind die Zahlen ohne festgehaltenen Servicegrad nicht vergleichbar; bei gleichem Servicegrad zählt der Bestand.")
    with tabs[3]:
        st.markdown("Weitere vorgerechnete Auswertungen der Messreihe (63 Kalibrierungsjobs, 500 Wiederholungen je Job).")
        st.markdown("**Abdeckung der Regler durch gemessene Zellen** (108 Kombinationen aus CV, ρ, L, p):")
        st.dataframe(UI.cell_coverage_frame(_COV), width="stretch", hide_index=True)
        st.caption(f"Nur {_COV['cell_exact']} von {_COV['combos']} Kombinationen liegen genau auf einer gemessenen Zelle (das Hauptszenario und neun Zellen, die je einen Parameter ändern); für alle anderen zeigt die App die nächstliegende Zelle und sagt es. "
                   f"Von {_COV['z_cases']} Fällen (Kombination × Glättung × Maßnahme) hat die Messreihe für {_COV['z_exact']} ein genau passendes z; für den Rest kommt z aus der nächstliegenden gemessenen Konfiguration oder aus der groben Live-Kalibrierung.")
        st.markdown("**Verstärkung und Lager je Zelle** (Regime, Glättung und Info teilen):")
        st.dataframe(UI.regime_frame(smooth_rows, info_rows, cell), width="stretch", hide_index=True)

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        rf"""
**Kette.** Vier Stufen: Einzelhandel → Großhandel → Distributor → Fabrik. Stufe 1 sieht den Endkundenbedarf $D_t$ = AR(1) mit Mittel 100, Randstreuung 100 · CV und Autokorrelation ρ; Stufe k+1 sieht die Bestellung der Stufe k
in derselben Periode. Die Fabrik bestellt bei einem unbegrenzten externen Lieferanten. Die Lieferzeit L (Bestellung am Periodenende, Ankunft zu Beginn von t + L) ist in jeder Stufe gleich; Backorders werden voll nachgeliefert.

**Politik je Stufe.** Order-up-to: $S = m\,\hat\mu + z\,\hat\sigma\sqrt{{m}}$ mit Mittel und Standardabweichung der letzten p Werte der Prognose-Eingangsreihe (einschließlich des aktuellen), Schutzintervall m = L; Bestellung $q = \max(0;\, S - IP)$ mit der Bestandsposition $IP$,
**nie negativ** (die Abschneidung bei 0 ist ein realer Dämpfer, den die lineare Theorie nicht kennt). **Maßnahmen:** *Glättung* (γ < 1) korrigiert nur den Anteil γ der Bestandslücke, $q = \max\bigl(0;\, \hat\mu + \gamma\,(S - IP - \hat\mu)\bigr)$; *Info teilen*: alle Stufen
prognostizieren aus dem Endkundenbedarf; *L halbiert*; *Fenster verdoppelt*; jede Maßnahme an allen Stufen oder nur im Einzelhandel. Es werden **immer alle Maßnahmen** gerechnet, damit jeder Regler in jedem Zustand wirkt; „L halbiert“ bei L = 1
und „Fenster verdoppelt“ bei p = 40 sind ohne Wirkung und werden dann ausgeblendet.

**Warum der Servicegrad festgehalten wird.** Kosten: Bestand h = 1 und Rückstand 9 je Einheit und Periode; Servicegrad β = 1 − nicht sofort gelieferter Bedarf / neuer Bedarf der Stufe. Ohne festen Servicegrad sieht Glättung billiger aus, als sie ist
(bei festem z fällt der Service). Deshalb wird z je Stufe per Bisektion so gewählt, dass β = 95 % (oberste Stufe zuerst, zwei Durchläufe; Kalibrierung auf 200 × 1200) und **auf einer unabhängigen Stichprobe** (500 × 1500) bewertet.

**Was die Live-Zahlen bedeuten.** Ein Verlauf streut; die Live-Zahlen tragen deshalb die Überschrift „ein Verlauf“, die Meldung stützt sich auf die **vorgerechnete Messreihe**: „senkt hier den Kettenbestand“ (Änderung der Summe Lager gegenüber der
Basis derselben Zelle höchstens −5 % und unter −2 Standardfehlern), „kostet hier“ (über +2 % und über 2 Standardfehlern) oder „bringt hier wenig“ (dazwischen). **Die Stufen der Regler sind gemessene Zellen.** Die Messreihe variiert je Zelle nur **einen**
Parameter ausgehend vom Hauptszenario (nur dieses hat alle 13 Maßnahmen, die neun übrigen Zellen Basis, Glättung γ 0,5 und Info teilen); für andere Kombinationen zeigt die App die nächstliegende gemessene Zelle und sagt es. Das z je Stufe stammt aus der
nächstliegenden gemessenen Konfiguration; der Knopf „Servicegrad live nachkalibrieren“ bestimmt es grob neu (20 × 700, ein Durchlauf: z weicht typisch um ±0,1 ab, der Servicegrad um etwa ±1 Punkt).

**Produktformel und exakte Theorie.** Ohne Sicherheitsbestand und Abschneidung ist die Verstärkung eine Faltung; die bekannte Produktformel ist ab Stufe 2 nur eine Untergrenze (Verhältnis exakt/Produkt an Stufe 4 zwischen {F.fmt_num(_E1_MIN['ratio4'], 2)} und
{F.fmt_num(_E1_MAX['ratio4'], 2)}). Die Kontrollzeile rechnet die lineare Kette auf Ihrem Verlauf nach.

**Grenzen dieses Modells** (bewusst so gewählt, damit die Aussage ehrlich bleibt):

- **Stark stilisiert** – vier Stufen mit gleichen L und p, ein Produkt, unbegrenzter Lieferant, keine Kapazitätsgrenzen, keine Losgrößen, keine Informationsverzögerung, keine Rationierung bei Engpässen. Parameter **erfunden, nicht kalibriert**; das Servicegrad-Ziel β = 95 % in **allen** Stufen ist eine Modellwahl.
- **Lehrbuchthema.** Der Bullwhip-Effekt und die Formel sind bekannt (Forrester, Lee et al., Chen et al.); neu sind hier die Kosten- und Anreizsicht bei gleichem Servicegrad, nicht die Formel. Chen et al. wurden nicht nachgeschlagen.
- **Varianzkosten nicht bewertet.** Kapazität, Umrüstung und Expresszuschläge fehlen; der Nutzen der Glättung für die oberen Stufen ist deshalb eher unterschätzt. Die Kosten von L-Verkürzung, Datenaustausch und längeren Fenstern ebenfalls nicht.
- **Kleine Fenster** (p ≤ 6) erzeugen durch die σ-Schätzung extreme Verstärkung: ein Schätzartefakt der Regel, kein Naturgesetz. **Glättung** ist als proportionaler Regler modelliert; andere Formen sind nicht getestet.
- **Die Sicherheitsbestandsrechnung** nutzt die Ein-Perioden-Streuung aus demselben Fenster; mit anderer Schätzung ändern sich die absoluten Zahlen.
- **Nicht Teil dieser Demo** – Kettenlängen außer vier, ungleiche L und p je Stufe, andere Prognoseverfahren (exponentielle Glättung, Trend), Losgrößen und Kapazitäten, Rationierung, Informationsverzögerung, die Kosten der Maßnahmen selbst, Kalibrierung an echten Lieferketten.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/bullwhip-effekt-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Kette mit vier Stufen, der Endkundenbedarf und die Regler der Seitenleiste. Gesucht ist, wie stark die Bestellungen je Stufe schwanken (Verstärkung $V_k$) und wie viel Lager jede Maßnahme bei gleichem Servicegrad kostet.

| Symbol | Bedeutung |
|---|---|
| $k$, $t$ | Stufe (1 = Einzelhandel), Periode |
| $p$ | Länge des Prognosefensters |
| $L_k = m_k$ | Wiederbeschaffungszeit, zugleich Schutzintervall |
| $\hat\mu_k$, $\hat\sigma_k$ | geschätztes Mittel und geschätzte Streuung des Bedarfs der Stufe |
| $z_k$ | Sicherheitsfaktor (so gewählt, dass der Servicegrad 95 % trifft) |
| $IP_k$ | Bestandsposition |
| $q_k$ | Bestellung der Stufe |
| $\gamma$ | Glättung (1 = keine) |
| $\rho$ | Autokorrelation des Bedarfs |
| $V_k$ | Verstärkung: Varianz der Bestellung geteilt durch Varianz des Bedarfs |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/bullwhip-effekt-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Order-up-to mit gleitendem Durchschnitt.** Stufe $k$ prognostiziert aus der Eingangsreihe $x_k$ (Bedarf der Stufe; mit geteilter Information der Endkundenbedarf) Mittel $\hat\mu_k(t)$ und Standardabweichung $\hat\sigma_k(t)$ der letzten $p$ Werte
(einschließlich des aktuellen) und bestellt bei Wiederbeschaffungszeit $L_k = m_k$:
$$S_k(t) = m_k\,\hat\mu_k(t) + z_k\,\hat\sigma_k(t)\sqrt{m_k},\qquad q_k(t) = \max\bigl(0,\; S_k(t) - IP_k(t)\bigr),\qquad x_{k+1}(t) = q_k(t).$$
Bestandsposition $IP_k$ = Lager + Unterwegs + beim Lieferanten offen − eigener Rückstand.

**Glättung.** Nur der Anteil $\gamma$ der Bestandslücke wird korrigiert: $q_k = \max\bigl(0,\; \hat\mu_k + \gamma\,(S_k - IP_k - \hat\mu_k)\bigr)$; $\gamma = 1$ ist die reine Order-up-to-Regel.

**Lineare Theorie** (ohne Sicherheitsbestand $z = 0$, ohne Abschneidung, $\gamma = 1$): jede Stufe ist ein FIR-Filter
$$q_t = \Bigl(1 + \tfrac{m}{p}\Bigr)x_t - \tfrac{m}{p}\,x_{t-p},$$
die Kette die Faltung $h_k = h^{(1)} * \dots * h^{(k)}$ der Stufenfilter. Die Verstärkung ist der Varianzquotient
$$V_k = \frac{\operatorname{Var}(q_k)}{\operatorname{Var}(D)} = h_k^\top \Sigma_\rho\, h_k,\qquad (\Sigma_\rho)_{ij} = \rho^{|i-j|},$$
für i.i.d.-Bedarf $V_k = \sum_j h_k(j)^2$. Stufe 1: $V_1 = 1 + 2m/p + 2m^2/p^2$ (exakt, mit $m$ = Schutzintervall); die Produktformel $\prod_k V_1$ ist ab Stufe 2 nur eine Untergrenze. Mit geteilter Information wächst
das Schutzintervall additiv, $M_k = L_1 + \dots + L_k$, und $V_k = 1 + 2M_k/p + 2M_k^2/p^2$.

**Servicegrad und Kosten.** Mengen-Servicegrad $\beta_k = 1 - \dfrac{\text{nicht sofort gelieferter Bedarf}}{\text{neuer Bedarf der Stufe } k}$, Kosten je Stufe $h \cdot \text{Lager} + \pi \cdot \text{Rückstand}$ je Periode ($h = 1$, $\pi = 9$).
**Gleicher Servicegrad:** $z_k$ so, dass $\beta_k = 95\,\%$ (Bisektion, oberste Stufe zuerst, auf einer Stichprobe kalibriert, auf einer unabhängigen bewertet).

**Änderung und Meldung.** $\Delta = 100\,\bigl(T_{\text{Maßnahme}}/T_{\text{Basis}} - 1\bigr)$ mit der Summe Lager $T$ über alle vier Stufen, Standardfehler nach Fehlerfortpflanzung. Meldung: „senkt“ für $\Delta \le -5\,\%$ und $\Delta < -2\,\text{SE}$,
„kostet“ für $\Delta > +2\,\%$ und $\Delta > 2\,\text{SE}$, sonst „bringt wenig“.

Implementiert in `bw_model.py` (Kette, Lagerführung, Kalibrierung), `bw_theory.py` (exakte lineare Theorie), `bw_live.py` (Live-Verlauf, Schnellkalibrierung), `bw_results.py` (Messreihe, Zell-Zuordnung, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zum Thema: [Bestandsmanagement optimieren](https://sebastianhanisch.net/bestandsmanagement-optimierung.html)."
)
