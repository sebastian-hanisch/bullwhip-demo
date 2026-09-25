"""Plotly-Figuren der Bullwhip-Demo: Zeitreihen (die Peitsche, Detail einer Stufe), Verstärkung und Lager je Stufe, und die vorgerechneten Auswertungen der Messreihe (Maßnahmen bei gleichem
Servicegrad, fester z gegen gleicher Servicegrad, wer zahlt, Regime-Karte, Kostenproxy, Formel gegen Simulation, Fenster mit Niveausprüngen, lineare Karte).

Konventionen des Portfolios: Achsen `fixedrange` (Touch-Scrollen), Vorlage plotly_white, Legende unten, Farben über alle Figuren konsistent, KEINE Farblisten in Marker-Eigenschaften (je Merkmal
eine eigene Spur; numerische Werte mit Farbskala sind erlaubt), Balken mit Standardfehler wo die Messreihe einen liefert. Plotly wird erst in den Funktionen importiert, damit die reine Rechnung
ohne Plotly testbar bleibt."""
import math

import bw_constants as C
import bw_format as F
import bw_results as R

PROXY_LABELLED = ("Basis", "nur Fabrik glättet, γ 0,3", "Wiederbeschaffungszeit halbiert", "Glättung γ 0,3 (alle)")       # nur diese Punkte tragen einen Text (die übrigen zeigt der Hover und die Tabelle)
LEGEND_BOTTOM = dict(orientation="h", yref="container", yanchor="bottom", y=0.0, x=0)
ERR = dict(color="#4b5d75", thickness=1.2)


def _lock_axes(fig):
    """Achsen fest: verhindert Zoomen und Verschieben per Touch, damit die Seite scrollbar bleibt (Hover bleibt)."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base_layout(fig, height, bottom=110, **kw):
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=20, b=bottom, l=10, r=10), legend=LEGEND_BOTTOM, **kw)
    return fig


def _series(live, m, key):
    return live["measures"][m][key][:, live["burn"]:]


# ---------------------------------------------------------------------------------------------------
# Live-Verlauf
# ---------------------------------------------------------------------------------------------------
def whip_figure(live, measure):
    """Die Peitsche: Endkundenbedarf und Bestellungen der vier Stufen, links die Basis, rechts die gewählte Maßnahme (gleiche Achse)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    burn = live["burn"]
    t = list(range(1, live["T"] - burn + 1))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=(C.MEASURE_LABELS[C.M_BASIS], C.MEASURE_LABELS[measure]))
    for col, m in ((1, C.M_BASIS), (2, measure)):
        q = _series(live, m, "q")
        fig.add_trace(go.Scatter(x=t, y=[float(v) for v in live["D"][burn:]], mode="lines", name="Endkundenbedarf", showlegend=(col == 1), line=dict(color=C.COLOR_DEMAND, width=2),
                                 hovertemplate="Endkundenbedarf %{y:.0f} in Periode %{x}<extra></extra>"), row=1, col=col)
        for k in range(C.N_STAGES):
            name = f"Stufe {k + 1} ({C.STAGE_NAMES[k]})"
            fig.add_trace(go.Scatter(x=t, y=[float(v) for v in q[k]], mode="lines", name=name, showlegend=(col == 1), line=dict(color=C.STAGE_COLORS[k], width=1.3),
                                     hovertemplate=f"{name}: Bestellung %{{y:.0f}} in Periode %{{x}}<extra></extra>"), row=1, col=col)
    _base_layout(fig, C.CHART_HEIGHT + 60, bottom=120)
    fig.update_yaxes(title_text="Bestellmenge je Periode", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="Periode")
    return _lock_axes(fig)


def detail_figure(live, measure, stage, view):
    """Eine Stufe im Detail: Basis gegen Maßnahme, als Bestellungen (mit Endkundenbedarf zum Vergleich) oder als Lager."""
    import plotly.graph_objects as go

    burn = live["burn"]
    t = list(range(1, live["T"] - burn + 1))
    k = int(stage) - 1
    key, unit = ("q", "Bestellmenge je Periode") if view == "Bestellungen" else ("onhand", "Lager am Periodenende")
    fig = go.Figure()
    if view == "Bestellungen":
        fig.add_trace(go.Scatter(x=t, y=[float(v) for v in live["D"][burn:]], mode="lines", name="Endkundenbedarf", line=dict(color=C.COLOR_DEMAND, width=1, dash="dot"),
                                 hovertemplate="Endkundenbedarf %{y:.0f} in Periode %{x}<extra></extra>"))
    for m, color in ((C.M_BASIS, C.COLOR_BASE), (measure, C.COLOR_MEASURE)):
        y = _series(live, m, key)[k]
        label = C.MEASURE_LABELS[m]
        fig.add_trace(go.Scatter(x=t, y=[float(v) for v in y], mode="lines", name=label, line=dict(color=color, width=1.6),
                                 hovertemplate=f"{label}: %{{y:.0f}} in Periode %{{x}}<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT - 40, bottom=100, xaxis_title="Periode", yaxis=dict(title=unit, rangemode="tozero"))
    return _lock_axes(fig)


def stage_bars_figure(live, measure):
    """Verstärkung (log) und mittleres Lager je Stufe: Basis gegen Maßnahme, dieser eine Verlauf; links Verstärkung, rechts Lager."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stages = [f"Stufe {k + 1}" for k in range(C.N_STAGES)]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=("Verstärkung Var(Bestellung)/Var(Bedarf)", "mittleres Lager je Stufe"))
    for m, color in ((C.M_BASIS, C.COLOR_BASE), (measure, C.COLOR_MEASURE)):
        v = live["measures"][m]
        label = C.MEASURE_LABELS[m]
        fig.add_trace(go.Bar(x=stages, y=v["amp"], name=label, marker_color=color, legendgroup=m, hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=stages, y=v["onhand_mean"], name=label, marker_color=color, legendgroup=m, showlegend=False, hovertemplate=f"{label}: %{{y:.0f}} Einheiten<extra></extra>"), row=1, col=2)
    _base_layout(fig, C.CHART_HEIGHT - 60, bottom=100, barmode="group")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    return _lock_axes(fig)


def theory_figure(control, live_amp=None):
    """Verstärkung je Stufe: Kontrolllauf dieses Verlaufs (z = 0, ohne Abschneidung), exakte Theorie und Produktformel (log)."""
    import plotly.graph_objects as go

    stages = [f"Stufe {k + 1}" for k in range(C.N_STAGES)]
    fig = go.Figure()
    for name, vals, color in (("Simulation dieses Verlaufs (z = 0)", control["amp_run"], C.COLOR_MEASURE), ("exakte Theorie (Faltung)", control["amp_exact"], C.COLOR_GOOD),
                              ("Produktformel (Untergrenze)", control["amp_product"], C.COLOR_WARN)):
        fig.add_trace(go.Bar(x=stages, y=vals, name=name, marker_color=color, hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT - 60, bottom=100, barmode="group", yaxis=dict(type="log", title="Verstärkung"))
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②: vorgerechnet
# ---------------------------------------------------------------------------------------------------
def measures_figure(rows):
    """Summe Lager (4 Stufen) bei gleichem Servicegrad je Maßnahme, Mittel ± Standardfehler; billiger als die Basis grün, teurer rot, Basis grau."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, color, test in (("weniger Bestand als die Basis", C.COLOR_GOOD, lambda r: r["key"] != "basis" and r["delta"] < 0), ("mehr Bestand als die Basis", C.COLOR_BAD, lambda r: r["key"] != "basis" and r["delta"] >= 0),
                              ("Basis", C.COLOR_NEUTRAL, lambda r: r["key"] == "basis")):
        sel = [r for r in rows if test(r)]
        if not sel:
            continue
        fig.add_trace(go.Bar(y=[r["label"] for r in sel], x=[r["total"] for r in sel], orientation="h", name=name, marker_color=color,
                             error_x=dict(type="data", array=[r["se"] for r in sel], visible=True, **ERR), customdata=[[r["delta"]] for r in sel],
                             hovertemplate="%{y}: Summe Lager %{x:.0f} (%{customdata[0]:+.1f} % gegenüber der Basis)<extra></extra>"))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[r["label"] for r in rows])
    _base_layout(fig, max(C.CHART_HEIGHT, 26 * len(rows) + 120), barmode="overlay", bargap=0.25, xaxis_title="Summe Lager (4 Stufen), β = 95 % je Stufe")
    return _lock_axes(fig)


def fixed_service_figure(rows, keys=("basis", "glaettung_0.5", "glaettung_0.3", "info_teilen")):
    """Servicegrad je Stufe bei festem z = 1,645 (die Kalibrierung entfällt): Basis, Glättung, Info teilen; gestrichelt das Ziel 95 %."""
    import plotly.graph_objects as go

    stages = [f"Stufe {k + 1}" for k in range(C.N_STAGES)]
    fig = go.Figure()
    styles = {"basis": C.COLOR_BASE, "glaettung_0.5": C.COLOR_MEASURE, "glaettung_0.3": C.COLOR_LIGHT, "info_teilen": C.COLOR_GOOD}
    for r in rows:
        if r["key"] not in keys:
            continue
        fig.add_trace(go.Bar(x=stages, y=[100 * b for b in r["beta_fixed"]], name=r["label"], marker_color=styles.get(r["key"], C.COLOR_NEUTRAL), hovertemplate=f"{r['label']}: β %{{y:.1f}} %<extra></extra>"))
    fig.add_hline(y=100 * C.TARGET_SERVICE, line=dict(color="#4b5d75", width=1.2, dash="dash"), annotation_text="Ziel 95 %", annotation_position="top left")
    _base_layout(fig, C.CHART_HEIGHT - 40, bottom=100, barmode="group", yaxis=dict(title="Servicegrad β bei festem z (%)", range=[60, 100]))
    return _lock_axes(fig)


def payer_figure(payer):
    """Wer zahlt: mittleres Lager je Stufe bei gleichem Servicegrad: Basis, alle glätten, nur Einzelhandel glättet (Hauptszenario)."""
    import plotly.graph_objects as go

    stages = [f"Stufe {k + 1}" for k in range(C.N_STAGES)]
    fig = go.Figure()
    for key, name, color in (("base", "Basis", C.COLOR_BASE), ("all", f"alle glätten (γ {F.fmt_level(payer['gamma_all'])})", C.COLOR_WARN),
                             ("retail", f"nur Einzelhandel glättet (γ {F.fmt_level(payer['gamma_retail'])})", C.COLOR_GOOD)):
        change = payer.get(key + "_change")
        custom = [[c] for c in change] if change else [[0.0]] * C.N_STAGES
        fig.add_trace(go.Bar(x=stages, y=payer[key], name=name, marker_color=color, customdata=custom,
                             hovertemplate=f"{name}: %{{y:.0f}} Einheiten" + ("" if not change else " (%{customdata[0]:+.1f} % gegenüber der Basis)") + "<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT - 20, bottom=100, barmode="group", xaxis=dict(tickangle=0), yaxis=dict(title="mittleres Lager je Stufe (Einheiten)", rangemode="tozero"))
    return _lock_axes(fig)


def regime_figure(smooth_rows, info_rows, marked_cell):
    """Regime-Karte: Änderung der Summe Lager gegenüber der Basis derselben Zelle (Mittel ± Standardfehler) für Glättung und Info teilen; die Zelle der Einstellung ist kräftig gefärbt."""
    import plotly.graph_objects as go

    fig = go.Figure()
    labels = [r["label"] for r in smooth_rows]
    for rows, name, strong, light in ((smooth_rows, "Glättung (alle Stufen)", C.COLOR_MEASURE, C.COLOR_LIGHT), (info_rows, "Info teilen", C.COLOR_GOOD, "#a9d3b9")):
        for highlight, suffix, color in ((True, " (Zelle Ihrer Einstellung)", strong), (False, "", light)):
            sel = [r for r in rows if (r["cell"] == marked_cell) == highlight]
            if not sel:
                continue
            fig.add_trace(go.Bar(y=[r["label"] for r in sel], x=[r["delta"] for r in sel], orientation="h", name=name + suffix, marker_color=color, legendgroup=name, offsetgroup=name,
                                 error_x=dict(type="data", array=[r["se"] for r in sel], visible=True, **ERR),
                                 hovertemplate=f"{name}: %{{x:+.1f}} % gegenüber der Basis derselben Zelle<extra></extra>"))
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    _base_layout(fig, max(C.CHART_HEIGHT, 46 * len(labels) + 120), barmode="group", bargap=0.2, xaxis_title="Änderung Summe Lager gegenüber Basis (%)")
    return _lock_axes(fig)


def proxy_figure(points):
    """Verstärkung an Stufe 4 (log) gegen Summe Lager je Maßnahme: das Ranking nach Verstärkung sagt das Ranking nach Bestand kaum voraus. Maßnahmen, die L oder p ändern, als Raute."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for lever, name, symbol, color in ((False, "Maßnahmen bei gleichem L und p", "circle", C.COLOR_MEASURE), (True, "Maßnahmen, die L oder p ändern", "diamond", C.COLOR_WARN)):
        sel = [p for p in points if p["lever"] == lever]
        if not sel:
            continue
        fig.add_trace(go.Scatter(x=[p["amp4"] for p in sel], y=[p["total"] for p in sel], mode="markers+text", name=name, text=[p["label"] if p["label"] in PROXY_LABELLED else "" for p in sel],
                                 customdata=[[p["label"]] for p in sel], textposition="top center", textfont=dict(size=10), marker=dict(symbol=symbol, size=11, color=color, line=dict(color="white", width=1)),
                                 hovertemplate="%{customdata[0]}: Verstärkung Stufe 4 %{x:.2f}, Summe Lager %{y:.0f}<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT + 20, bottom=100, xaxis=dict(type="log", title="Verstärkung an Stufe 4 (log)"), yaxis=dict(title="Summe Lager, β = 95 % je Stufe", rangemode="tozero"))
    return _lock_axes(fig)


def formula_figure(row):
    """Formel gegen Simulation (i.i.d., z = 0, ohne Abschneidung): Verstärkung je Stufe aus Simulation, exakter Theorie und Produktformel."""
    import plotly.graph_objects as go

    stages = [f"Stufe {k + 1}" for k in range(C.N_STAGES)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stages, y=row["sim"], name="Simulation", marker_color=C.COLOR_MEASURE, error_y=dict(type="data", array=row["sim_se"], visible=True, **ERR),
                         hovertemplate="Simulation %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=stages, y=row["exact"], name="exakte Theorie (Faltung)", marker_color=C.COLOR_GOOD, hovertemplate="exakte Theorie %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=stages, y=row["product"], name="Produktformel (Untergrenze)", marker_color=C.COLOR_WARN, hovertemplate="Produktformel %{y:.2f}<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT - 40, bottom=100, barmode="group", yaxis=dict(title="Verstärkung Var(Bestellung)/Var(Bedarf)", rangemode="tozero"))
    return _lock_axes(fig)


def window_jump_figure(rows):
    """Fenster mit Niveausprüngen: Summe Lager bei gleichem Servicegrad über das Fenster, i.i.d.-Bedarf gegen Bedarf mit zufälligen Niveausprüngen (log)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for key, se_key, name, color in (("iid", "iid_se", "i.i.d.-Bedarf", C.COLOR_MEASURE), ("shift", "shift_se", "Bedarf mit Niveausprüngen", C.COLOR_WARN)):
        fig.add_trace(go.Scatter(x=[r["p"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=2), marker=dict(size=8),
                                 error_y=dict(type="data", array=[r[se_key] for r in rows], visible=True, **ERR), hovertemplate=f"{name}: Fenster %{{x}}, Summe Lager %{{y:.0f}}<extra></extra>"))
    _base_layout(fig, C.CHART_HEIGHT - 40, bottom=100, xaxis=dict(type="log", title="Prognosefenster p", tickvals=[r["p"] for r in rows]), yaxis=dict(type="log", title="Summe Lager, β = 95 % je Stufe"))
    return _lock_axes(fig)


def linear_map_figure(data, rho, stage=4):
    """Lineare Karte (exakte Theorie): Verstärkung an Stufe 4 über m (Wiederbeschaffungszeit) und p (Fenster) bei Autokorrelation ρ, lokal gegen geteilte Bedarfsinformation (Verhältnis)."""
    import plotly.graph_objects as go

    rows = [r for r in data["E3"] if abs(r["rho"] - rho) < 1e-9]
    ms = sorted({r["m"] for r in rows})
    ps = sorted({r["p"] for r in rows})
    z = [[math.log10(next(r for r in rows if r["m"] == m and r["p"] == p)["local"][stage - 1]) for p in ps] for m in ms]
    txt = [[F.fmt_num(next(r for r in rows if r["m"] == m and r["p"] == p)["local"][stage - 1], 1) for p in ps] for m in ms]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(len(ps))), y=list(range(len(ms))), text=txt, texttemplate="%{text}", colorscale="YlOrRd", showscale=False,
                               hovertemplate="Verstärkung Stufe " + str(stage) + ": %{text}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 80, margin=dict(t=20, b=50, l=10, r=10),
                      xaxis=dict(tickvals=list(range(len(ps))), ticktext=[f"p = {p}" for p in ps], title="Prognosefenster p"),
                      yaxis=dict(tickvals=list(range(len(ms))), ticktext=[f"m = {m}" for m in ms], autorange="reversed", title="Schutzintervall m = L"))
    return _lock_axes(fig)
