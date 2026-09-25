"""Plotly-Figuren: Konventionen des Portfolios (Achsen fest, Vorlage plotly_white, keine Farblisten in Marker-Eigenschaften, Legende unten) und der Inhalt (Spuren, Werte, Standardfehler,
hervorgehobene Zelle)."""
import math

import pytest

import bw_constants as C
import bw_frozen as FZ
import bw_live as LV
import bw_results as R
import bw_visualization as V

DATA = R.load_results()
LIVE = LV.solve_live(0.2, 0.0, 2, 10, 0.5, C.ZMODE_EQUAL, C.SEED_DEFAULT, demand=FZ.preset_series(0.2, 0.0))
MAIN = R.main_rows(DATA)


def all_figures():
    payer = R.payer_rows(DATA, 0.5)
    sm, inf = R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5)
    return {
        "whip": V.whip_figure(LIVE, C.M_SMOOTH), "detail_q": V.detail_figure(LIVE, C.M_SMOOTH, 4, "Bestellungen"), "detail_oh": V.detail_figure(LIVE, C.M_INFO, 1, "Lager"),
        "stage_bars": V.stage_bars_figure(LIVE, C.M_SMOOTH), "theory": V.theory_figure(LIVE["control"]), "measures": V.measures_figure(MAIN), "fixed": V.fixed_service_figure(R.fixed_rows(DATA)),
        "payer": V.payer_figure(payer), "regime": V.regime_figure(sm, inf, "rho0.8"), "proxy": V.proxy_figure(R.proxy_rows(DATA)["points"]), "formula": V.formula_figure(R.formula_case(DATA, 2, 5)),
        "jump": V.window_jump_figure(R.window_jump_rows(DATA)), "map": V.linear_map_figure(DATA, 0.0),
    }


FIGS = all_figures()


@pytest.mark.parametrize("name", list(FIGS))
def test_every_figure_locks_its_axes_and_uses_the_portfolio_template(name):
    fig = FIGS[name]
    assert all(ax.fixedrange is True for ax in fig.select_xaxes()) and all(ax.fixedrange is True for ax in fig.select_yaxes())
    assert fig.layout.template.layout.plot_bgcolor == "white"
    assert fig.layout.height and fig.layout.height >= 250 and fig.data


@pytest.mark.parametrize("name", list(FIGS))
def test_no_figure_uses_color_lists_in_marker_properties(name):
    for tr in FIGS[name].data:
        marker = getattr(tr, "marker", None)
        if marker is None:
            continue
        color = marker.color
        assert color is None or isinstance(color, str), (name, tr.name, type(color))
        size = getattr(marker, "size", None)
        assert size is None or isinstance(size, (int, float)), (name, tr.name)


def test_legends_sit_below_the_plot():
    for name in ("whip", "detail_q", "measures", "fixed", "payer", "regime", "proxy", "formula", "jump", "theory", "stage_bars"):
        lg = FIGS[name].layout.legend
        assert lg.orientation == "h" and lg.y == 0.0 and lg.yanchor == "bottom", name


def test_whip_figure_shows_demand_and_four_stages_for_basis_and_the_measure():
    f = FIGS["whip"]
    assert len(f.data) == 10 and [t.name for t in f.data[:5]] == ["Endkundenbedarf", "Stufe 1 (Einzelhandel)", "Stufe 2 (Großhandel)", "Stufe 3 (Distributor)", "Stufe 4 (Fabrik)"]
    assert [t.showlegend for t in f.data] == [True] * 5 + [False] * 5
    assert len(f.data[0].x) == 300 and f.data[0].x[0] == 1 and f.data[0].x[-1] == 300 and [a.text for a in f.layout.annotations] == ["Basis", "alle glätten"]
    q_basis = LIVE["measures"][C.M_BASIS]["q"][3, 100:]
    assert list(f.data[4].y) == pytest.approx(list(q_basis)) and list(f.data[9].y) == pytest.approx(list(LIVE["measures"][C.M_SMOOTH]["q"][3, 100:]))
    assert list(f.data[0].y) == pytest.approx(list(LIVE["D"][100:])) and f.layout.yaxis2.matches == "y"


def test_detail_figure_switches_between_orders_and_stock_and_picks_the_stage():
    q, oh = FIGS["detail_q"], FIGS["detail_oh"]
    assert [t.name for t in q.data] == ["Endkundenbedarf", "Basis", "alle glätten"] and [t.name for t in oh.data] == ["Basis", "Info teilen"]
    assert list(q.data[2].y) == pytest.approx(list(LIVE["measures"][C.M_SMOOTH]["q"][3, 100:])) and list(oh.data[1].y) == pytest.approx(list(LIVE["measures"][C.M_INFO]["onhand"][0, 100:]))
    assert q.layout.yaxis.title.text == "Bestellmenge je Periode" and oh.layout.yaxis.title.text == "Lager am Periodenende"
    for stage in (1, 2, 3, 4):
        f = V.detail_figure(LIVE, C.M_SMOOTH, stage, "Lager")
        assert list(f.data[0].y) == pytest.approx(list(LIVE["measures"][C.M_BASIS]["onhand"][stage - 1, 100:]))


def test_stage_bars_show_amplification_and_stock_for_basis_and_measure():
    f = FIGS["stage_bars"]
    assert len(f.data) == 4 and f.layout.yaxis.type == "log" and list(f.data[0].y) == pytest.approx(LIVE["measures"][C.M_BASIS]["amp"]) and list(f.data[3].y) == pytest.approx(LIVE["measures"][C.M_SMOOTH]["onhand_mean"])
    assert [t.name for t in f.data] == ["Basis", "Basis", "alle glätten", "alle glätten"] and [t.showlegend for t in f.data][1::2] == [False, False]


def test_theory_figure_has_run_theory_and_product_bars_on_a_log_axis():
    f = FIGS["theory"]
    assert [t.name for t in f.data] == ["Simulation dieses Verlaufs (z = 0)", "exakte Theorie (Faltung)", "Produktformel (Untergrenze)"] and f.layout.yaxis.type == "log"
    assert list(f.data[1].y) == pytest.approx(LIVE["control"]["amp_exact"]) and list(f.data[2].y) == pytest.approx(LIVE["control"]["amp_product"])


def test_measures_figure_colours_by_change_and_shows_standard_errors():
    f = FIGS["measures"]
    by = {t.name: t for t in f.data}
    assert set(by) == {"weniger Bestand als die Basis", "mehr Bestand als die Basis", "Basis"}
    assert list(by["Basis"].y) == ["Basis"] and by["Basis"].x[0] == pytest.approx(270.158, abs=0.01) and by["Basis"].marker.color == C.COLOR_NEUTRAL
    less = by["weniger Bestand als die Basis"]
    assert "Fabrik glättet" not in " ".join(less.y) and len(less.y) == 11 and less.marker.color == C.COLOR_GOOD and all(e > 0 for e in less.error_x.array)
    more = by["mehr Bestand als die Basis"]
    assert list(more.y) == ["nur Fabrik glättet, γ 0,3"] and more.marker.color == C.COLOR_BAD and f.layout.yaxis.autorange == "reversed"
    assert sum(len(t.y) for t in f.data) == 13 and f.layout.height >= 26 * 13 + 120


def test_fixed_service_figure_shows_the_service_drop_and_the_target_line():
    f = FIGS["fixed"]
    assert [t.name for t in f.data] == ["Basis", "Glättung γ 0,5 (alle)", "Glättung γ 0,3 (alle)", "Bedarfsinformation teilen"]
    assert list(f.data[2].y)[0] == pytest.approx(86.8, abs=0.1) and list(f.data[0].y)[3] == pytest.approx(93.5, abs=0.1)
    assert any(s.y0 == 95.0 for s in f.layout.shapes) and f.layout.yaxis.range == (60, 100)


def test_payer_figure_shows_three_series_and_the_change_in_the_hover():
    f = FIGS["payer"]
    assert [t.name for t in f.data] == ["Basis", "alle glätten (γ 0,5)", "nur Einzelhandel glättet (γ 0,5)"] and list(f.data[0].x) == ["Stufe 1", "Stufe 2", "Stufe 3", "Stufe 4"] and f.layout.xaxis.tickangle == 0
    assert list(f.data[0].y)[0] == pytest.approx(29.5, abs=0.1) and "gegenüber der Basis" in f.data[1].hovertemplate and "gegenüber der Basis" not in f.data[0].hovertemplate


def test_regime_figure_highlights_the_cell_of_the_setting():
    f = FIGS["regime"]
    names = [t.name for t in f.data]
    assert names == ["Glättung (alle Stufen) (Zelle Ihrer Einstellung)", "Glättung (alle Stufen)", "Info teilen (Zelle Ihrer Einstellung)", "Info teilen"]
    assert list(f.data[0].y) == ["ρ 0,8"] and len(f.data[1].y) == 9 and f.data[0].marker.color == C.COLOR_MEASURE and f.data[1].marker.color == C.COLOR_LIGHT
    assert f.data[0].x[0] == pytest.approx(6.9, abs=0.1) and all(e > 0 for e in f.data[1].error_x.array) and list(f.layout.yaxis.categoryarray)[0] == "Hauptszenario"
    nothing = V.regime_figure(R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5), "gibt es nicht")
    assert [t.name for t in nothing.data] == ["Glättung (alle Stufen)", "Info teilen"]


def test_proxy_figure_separates_lever_measures_and_uses_a_log_axis():
    f = FIGS["proxy"]
    assert [t.name for t in f.data] == ["Maßnahmen bei gleichem L und p", "Maßnahmen, die L oder p ändern"] and len(f.data[0].x) == 10 and len(f.data[1].x) == 3
    assert f.layout.xaxis.type == "log" and f.data[1].marker.symbol == "diamond" and {c[0] for c in f.data[1].customdata} == {"Wiederbeschaffungszeit halbiert", "L halbiert + Info teilen", "Fenster verdoppelt (p 20)"}
    assert sorted(t for tr in f.data for t in tr.text if t) == sorted(V.PROXY_LABELLED)                                      # nur vier Punkte tragen einen Text
    assert "customdata[0]" in f.data[0].hovertemplate


def test_formula_figure_has_simulation_with_error_bars_and_the_two_formulas():
    f = FIGS["formula"]
    assert [t.name for t in f.data] == ["Simulation", "exakte Theorie (Faltung)", "Produktformel (Untergrenze)"] and list(f.data[1].y)[3] == pytest.approx(37.70, abs=0.01) and list(f.data[2].y)[3] == pytest.approx(20.20, abs=0.01)
    assert all(e > 0 for e in f.data[0].error_y.array) and list(f.data[0].x) == ["Stufe 1", "Stufe 2", "Stufe 3", "Stufe 4"]


def test_window_jump_figure_has_both_series_over_the_windows():
    f = FIGS["jump"]
    assert [t.name for t in f.data] == ["i.i.d.-Bedarf", "Bedarf mit Niveausprüngen"] and list(f.data[0].x) == [4, 8, 16, 32, 64] and f.layout.xaxis.type == "log" and f.layout.yaxis.type == "log"
    assert min(f.data[1].y) == f.data[1].y[3] and f.data[0].y[0] > 1000


def test_linear_map_figure_is_the_log_of_the_exact_stage_amplification():
    f = FIGS["map"]
    z = f.data[0].z
    assert len(z) == 4 and len(z[0]) == 5 and z[0][0] == pytest.approx(math.log10(R.formula_case(DATA, 1, 3)["exact"][3]), abs=1e-4)
    local = {(r["m"], r["p"]): r["local"][3] for r in DATA["E3"] if r["rho"] == 0.0}
    assert f.data[0].text[0][0] == "21,4" and f.data[0].text[2][1] == R.F.fmt_num(local[(4, 5)], 1) and f.data[0].text[3][0] == R.F.fmt_num(local[(8, 3)], 1)
    f9 = V.linear_map_figure(DATA, 0.9, stage=1)
    assert len(f9.data[0].z) == 4 and "Stufe 1" in f9.data[0].hovertemplate


def test_lock_axes_and_layout_helpers():
    import plotly.graph_objects as go
    fig = V._lock_axes(go.Figure())
    assert fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True
    assert V.LEGEND_BOTTOM["orientation"] == "h" and V.ERR["thickness"] > 0


# ---------------------------------------------------------------------------------------------------
# Lücken aus dem Fehler-Einbau-Test: Höhen, Fehlerbalken, Farbskala, Kartenwerte
# ---------------------------------------------------------------------------------------------------
def test_figure_heights_follow_the_chart_height_constant():
    h = C.CHART_HEIGHT
    want = {"whip": h + 60, "detail_q": h - 40, "detail_oh": h - 40, "stage_bars": h - 60, "theory": h - 60, "measures": max(h, 26 * 13 + 120), "fixed": h - 40, "payer": h - 20,
            "regime": max(h, 46 * 10 + 120), "proxy": h + 20, "formula": h - 40, "jump": h - 40, "map": h - 80}
    assert {k: FIGS[k].layout.height for k in want} == want


def test_standard_errors_are_drawn_where_the_measurement_series_has_them():
    assert all(t.error_x.visible is True and list(t.error_x.array) for t in FIGS["measures"].data)
    assert all(t.error_x.visible is True and all(e >= 0 for e in t.error_x.array) for t in FIGS["regime"].data)
    assert FIGS["formula"].data[0].error_y.visible is True and all(t.error_y.visible is True for t in FIGS["jump"].data)
    assert FIGS["measures"].data[0].error_x.array[0] == pytest.approx(0.3, abs=0.15)


def test_measures_bars_are_split_by_the_sign_of_the_change():
    by = {t.name: t for t in FIGS["measures"].data}
    less, more = by["weniger Bestand als die Basis"], by["mehr Bestand als die Basis"]
    assert all(c < 0 for c in (cd[0] for cd in less.customdata)) and all(c > 0 for c in (cd[0] for cd in more.customdata)) and by["Basis"].customdata[0][0] == 0.0


def test_linear_map_has_no_colour_bar_and_the_exact_log_values_in_every_cell():
    f = FIGS["map"]
    assert f.data[0].showscale is False
    rows = {(r["m"], r["p"]): r["local"][3] for r in DATA["E3"] if r["rho"] == 0.0}
    ms, ps = (1, 2, 4, 8), (3, 5, 10, 20, 40)
    for i, m in enumerate(ms):
        for j, p in enumerate(ps):
            assert f.data[0].z[i][j] == pytest.approx(math.log10(rows[(m, p)]), abs=1e-4), (m, p)
    assert list(f.layout.yaxis.ticktext) == ["m = 1", "m = 2", "m = 4", "m = 8"] and list(f.layout.xaxis.ticktext) == ["p = 3", "p = 5", "p = 10", "p = 20", "p = 40"]


def test_regime_figure_height_grows_with_the_number_of_cells():
    sm, inf = R.regime_rows(DATA, C.M_SMOOTH, 0.5), R.regime_rows(DATA, C.M_INFO, 0.5)
    assert V.regime_figure(sm[:3], inf[:3], "main").layout.height == C.CHART_HEIGHT and V.regime_figure(sm, inf, "main").layout.height == 46 * 10 + 120
