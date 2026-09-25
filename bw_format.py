"""Zahlenformate mit deutschem Dezimalkomma und Tausenderpunkt."""


def fmt_num(v, digits=1, signed=False):
    text = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return text.replace(".", ",")


def fmt_level(v):
    """Reglerstufe (Schwankung, Autokorrelation, Glättung) mit Dezimalkomma: 0 -> '0,0', 0.1 -> '0,1', 0.25 -> '0,25', 0.8 -> '0,8'."""
    text = f"{v:.1f}" if abs(v - round(v, 1)) < 1e-9 else f"{v:g}"
    return text.replace(".", ",")


def fmt_share(v):
    """Anteil (0..1) als ganze Prozent; None (nicht bestimmbar) als Gedankenstrich."""
    return "–" if v is None else f"{100.0 * v:.0f} %"


def fmt_pct(v, digits=1, signed=False):
    """Anteil (0..1) als Prozent mit Komma: 0.9512 -> '95,1 %'."""
    return f"{fmt_num(100.0 * v, digits, signed)} %"


def fmt_band(mean, se, digits=1, signed=True):
    """Mittel ± Standardfehler mit Vorzeichen: '-15,7 ± 0,1'."""
    return f"{fmt_num(mean, digits, signed)} ± {fmt_num(se, digits)}"


def fmt_int(v, signed=False):
    """Ganze Zahl mit Tausenderpunkt: '14.694'."""
    text = f"{v:+,.0f}" if signed else f"{v:,.0f}"
    return text.replace(",", ".")
