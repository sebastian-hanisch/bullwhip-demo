"""Abnahmekriterien der Presets (Detailplan plan_bullwhip.html, Abschnitt 7): welche Geschichte erzählt jedes Beispielszenario, und woran erkennt man, dass sie trägt?

Einzige Quelle für tests/test_stories.py (Schwellen einzeln kippen) und tests/test_preset_stories.py (Abnahme der echten Presets). Die Kriterien der Messreihe (`criteria`) stehen auf
data/bw_results.json: Änderung der Summe Lager gegenüber der Basis derselben Zelle bei gleichem Servicegrad, Vorzeichen-Kriterien IMMER zusammen mit der Standardfehler-Bedingung (Betrag > 2 SE).
Der gezeigte Verlauf (`day_criteria`) bleibt qualitativ, weil ein Verlauf streut; sein Seed liegt bewusst außerhalb der Messreihen-Seeds (alle Nachfrage-Seeds der Messreihe sind >= 10000).
Kein Löser mit Wall-Clock-Grenze in den Kriterien.

`criteria(name, data)` liefert für jedes Preset eine Liste (erfüllt, Text). Die Schwellen stehen als Konstanten oben, damit ein Test jede einzeln an ihre Grenze schieben kann."""
import bw_constants as C
import bw_results as R

FACTOR = C.SE_FACTOR

# Standard: Glättung spart mindestens 10 % Kettenbestand, drückt die Verstärkung an der Fabrik um mindestens 70 %, Info teilen spart mindestens 20 %
STANDARD_SMOOTH_MAX = -10.0
STANDARD_AMP_DROP_MIN = 0.70
STANDARD_INFO_MAX = -20.0
# Kurze Lieferzeit (L = 1): Glättung ändert fast nichts, Info teilen nur etwa -10 %
SHORT_SMOOTH_ABS_MAX = 2.0
SHORT_INFO_LO, SHORT_INFO_HI = -15.0, -5.0
# Träger Bedarf (ρ = 0,8): Glättung kostet (Änderung über 0), Info teilen spart mindestens 30 %
SLUGGISH_INFO_MAX = -30.0
# Kleines Fenster (p = 6): Glättung und Info teilen sparen mindestens 40 %
SMALL_SMOOTH_MAX = -40.0
SMALL_INFO_MAX = -40.0
# Großes Fenster (p = 40): weder Info teilen noch Glättung ändern mehr als 3 %
LARGE_ABS_MAX = 3.0

# --- gezeigter Verlauf (qualitativ) --------------------------------------------------------------------------------------
DAY_STANDARD_AMP_RATIO_MAX = 0.5       # Standard: Verstärkung der Fabrik mit Glättung höchstens halb so groß wie ohne (Messreihe im Mittel 0,26)
DAY_STANDARD_INFO_MAX = -10.0          # Standard: Info teilen spart auf dem Verlauf mindestens 10 % (Messreihe -26,9 %)
DAY_SHORT_SMOOTH_ABS_MAX = 15.0        # Kurze Lieferzeit: |Änderung durch Glättung| auf dem Verlauf unter 15 % (Messreihe +0,6 %)
DAY_SHORT_INFO_MIN = -20.0             # Kurze Lieferzeit: Info teilen spart auf dem Verlauf weniger als 20 % (Messreihe -10,0 %), aber etwas
DAY_SLUGGISH_SMOOTH_MIN = 0.0          # Träger Bedarf: Glättung ändert die Summe Lager auf dem Verlauf ins Positive (Messreihe +6,9 %)
DAY_SLUGGISH_INFO_MAX = -20.0          # Träger Bedarf: Info teilen spart auf dem Verlauf mindestens 20 % (Messreihe -36,0 %)
DAY_SMALL_MAX = -25.0                  # Kleines Fenster: Glättung und Info teilen sparen auf dem Verlauf mindestens 25 % (Messreihe -47 % und -53 %)
DAY_LARGE_ABS_MAX = 10.0               # Großes Fenster: |Änderung| auf dem Verlauf unter 10 % (Messreihe -1,8 % und -1,9 %)

PRESET_CELL_SETTING = {name: (p["cv"], p["rho"], p["lead"], p["window"]) for name, p in C.PRESETS.items()}


def _de(v, digits=1, sign=False):
    text = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return text.replace(".", ",")


def _band(mean, se):
    return f"{_de(mean, 1, True)} ± {_de(se)}"


def _pair(data, name, measure):
    """Änderung der Summe Lager der Maßnahme gegenüber der Basis in der Zelle des Presets `name` (Zelle und Glättung müssen genau gemessen sein)."""
    p = C.PRESETS[name]
    r = R.resolve_pair(data, measure, p["gamma"], p["cv"], p["rho"], p["lead"], p["window"])
    if not r["exact"]:
        raise ValueError(f"{name}: keine genau gemessene Zelle für {measure}")
    return r


def _below(delta, se, limit):
    """Änderung höchstens `limit` (negativ) und unter -2 Standardfehlern."""
    return delta <= limit and delta < -FACTOR * se


def criteria(name, data):
    """Abnahmekriterien des Presets `name` auf der Messreihe `data`: Liste (erfüllt, Text)."""
    if name not in C.PRESETS:
        raise KeyError(name)
    sm, info = _pair(data, name, C.M_SMOOTH), _pair(data, name, C.M_INFO)
    if name == "Standard":
        b, s = R.job(data, sm["base_name"]), R.job(data, sm["name"])
        a0, a1 = R.amp_last(b)[0], R.amp_last(s)[0]
        drop = 1.0 - a1 / a0
        return [
            (_below(sm["delta"], sm["se"], STANDARD_SMOOTH_MAX), f"Glättung γ 0,5: Summe Lager höchstens −10 % und unter −2 Standardfehlern: {_band(sm['delta'], sm['se'])} %"),
            (drop >= STANDARD_AMP_DROP_MIN, f"Stufe-4-Verstärkung sinkt um mindestens 70 %: {_de(a0)} auf {_de(a1)} ({_de(-100 * drop, 0, True)} %)"),
            (_below(info["delta"], info["se"], STANDARD_INFO_MAX), f"Info teilen: Summe Lager höchstens −20 % und unter −2 Standardfehlern: {_band(info['delta'], info['se'])} %"),
        ]
    if name == "Kurze Lieferzeit":
        return [
            (abs(sm["delta"]) < SHORT_SMOOTH_ABS_MAX, f"|Änderung durch Glättung| unter 2 %: {_band(sm['delta'], sm['se'])} %"),
            (SHORT_INFO_LO <= info["delta"] <= SHORT_INFO_HI and info["delta"] < -FACTOR * info["se"],
             f"Info teilen nur etwa −10 % (zwischen −15 % und −5 %, unter −2 Standardfehlern): {_band(info['delta'], info['se'])} %"),
        ]
    if name == "Träger Bedarf":
        return [
            (sm["delta"] > 0 and sm["delta"] > FACTOR * sm["se"], f"Glättung kostet (Änderung über 0 und über 2 Standardfehlern): {_band(sm['delta'], sm['se'])} %"),
            (_below(info["delta"], info["se"], SLUGGISH_INFO_MAX), f"Info teilen: Summe Lager höchstens −30 % und unter −2 Standardfehlern: {_band(info['delta'], info['se'])} %"),
        ]
    if name == "Kleines Fenster":
        return [
            (_below(sm["delta"], sm["se"], SMALL_SMOOTH_MAX), f"Glättung: Summe Lager höchstens −40 % und unter −2 Standardfehlern: {_band(sm['delta'], sm['se'])} %"),
            (_below(info["delta"], info["se"], SMALL_INFO_MAX), f"Info teilen: Summe Lager höchstens −40 % und unter −2 Standardfehlern: {_band(info['delta'], info['se'])} %"),
        ]
    if name == "Großes Fenster":
        return [
            (abs(info["delta"]) < LARGE_ABS_MAX, f"|Änderung durch Info teilen| unter 3 %: {_band(info['delta'], info['se'])} %"),
            (abs(sm["delta"]) < LARGE_ABS_MAX, f"|Änderung durch Glättung| unter 3 %: {_band(sm['delta'], sm['se'])} %"),
        ]
    raise KeyError(name)


def live_changes(live):
    """Änderung der Summe Lager gegenüber der Basis in % je Maßnahme auf dem gezeigten Verlauf, dazu das Verhältnis der Stufe-4-Verstärkung (Maßnahme / Basis)."""
    base = live["measures"][C.M_BASIS]
    out = {}
    for m, v in live["measures"].items():
        out[m] = {"delta": 100.0 * (v["total_onhand"] / base["total_onhand"] - 1.0), "amp_ratio": v["amp"][-1] / base["amp"][-1]}
    return out


def day_criteria(name, live):
    """Qualitative Kriterien am gezeigten Verlauf `live` (bw_live.solve_live): ein Verlauf streut, deshalb nur die Richtung der Geschichte. Liste (erfüllt, Text)."""
    x = live_changes(live)
    sm, info = x[C.M_SMOOTH], x[C.M_INFO]
    if name == "Standard":
        return [(sm["amp_ratio"] <= DAY_STANDARD_AMP_RATIO_MAX, f"Verstärkung der Fabrik mit Glättung höchstens halb so groß: Faktor {_de(sm['amp_ratio'], 2)}"),
                (sm["delta"] < 0, f"Glättung senkt die Summe Lager: {_de(sm['delta'], 1, True)} %"),
                (info["delta"] <= DAY_STANDARD_INFO_MAX, f"Info teilen spart mindestens 10 %: {_de(info['delta'], 1, True)} %")]
    if name == "Kurze Lieferzeit":
        return [(abs(sm["delta"]) < DAY_SHORT_SMOOTH_ABS_MAX, f"|Änderung durch Glättung| unter 15 %: {_de(sm['delta'], 1, True)} %"),
                (DAY_SHORT_INFO_MIN < info["delta"] < 0, f"Info teilen spart etwas, aber weniger als 20 %: {_de(info['delta'], 1, True)} %")]
    if name == "Träger Bedarf":
        return [(sm["delta"] > DAY_SLUGGISH_SMOOTH_MIN, f"Glättung kostet Kettenbestand: {_de(sm['delta'], 1, True)} %"),
                (info["delta"] <= DAY_SLUGGISH_INFO_MAX, f"Info teilen spart mindestens 20 %: {_de(info['delta'], 1, True)} %")]
    if name == "Kleines Fenster":
        return [(sm["delta"] <= DAY_SMALL_MAX, f"Glättung spart mindestens 25 %: {_de(sm['delta'], 1, True)} %"),
                (info["delta"] <= DAY_SMALL_MAX, f"Info teilen spart mindestens 25 %: {_de(info['delta'], 1, True)} %")]
    if name == "Großes Fenster":
        return [(abs(sm["delta"]) < DAY_LARGE_ABS_MAX, f"|Änderung durch Glättung| unter 10 %: {_de(sm['delta'], 1, True)} %"),
                (abs(info["delta"]) < DAY_LARGE_ABS_MAX, f"|Änderung durch Info teilen| unter 10 %: {_de(info['delta'], 1, True)} %")]
    raise KeyError(name)
