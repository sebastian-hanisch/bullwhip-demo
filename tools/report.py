"""Textbericht aus sweep_data.json (alle Zahlen der ERGEBNIS.md stammen aus dieser Ausgabe). Mechanisch aus messreihe_bullwhip/report.py übernommen (nur die Pfade sind angepasst).

Aufruf im Projektordner:  python tools/report.py [sweep_data.json]   (Standard tools/_out/sweep_data.json; schreibt sweep_report.txt daneben)"""
import json
import pathlib
import sys

sys.dont_write_bytecode = True
PATH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / "_out" / "sweep_data.json"
D = json.load(open(PATH, encoding="utf-8"))
out = []


def w(s=""):
    out.append(s)


f2 = lambda xs: " ".join(f"{x:7.2f}" for x in xs)
J = {j["name"]: j for j in D["jobs"]}

w("=== E1 lineare Theorie (z=0, ohne Abschneidung, iid): je Stufe 1-4: Simulation | exakt | Chen-Produktformel (Untergrenze) | Info geteilt sim | exakt")
for r in D["E1"]:
    w(f"m={r['m']} p={r['p']:2d}: sim {f2(r['sim'])} | exakt {f2(r['exact'])} | Produkt {f2(r['chen_product'])} | exakt/Produkt Stufe4 {r['exact'][3] / r['chen_product'][3]:.2f} | geteilt sim {f2(r['share_sim'])} | exakt {f2(r['share_exact'])}")
w()
w("=== E2 realistische Kette: Stufen 1-4 Verstaerkung (Var(q)/Var(D)); lineare Theorie | z=0 mit Abschneidung | z=1,645 mit Abschneidung | z=1,645 ohne Abschneidung | Anteil Bestellungen = 0 (z=1,645)")
for r in D["E2"]:
    w(f"cv={r['cv']} m={r['m']} p={r['p']:2d}: lin {f2(r['linear_exact'])} | z0 {f2(r['z0_nonneg']['amp'])} | z1.645 {f2(r['z1645_nonneg']['amp'])} | z1.645 lin {f2(r['z1645_linear']['amp'])} | zero% {' '.join(f'{100 * x:4.1f}' for x in r['z1645_nonneg']['zero_share'])}")
w()
w("=== E3 exakte lineare Theorie, Stufe 4 (und Stufe 1): lokal | geteilte Bedarfsinformation | Verhaeltnis lokal/geteilt")
for rho in (0.0, 0.3, 0.6, 0.9):
    w(f"-- rho={rho}: Zeilen m, Spalten p=3,5,10,20,40: Stufe1 | Stufe4 lokal / Stufe4 geteilt (Verhaeltnis)")
    for m in (1, 2, 4, 8):
        rows = [r for r in D["E3"] if r["rho"] == rho and r["m"] == m]
        w(f"m={m}: " + "  ".join(f"p{r['p']}: {r['local'][0]:.2f} | {r['local'][3]:.1f}/{r['share'][3]:.1f} ({r['local'][3] / r['share'][3]:.1f}x)" for r in rows))


def es(j, key="equal_service"):
    return j[key]


def line(name, key="equal_service"):
    j = J[name]
    e = j[key]
    zs = j.get("z")
    s = f"{name:44s} "
    if key == "equal_service":
        s += f"z {' '.join(f'{z:4.2f}' for z in zs)}{'*' if any(j['z_at_cap']) else ' '}| "
    s += f"amp {' '.join(f'{a[0]:6.2f}' for a in e['amp'])} | beta {' '.join(f'{100 * b:5.1f}' for b in e['beta'])} | on-hand {' '.join(f'{x:6.0f}' for x in e['onhand'])} | Summe {e['total_onhand']:6.0f}+-{e['total_onhand_se']:.1f} | Kosten {e['total_cost']:6.0f}"
    return s


w()
w("=== E4 Hauptszenario (cv 0,2, rho 0, L=2 je Stufe, p=10, 4 Stufen), GLEICHER Servicegrad beta=95 % je Stufe (z kalibriert, out-of-sample)")
for name in [n for n in J if n.startswith("main/")]:
    w(line(name))
w()
w("=== E4b dieselben Politiken bei FESTEM z=1,645 (kein Nachkalibrieren): Servicegrad faellt")
for name in [n for n in J if n.startswith("main/")]:
    w(line(name, "fixed_z"))
w()
w("=== Gewinner/Verlierer: Lager je Stufe relativ zur Basis (gleicher Servicegrad), in %")
b = J["main/basis"]["equal_service"]["onhand"]
for name in [n for n in J if n.startswith("main/") and n != "main/basis"]:
    o = J[name]["equal_service"]["onhand"]
    w(f"{name:44s} " + " ".join(f"{100 * (x / y - 1):+6.1f}" for x, y in zip(o, b)) + f" | Summe {100 * (sum(o) / sum(b) - 1):+6.1f}")
w()
w("=== E4c Wiederholung mit anderen Seeds (main2): Summe Lager, Verstaerkung Stufe 4, Rangfolge; Vergleich mit main")
names = [n[len("main/"):] for n in J if n.startswith("main/")]
import numpy as _np
a1 = [J["main/" + n]["equal_service"]["total_onhand"] for n in names]
a2 = [J["main2/" + n]["equal_service"]["total_onhand"] for n in names]
for n, x, y in zip(names, a1, a2):
    w(f"{n:40s} Summe main {x:6.0f} | main2 {y:6.0f} | Diff {100 * (y / x - 1):+5.1f} % | amp4 {J['main/' + n]['equal_service']['amp'][3][0]:6.2f} | {J['main2/' + n]['equal_service']['amp'][3][0]:6.2f}")
def rank(v):
    o = _np.argsort(v); r_ = _np.empty(len(v)); r_[o] = _np.arange(len(v)); return r_
w(f"Rangkorrelation (Spearman) der Summe Lager zwischen main und main2: {float(_np.corrcoef(rank(a1), rank(a2))[0, 1]):.3f}")
amp4 = [J["main/" + n]["equal_service"]["amp"][3][0] for n in names]
w(f"Rangkorrelation Verstaerkung Stufe 4 gegen Summe Lager (13 Politiken, main): {float(_np.corrcoef(rank(amp4), rank(a1))[0, 1]):.3f}; ohne die zwei Politiken mit Aenderung von L/p: {float(_np.corrcoef(rank([a for n, a in zip(names, amp4) if not n.startswith(('L_', 'Fenster'))]), rank([a for n, a in zip(names, a1) if not n.startswith(('L_', 'Fenster'))]))[0, 1]):.3f}")
w("Gewinner/Verlierer auch in main2 (Summe Lager relativ zur Basis, in %): " + "; ".join(f"{n}: {100 * (y / a2[0] - 1):+.1f}" for n, y in zip(names[1:], a2[1:])))
w()
w("=== E5 Regime-Landkarten (gleicher Servicegrad): Summe Lager je Politik und Verhaeltnisse zur Basis")
for grp, names in (("rho", ["main/basis", "rho0.5/basis", "rho0.8/basis"]),):
    pass
for label, sel in (("rho", [("0.0", "main"), ("0.5", "rho0.5"), ("0.8", "rho0.8")]), ("L", [("1", "L1"), ("2", "main"), ("4", "L4")]),
                   ("p", [("6", "p6"), ("10", "main"), ("20", "p20"), ("40", "p40")]), ("cv", [("0.1", "cv0.1"), ("0.2", "main"), ("0.3", "cv0.3")])):
    w(f"-- {label}")
    for val, pre in sel:
        row = []
        for pol in ("basis", "glaettung_0.5", "info_teilen"):
            j = J[f"{pre}/{pol}"]
            e = j["equal_service"]
            row.append(f"{pol}: Summe {e['total_onhand']:6.0f} amp4 {e['amp'][3][0]:6.2f} z {j['z'][0]:.2f}..{j['z'][3]:.2f}{'*' if any(j['z_at_cap']) else ''}")
        bb = J[f"{sel[0][1]}/basis"] if False else None
        w(f"{label}={val}: " + " | ".join(row))
w()
w("Hinweis: * = mindestens ein z am Suchgrenzwert 10 (Servicegrad-Ziel nicht erreichbar oder nur mit extremem z)")
w("Achieved beta (out-of-sample) minimal/maximal je Job und in-sample:")
for j in D["jobs"]:
    e = j["equal_service"]
    w(f"{j['name']:40s} out {min(e['beta']):.4f}..{max(e['beta']):.4f} in {min(j['insample_beta']):.4f}..{max(j['insample_beta']):.4f}")
w()
w("=== E6 Niveauspruenge (rate 1/50, +-20 %): Fenster p gegen Summe Lager bei gleichem Servicegrad (beta=95 %)")
for kind in ("ar1", "shift"):
    row = []
    for p in (4, 8, 16, 32, 64):
        j = J[f"{kind}/p{p}/basis"]
        e = j["equal_service"]
        row.append(f"p{p}: {e['total_onhand']:6.0f} (amp4 {e['amp'][3][0]:6.1f}, z {j['z'][0]:.2f}..{j['z'][3]:.2f}{'*' if any(j['z_at_cap']) else ''})")
    w(f"{kind}: " + " | ".join(row))
w("fest z=1,645: " + "; ".join(f"{kind} p{p}: beta1 {100 * J[f'{kind}/p{p}/basis']['fixed_z']['beta'][0]:.1f}" for kind in ('ar1', 'shift') for p in (4, 8, 16, 32, 64)))
open(PATH.parent / "sweep_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
