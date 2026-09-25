"""Preset-Abstimmung: sucht die Anzeige-Seeds der Beispielszenarien. Der Seed bestimmt nur den GEZEIGTEN Verlauf (alle Nachfrage-Seeds der Messreihe liegen bei 10000 und darüber),
deshalb wird nur im Bereich 200 bis 299 gesucht. Für jeden Seed werden die fünf Presets auf dem echten Verlauf gerechnet und die qualitativen Kriterien aus bw_stories.day_criteria geprüft;
ausgegeben werden die Seeds, die alle fünf erfüllen, mit der Änderung der Summe Lager durch Glättung im Standard (typisch = nah am Mittel der Messreihe).

Aufruf (im Projektordner):  python tools/tune_presets.py [von bis] [--jobs N]   (Standard 200 300, 1 Prozess)"""
import multiprocessing as mp
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bw_constants as C  # noqa: E402
import bw_live as LV  # noqa: E402
import bw_results as R  # noqa: E402
import bw_stories as ST  # noqa: E402


def solve_preset(p, seed, demand=None):
    return LV.solve_live(p["cv"], p["rho"], p["lead"], p["window"], p["gamma"], p["zmode"], seed, demand=demand)


def evaluate(seed, presets=None):
    """Ergebnis je Preset auf dem Verlauf `seed`: (alle Kriterien erfüllt, Liste der Texte, Änderung der Summe Lager durch Glättung in %)."""
    out = {}
    for name, p in (presets or C.PRESETS).items():
        live = solve_preset(p, seed)
        crit = ST.day_criteria(name, live)
        out[name] = (all(ok for ok, _ in crit), crit, ST.live_changes(live)[C.M_SMOOTH]["delta"])
    return out


def _one(seed):
    res = evaluate(seed)
    return seed, "".join("+" if ok else "-" for ok, _, _ in res.values()), res["Standard"][2]


def main(argv):
    jobs = 1
    if "--jobs" in argv:
        i = argv.index("--jobs")
        jobs = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    lo, hi = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (200, 300)
    data = R.load_results()
    ref = R.resolve_pair(data, C.M_SMOOTH, C.GAMMA_DEFAULT, C.CV_DEFAULT, C.RHO_DEFAULT, C.LEAD_DEFAULT, C.WINDOW_DEFAULT)["delta"]
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            rows = pool.map(_one, range(lo, hi))
    else:
        rows = [_one(s) for s in range(lo, hi)]
    good = []
    for seed, flags, std in rows:
        print(f"Seed {seed}: {flags}  Änderung Summe Lager durch Glättung im Standard {std:+.1f} % (Messreihe Mittel {ref:+.1f} %)", flush=True)
        if "-" not in flags:
            good.append((abs(std - ref), seed))
    good.sort()
    print("\nSeeds, die alle fünf Tageskriterien erfüllen (nach Nähe zum Mittel der Messreihe):", [s for _, s in good])
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main(sys.argv[1:]))
