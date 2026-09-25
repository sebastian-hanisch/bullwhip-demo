"""Wie grob ist die Schnellkalibrierung? (AP 0 c) Für die drei Maßnahmen Basis, Glättung γ 0,5 und Info teilen im Hauptszenario wird z je Stufe mit der Schnellkalibrierung der App (20 Wiederholungen ×
700 Perioden, ein Durchlauf) für mehrere Seeds bestimmt und gegen die volle Kalibrierung der Messreihe (200 × 1200, zwei Durchläufe) gelegt: Abweichung von z, der Servicegrad, den das schnelle z auf
einer unabhängigen Stichprobe (300 × 1200) tatsächlich erreicht, und die Summe Lager. Ergebnis im README (Abschnitt „Ehrliche Grenzen“).

Aufruf (im Projektordner):  python tools/quick_check.py [Zahl der Seeds] [--jobs N]   (Standard 12 Seeds, 12 Prozesse; etwa 40 Sekunden)"""
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bw_constants as C  # noqa: E402
import bw_live as LV  # noqa: E402
import bw_model as M  # noqa: E402
import bw_results as R  # noqa: E402

MEASURES = {"Basis": (C.M_BASIS, "basis"), "Glättung γ 0,5": (C.M_SMOOTH, "glaettung_0.5"), "Info teilen": (C.M_INFO, "info_teilen")}
EVAL_OFFSET = 200000                                            # unabhängige Stichprobe: Seed + Offset
EVAL_R, EVAL_T = 300, 1200


def one(task):
    label, seed = task
    measure = MEASURES[label][0]
    t = time.time()
    z = LV.quick_calibrate(0.2, 0.0, 2, 10, 0.5, seed, [measure])["z"][measure]
    seconds = time.time() - t
    Ls, p, kw = LV.measure_setup(measure, 0.5, 2, 10)
    D = M.gen_ar1(np.random.default_rng(seed + EVAL_OFFSET), EVAL_R, EVAL_T, C.MU, C.MU * 0.2, 0.0)
    q, _, _ = M.order_streams(D, Ls, p, z=z, **kw)
    ph = M.physical(D, q, Ls, burn=C.BURN)
    return label, seed, z, [float(b) for b in ph["beta"].mean(axis=1)], float(ph["onhand"].sum(axis=0).mean()), seconds


def summarize(results, data):
    """Je Maßnahme: mittleres z und Standardabweichung über die Seeds, Abweichung vom vollen z, Servicegrad (Mittel, kleinster, größter Wert) und Summe Lager gegen die Messreihe."""
    out = {}
    for label, (_, job) in MEASURES.items():
        rows = [r for r in results if r[0] == label]
        Z, B, T = np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
        full = data["jobs"][f"main/{job}"]
        out[label] = {"n": len(rows), "z_mean": Z.mean(axis=0).tolist(), "z_sd": Z.std(axis=0, ddof=1).tolist(), "z_full": full["z"], "z_diff": (Z.mean(axis=0) - np.array(full["z"])).tolist(),
                      "z_max_abs_diff": float(np.abs(Z - np.array(full["z"])).max()), "beta_mean": B.mean(axis=0).tolist(), "beta_min": float(B.min()), "beta_max": float(B.max()),
                      "total_mean": float(T.mean()), "total_sd": float(T.std(ddof=1)), "total_full": full["equal"]["total_onhand"], "seconds": float(np.mean([r[5] for r in rows]))}
    return out


def main(argv):
    jobs = 12
    if "--jobs" in argv:
        i = argv.index("--jobs")
        jobs = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    n = int(argv[0]) if argv else 12
    tasks = [(label, 300 + s) for label in MEASURES for s in range(n)]
    with mp.Pool(jobs) as pool:
        results = pool.map(one, tasks)
    data = R.load_results()
    f = lambda xs: " ".join(f"{x:.2f}" for x in xs)
    for label, s in summarize(results, data).items():
        print(f"== {label} ({s['n']} Seeds): volle Kalibrierung z {f(s['z_full'])}; Schnellkalibrierung z {f(s['z_mean'])} (Standardabweichung {f(s['z_sd'])}); Mittel-Abweichung {f(s['z_diff'])}; größte Einzel-Abweichung {s['z_max_abs_diff']:.2f}")
        print(f"   Servicegrad auf unabhängiger Stichprobe (Ziel 95 %): Mittel {' '.join(f'{100 * b:.1f}' for b in s['beta_mean'])} %, kleinster {100 * s['beta_min']:.1f} %, größter {100 * s['beta_max']:.1f} %")
        print(f"   Summe Lager {s['total_mean']:.0f} (Standardabweichung {s['total_sd']:.1f}) gegen {s['total_full']:.0f} der Messreihe; {s['seconds']:.1f} s je Kalibrierung (parallel)")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main(sys.argv[1:]))
