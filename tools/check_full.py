"""Das VOLLE Bau-Gate: alle 47 Korrektheits-Checks der Messreihe (messreihe_bullwhip/check.py) in voller Größe mit den bw_-Modulen, dazu die Wiederholung des GANZEN Sweeps (E1, E2, E3 und alle 63
Kalibrierungsjobs) gegen sweep_data.json der Messreihe: jede Zahl (z je Stufe, Verstärkung, Summe Lager, Servicegrad, Kosten ...) muss BITGLEICH stimmen, und die Ergebnisdatei
data/bw_results.json muss genau die Ausgabe von tools/build_results.py sein. Läuft einmal lokal vor dem Commit (etwa 2,5 Minuten), nicht in der CI: dort laufen dieselben Checks in
tests/test_checks.py und die Bitgleichheit auf EINGEFRORENEN Nachfragereihen in tests/test_frozen_reference.py (die CI installiert immer das neueste NumPy, und die Ströme von
default_rng dürfen sich zwischen Versionen ändern).

Die Zahlen der Wiederholung gelten für die NumPy-Version, mit der die Messreihe gerechnet wurde; mit einer anderen Version können sich die Zufallsströme ändern (die Invarianten nicht). Der Lauf
meldet die Version und verlangt die Bitgleichheit nur, wenn sie übereinstimmt (sonst nur die Toleranz 1e-6 relativ für die Aggregate).

Aufruf (im Projektordner):  python tools/check_full.py [Ordner_der_Messreihe]   (Standard ../bestand-planung/messreihe_bullwhip)"""
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests", ROOT / "tools"):
    sys.path.insert(0, str(p))

import bw_checks as K  # noqa: E402
import build_results as BR  # noqa: E402
import sweep as SW  # noqa: E402

DEFAULT_SOURCES = ROOT.parent / "bestand-planung" / "messreihe_bullwhip"
RESULTS_PATH = ROOT / "data" / "bw_results.json"
MEASURED_WITH_NUMPY = "2.5.3"


def say(text):
    print(text, flush=True)


def flatten(obj, prefix=""):
    """Alle Zahlen eines verschachtelten dict/list als {Pfad: Wert}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def main(argv):
    src = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCES
    t0 = time.time()
    stream_ok = np.__version__ == MEASURED_WITH_NUMPY
    say(f"NumPy {np.__version__} ({'wie bei der Messreihe: Bitgleichheit wird verlangt' if stream_ok else 'anders als bei der Messreihe: nur Aggregate mit Toleranz'})")

    # 1. die 47 Checks in voller Größe
    t = time.time()
    failed = []
    for name, fn, args in K.ALL_CHECKS:
        ok, detail = fn(*args)
        if not ok:
            failed.append((name, detail))
    say(f"1. Korrektheits-Checks der Messreihe: {len(K.ALL_CHECKS) - len(failed)} von {len(K.ALL_CHECKS)} bestanden ({time.time() - t:.0f} s)")
    assert not failed, failed
    assert len(K.ALL_CHECKS) == 47

    # 2. der ganze Sweep gegen sweep_data.json
    ref_path = src / "sweep_data.json"
    if not ref_path.exists():
        say("2. Wiederholung des Sweeps: übersprungen (sweep_data.json der Messreihe nicht gefunden).")
        return 0
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    t = time.time()
    got = json.loads(json.dumps(SW.run_all(min(12, mp.cpu_count()), lambda s: say("   " + s))))
    fr, fg = flatten(ref), flatten(got)
    assert set(fr) == set(fg), (len(fr), len(fg))
    bad = [k for k in fr if fr[k] != fg[k]]
    close = [k for k in bad if isinstance(fr[k], float) and abs(fr[k] - fg[k]) <= 1e-6 * max(1.0, abs(fr[k]))]
    say(f"2. Wiederholung des GANZEN Sweeps mit den bw_-Modulen (E1, E2, E3, {len(ref['jobs'])} Kalibrierungsjobs; {len(fr)} Zahlen) gegen sweep_data.json: {len(bad)} Zahlen abweichend "
        f"({len(close)} davon innerhalb 1e-6 relativ), {time.time() - t:.0f} s")
    zs = [(j["name"], j["z"]) for j in got["jobs"]]
    ref_zs = [(j["name"], j["z"]) for j in ref["jobs"]]
    say(f"   z je Stufe: {sum(1 for a, b in zip(zs, ref_zs) if a == b)} von {len(zs)} Jobs bitgleich")
    if stream_ok:
        assert not bad, bad[:5]
    else:
        assert len(bad) == len(close), [k for k in bad if k not in close][:5]

    # 3. die Ergebnisdatei
    out = BR.dumps(BR.build(got if stream_ok else ref))
    file_text = RESULTS_PATH.read_text(encoding="utf-8")
    say(f"3. data/bw_results.json == Ausgabe von tools/build_results.py aus der wiederholten Messreihe: {out == file_text}")
    if stream_ok:
        assert out == file_text
    say(f"\nalle Checks bestanden ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main(sys.argv[1:]))
