"""Die 47 Korrektheits-Checks der Messreihe (check.py) in voller Größe (etwa 9 s): lineare Theorie == Simulation, welche Formel passt, unabhängige Faltung, Handinstanz, Bestandspositions-
Identität und Warenerhaltung, Einstufig gegen Newsvendor-Analytik, Zweig-Tests, Kalibrierung, Wiederholbarkeit. Ausgeführt mit bw_model / bw_theory. Die statistischen Checks laufen mit
Toleranz (4 Standardfehler + 1 %) und hängen an keinem einzelnen Seed; die exakten Identitäten sind unabhängig vom Zufallsstrom (siehe bw_checks.py)."""
import os
import subprocess
import sys

import pytest

import bw_checks as K

IDS = [name for name, _, _ in K.ALL_CHECKS]


def test_there_are_exactly_47_checks_as_in_the_measurement_series():
    assert len(K.ALL_CHECKS) == 47 and len(set(IDS)) == 47


@pytest.mark.parametrize("name,fn,args", K.ALL_CHECKS, ids=IDS)
def test_check(name, fn, args):
    ok, detail = fn(*args)
    assert ok, f"{name}: {detail}"


@pytest.mark.skipif(os.environ.get("BW_MUTATION_RUN") == "1", reason="in Mutantenkopien nicht nötig")
def test_the_statistical_checks_do_not_hang_on_one_seed():
    """Die statistischen Checks noch einmal mit verschobenen Seeds (Unterprozess): sie dürfen an keinem einzelnen Seed hängen (CI-Falle: die Zufallsströme können sich ändern)."""
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import bw_checks as K; "
            "bad = [n for n, f, a in K.ALL_CHECKS if n[0] in '12678' and not f(*a)[0]]; print(bad); sys.exit(1 if bad else 0)") % (os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__))
    env = dict(os.environ, BW_CHECK_SEED_SHIFT="1000", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
