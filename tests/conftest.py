"""Projektwurzel und tests/ auf den Importpfad, damit `pytest tests/` auch ohne `python -m` die bw_-Module und die Test-Hilfsmodule (bw_checks, bw_frozen) findet.
Die CI installiert immer das neueste NumPy (Ströme von default_rng dürfen sich ändern): Referenzwerte laufen deshalb über EINGEFRORENE Nachfragereihen (tests/data), Tests von gen_ar1
prüfen nur Eigenschaften mit Toleranz (Standardfehler), und die statistischen Prüfungen der Messreihe (bw_checks) hängen an keinem einzelnen Seed."""
import pathlib
import sys

sys.dont_write_bytecode = True
TESTS = str(pathlib.Path(__file__).resolve().parent)
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
for p in (ROOT, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
