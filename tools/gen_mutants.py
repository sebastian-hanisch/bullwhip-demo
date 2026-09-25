"""Erzeugt die Mutantenliste für tools/mutation_check.py: maschinell aus dem Quelltext (Vergleichsoperatoren, Plus/Minus, and/or, True/False, min/max, Zahlen +1 bzw. +10 %),
stichprobenartig über alle Module verteilt (feste Zufallszahlen), jeweils mit gerade so viel Umgebung, dass die Stelle im Quelltext eindeutig ist. Die Liste wird einmal erzeugt und in
mutation_check.py als Literal eingetragen (so bleibt der Lauf reproduzierbar, auch wenn sich der Quelltext später ändert; test_tools prüft, dass jede Stelle noch genau einmal vorkommt).

Aufruf (im Projektordner):  python tools/gen_mutants.py > tools/_out/mutants.txt"""
import io
import pathlib
import random
import sys
import tokenize

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modul -> Zahl der Mutanten (Stichprobe; None = alle)
TARGETS = {
    "bw_model.py": None, "bw_theory.py": None, "bw_live.py": None, "bw_results.py": 260, "bw_stories.py": 90, "bw_presets.py": 40, "bw_constants.py": 40, "bw_format.py": None,
    "bw_ui_panel.py": 130, "bw_visualization.py": 60, "bw_pdf_export.py": 20,
    "tools/build_results.py": 30, "tools/tune_presets.py": 25, "tools/freeze_reference.py": 30, "tools/check_full.py": 15, "tools/quick_check.py": 20,
}
NO_NUMBERS = {"bw_visualization.py", "bw_pdf_export.py"}            # Ränder, Höhen, Deckkraft, Zeilenhöhen: Darstellung ohne Verhaltensänderung
SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==", "+": "-", "-": "+", "+=": "-=", "-=": "+=", "and": "or", "or": "and", "True": "False", "False": "True",
        "min": "max", "max": "min"}


def candidates(source):
    """(Offset, Länge, Ersatz) aller mutierbaren Token."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    out = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for i, tok in enumerate(tokens):
        (row, col), text = tok.start, tok.string
        off = starts[row - 1] + col
        if tok.type == tokenize.OP and text in SWAP:
            prev = tokens[i - 1] if i else None
            if text in ("-", "+") and prev is not None and prev.type == tokenize.OP and prev.string not in (")", "]", "}"):
                continue                                               # unäres Vorzeichen: nicht als Rechenoperator mutieren
            out.append((off, len(text), SWAP[text]))
        elif tok.type == tokenize.NAME and text in SWAP:
            if text in ("min", "max") and not (i + 1 < len(tokens) and tokens[i + 1].string == "("):
                continue
            out.append((off, len(text), SWAP[text]))
        elif tok.type == tokenize.NUMBER:
            try:
                if text.isdigit():
                    out.append((off, len(text), str(int(text) + 1)))
                else:
                    out.append((off, len(text), repr(round(float(text) * 1.1 + (0.1 if float(text) == 0 else 0.0), 6))))
            except ValueError:
                continue
    return out


def unique_window(source, off, length, new):
    """Kürzeste Umgebung um die Stelle, die im Quelltext genau einmal vorkommt: (alt, neu) oder None."""
    for k in range(6, 90, 4):
        a, b = max(0, off - k), min(len(source), off + length + k)
        old = source[a:b]
        if "\n" in old:
            lo = old.rfind("\n", 0, off - a)                             # nicht über Zeilengrenzen hinweg (der Test vergleicht LF-Quelltext, die Zeile genügt)
            hi = old.find("\n", off - a + length)
            old = old[lo + 1: hi if hi != -1 else len(old)]
            a = a + lo + 1
        if source.count(old) == 1 and len(old.strip()) >= 4:
            return old, old[:off - a] + new + old[off - a + length:]
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")                   # die Mutanten enthalten Umlaute und Pfeile
    rng = random.Random(20260925)
    result = []
    for name, count in TARGETS.items():
        source = (ROOT / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
        cands = candidates(source)
        if name in NO_NUMBERS:
            cands = [c for c in cands if not source[c[0]:c[0] + c[1]][0].isdigit()]
        rng.shuffle(cands)
        chosen = []
        seen = set()
        for off, length, new in cands:
            w = unique_window(source, off, length, new)
            if w is None or w in seen:
                continue
            seen.add(w)
            chosen.append((name, *w))
            if count is not None and len(chosen) >= count:
                break
        result += chosen
    print("MUTANTS = [")
    for item in result:
        print("    " + repr(item) + ",")
    print("]")
    print(f"# {len(result)} Mutanten", file=sys.stderr)


if __name__ == "__main__":
    main()
