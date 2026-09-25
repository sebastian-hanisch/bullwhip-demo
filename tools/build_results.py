"""Erzeugt data/bw_results.json aus der Messreihe (nur lesend): sweep_data.json (E1 lineare Theorie gegen Simulation, E2 realistische Kette, E3 lineare Karte, 63 Kalibrierungsjobs mit z je Stufe
und den Kennzahlen bei gleichem Servicegrad und bei festem z). Nur das, was die App zeigt: keine Rohdaten je Wiederholung, Zahlen auf 6 gültige Stellen gerundet, feste Schlüsselreihenfolge,
deterministische Ausgabe.

Aufruf (im Projektordner):  python tools/build_results.py [Quellordner-oder-sweep_data.json]   (Standard ../bestand-planung/messreihe_bullwhip/sweep_data.json)"""
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "bw_results.json"
DEFAULT_SOURCE = ROOT.parent / "bestand-planung" / "messreihe_bullwhip" / "sweep_data.json"
SIGNIFICANT = 6

STAT_KEYS = ("beta", "onhand", "backlog", "cost", "zero_share")           # je Stufe
TOTAL_KEYS = ("total_onhand", "total_onhand_se", "total_cost", "total_cost_se")


def rnd(x, digits=SIGNIFICANT):
    """Rekursiv auf gültige Stellen runden; bool, None und Text bleiben, Tupel werden Listen."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, int):
        return x
    if isinstance(x, dict):
        return {k: rnd(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, digits) for v in x]
    return x


def build_stats(block):
    """Kennzahlen eines Kalibrierlaufs (gleicher Servicegrad oder fester z): Verstärkung je Stufe als [Mittel, Standardfehler] (ohne Median), dazu die Stufenwerte und die Summen."""
    out = {"amp": [[a[0], a[1]] for a in block["amp"]]}
    for k in STAT_KEYS:
        out[k] = list(block[k])
    for k in TOTAL_KEYS:
        out[k] = block[k]
    return out


def build_job(job):
    out = {"scenario": {k: job["scenario"][k] for k in ("kind", "cv", "rho", "L", "p", "seed")}, "policy": job["policy"], "L": job["L"], "p": job["p"]}
    out["z"] = list(job["z"])
    out["z_at_cap"] = list(job["z_at_cap"])
    out["equal"] = build_stats(job["equal_service"])
    out["fixed"] = build_stats(job["fixed_z"]) if "fixed_z" in job else None
    return out


def build_e2(rows):
    out = []
    for r in rows:
        row = {"cv": r["cv"], "m": r["m"], "p": r["p"], "linear_exact": r["linear_exact"]}
        for k in ("z0_nonneg", "z1645_nonneg", "z1645_linear"):
            row[k] = {"amp": r[k]["amp"], "se": r[k]["se"], "zero_share": r[k]["zero_share"]}
        out.append(row)
    return out


def build(sweep):
    cfg = sweep["config"]
    out = {"_meta": {"config": cfg, "n_jobs": len(sweep["jobs"]), "source": "messreihe_bullwhip/sweep_data.json (sweep.py, 63 Kalibrierungsjobs)"},
           "E1": [dict(r) for r in sweep["E1"]], "E2": build_e2(sweep["E2"]), "E3": [dict(r) for r in sweep["E3"]],
           "jobs": {j["name"]: build_job(j) for j in sweep["jobs"]}}
    return rnd(out)


def dumps(obj):
    return json.dumps(obj, sort_keys=False, separators=(",", ":"), ensure_ascii=False) + "\n"


def main(argv):
    src = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCE
    if src.is_dir():
        src = src / "sweep_data.json"
    sweep = json.loads(src.read_text(encoding="utf-8"))
    out = build(sweep)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(dumps(out), encoding="utf-8", newline="\n")
    print(f"geschrieben: {OUT} ({OUT.stat().st_size // 1024} KB, {len(out['jobs'])} Jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
