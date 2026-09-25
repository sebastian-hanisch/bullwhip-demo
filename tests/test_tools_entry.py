"""Die Kommandozeilen-Einstiege der Werkzeuge (`python tools/<name>.py ...`) im Unterprozess: die Argumente müssen ankommen (sys.argv[1:])."""
import json
import pathlib
import shutil
import subprocess
import sys

from test_tools import ROOT, synthetic_sweep


def test_build_results_command_line_takes_the_source_argument(tmp_path):
    proj = tmp_path / "proj"
    (proj / "tools").mkdir(parents=True)
    (proj / "data").mkdir()
    shutil.copy(ROOT / "tools" / "build_results.py", proj / "tools" / "build_results.py")
    src = tmp_path / "src"
    src.mkdir()
    (src / "sweep_data.json").write_text(json.dumps(synthetic_sweep()), encoding="utf-8")
    r = subprocess.run([sys.executable, str(proj / "tools" / "build_results.py"), str(src)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "geschrieben:" in r.stdout, r.stdout + r.stderr
    assert json.loads((proj / "data" / "bw_results.json").read_text(encoding="utf-8"))["_meta"]["n_jobs"] == 2
    bad = subprocess.run([sys.executable, str(proj / "tools" / "build_results.py"), str(tmp_path / "gibt-es-nicht")], capture_output=True, text=True, timeout=120)
    assert bad.returncode != 0


def test_tune_presets_command_line_takes_the_seed_range():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "tune_presets.py"), "200", "201"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "Seed 200:" in r.stdout and "Seed 201:" not in r.stdout
